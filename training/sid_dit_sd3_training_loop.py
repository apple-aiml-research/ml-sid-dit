#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""Distill DiT-based diffusion/flow-matching models using the SiD few-step techniques described in the
paper "Score Distillation of Flow"."""


"""Main training loop."""

import os
import re
import time
import copy
import json
import pickle
import psutil
from typing import Optional
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from metrics import sid_metric_main as metric_main

#from training.sid_dit_util import load_dit, sid_dit_generate, sid_dit_denoise

from training.sid_dit_sd3_util import load_dit, sid_dit_generate, sid_dit_denoise, encode_prompt    

from functools import partial
# Import ShardedGradScaler for gradient scaling with autocast+fp16.
# Note: By default, this code uses autocast with bfloat16 (bf16), so ShardedGradScaler is not actively used.
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import gc

from training.precompute_latents import (
    #process_and_save_latents_with_pipeline_multigpu,
    process_and_save_latents_with_pipeline_multigpu_sd3,
)

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    StateDictType,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)

# Data loading setup:
# Multiple dataloaders are supported:
#   - Training with text-only datasets
#   - Training with paired text+image datasets
#   - Evaluation mode (e.g., COCO2014 dataset for FID/CLIP metrics)
# The appropriate loader is selected based on the training/evaluation configuration.
from training.precompute_latents import Text2ImageDataset, TextDataset

# Suppress NCCL INFO messages
if 'NCCL_DEBUG' not in os.environ:
    os.environ['NCCL_DEBUG'] = 'WARN'
    
#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(3840 // training_set.resolution, 4, 32)
    gh = np.clip(2160 // training_set.resolution, 2, 32)

    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    images, contexts = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), contexts

from itertools import islice
def split_list(lst, split_sizes):
    """
    Splits a list into chunks based on split_sizes.

    Parameters:
    - lst (list): The list to be split.
    - split_sizes (list or int): Sizes of the chunks to split the list into. 
                                 If it's an integer, the list will be divided into chunks of this size.
                                 If it's a list of integers, the list will be divided into chunks of varying sizes specified by the list.

    Returns:
    - list of lists: The split list.
    """
    if isinstance(split_sizes, int):
        # If split_sizes is an integer, create a list of sizes to split the list evenly, except the last chunk which may be smaller.
        split_sizes = [split_sizes] * (len(lst) // split_sizes) + ([len(lst) % split_sizes] if len(lst) % split_sizes != 0 else [])
    it = iter(lst)
    return [list(islice(it, size)) for size in split_sizes]

#----------------------------------------------------------------------------
# Helper methods

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    N, C, H, W = img.shape
    expected = gh * gw

    if N < expected:
        pad = expected - N
        print(f"[save_image_grid] Padding with {pad} black images to reach {expected}")
        pad_img = np.zeros((pad, C, H, W), dtype=np.uint8)
        img = np.concatenate([img, pad_img], axis=0)
    elif N > expected:
        print(f"[save_image_grid] Trimming {N - expected} extra images")
        img = img[:expected]

    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)  # (gh, H, gw, W, C)
    img = img.reshape(gh * H, gw * W, C)

    from PIL import Image
    assert C in [1, 3], f"Unsupported channel count: {C}"
    if C == 1:
        Image.fromarray(img[:, :, 0], 'L').save(fname)
    else:
        Image.fromarray(img, 'RGB').save(fname)

def save_data(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def save_pt(pt, fname, dtype=None):
    """
    Save a PyTorch state dictionary to disk, converting all tensors to the specified dtype.
    Handles nested dictionaries and None values gracefully.

    Args:
        pt (dict): State dictionary to save.
        fname (str): File path to save the state dictionary.
        dtype (torch.dtype): Desired data type for tensors (default: torch.float16).
    """
    if dtype is None:
        torch.save(pt, fname)
    else:
        def convert_tensor(val):
            if torch.is_tensor(val):
                return val.to(dtype=dtype)
            return val

        def convert_dict(d):
            if d is None:
                return None
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            return convert_tensor(d)

        pt_converted = {key: convert_dict(value) for key, value in pt.items()}
        torch.save(pt_converted, fname)


def append_line(jsonl_line, fname):
    with open(fname, 'at') as f:
        f.write(jsonl_line + '\n')


import contextlib
@contextlib.contextmanager
def fsdp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync:
        yield
    else:
        with module.no_sync():
            yield


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
# def move_model_to_device(model, device):
#     if isinstance(model, torch.nn.Module):
#         if model.device != device:
#             model.to(device)
#     elif isinstance(model, list):
#         for m in model:
#             move_model_to_device(m, device)
#     elif isinstance(model, dict):   
#         for k, v in model.items():
#             model[k] = move_model_to_device(v, device)
#     return model


# @contextlib.contextmanager
# def temporary_model_on_device(model, device="cuda"):
#     """Temporarily move model to a device and move it back after use (if needed)."""
#     target_device = torch.device(device)
#     original_device = model.device

#     moved = original_device != target_device
#     if moved:
#         model.to(target_device)
#         torch.cuda.empty_cache()

#     try:
#         yield model
#     finally:
#         if moved:
#             model.to(original_device)
#             gc.collect()
#             if original_device.type == "cuda":
#                 torch.cuda.empty_cache()



@contextlib.contextmanager
def temporary_model_on_device(models, device="cuda"): #,precompute_latents=True):
    """
    Temporarily move model(s) to a device and move them back after use (if needed).
    Supports a single model, a list/tuple of models, or a dict of models.
    """
    num_nodes = int(os.environ['WORLD_SIZE']) // int(os.environ['LOCAL_WORLD_SIZE'])
    if 0: #not precompute_latents: #num_nodes > 1:
        yield models
        #pass
    else:
        import torch
        import gc

        target_device = torch.device(device)

        def get_device(m):
            return getattr(m, 'device', None)

        def move_to_device(m, dev):
            if hasattr(m, 'to'):
                m.to(dev)
            return m

        def move_models_to_device(models, dev):
            if isinstance(models, (list, tuple)):
                return [move_to_device(m, dev) for m in models]
            elif isinstance(models, dict):
                return {k: move_to_device(v, dev) for k, v in models.items()}
            else:
                return move_to_device(models, dev)

        def get_original_devices(models):
            if isinstance(models, (list, tuple)):
                return [get_device(m) for m in models]
            elif isinstance(models, dict):
                return {k: get_device(v) for k, v in models.items()}
            else:
                return get_device(models)

        def devices_differ(orig, tgt):
            if isinstance(orig, (list, tuple)):
                return any(o != tgt for o in orig)
            elif isinstance(orig, dict):
                return any(v != tgt for v in orig.values())
            else:
                return orig != tgt

        def move_back(models, orig_devices):
            if isinstance(models, (list, tuple)):
                for m, d in zip(models, orig_devices):
                    if d is not None and get_device(m) != d:
                        move_to_device(m, d)
            elif isinstance(models, dict):
                for k, v in models.items():
                    d = orig_devices[k]
                    if d is not None and get_device(v) != d:
                        move_to_device(v, d)
            else:
                if orig_devices is not None and get_device(models) != orig_devices:
                    move_to_device(models, orig_devices)

        original_devices = get_original_devices(models)
        moved = devices_differ(original_devices, target_device)
        if moved:
            move_models_to_device(models, target_device)
            torch.cuda.empty_cache()

        try:
            yield models
        finally:
            if moved:
                move_back(models, original_devices)
                gc.collect()
                # Try to clear CUDA cache for all original devices if any were CUDA
                def is_cuda(dev):
                    return hasattr(dev, 'type') and dev.type == "cuda"
                orig_devs = []
                if isinstance(original_devices, (list, tuple)):
                    orig_devs = original_devices
                elif isinstance(original_devices, dict):
                    orig_devs = list(original_devices.values())
                else:
                    orig_devs = [original_devices]
                if any(is_cuda(d) for d in orig_devs if d is not None):
                    torch.cuda.empty_cache()
    
from datasets import load_dataset, load_from_disk
from filelock import FileLock
def load_dataset_distributed_safe(
    file_path: str,
    actual_datapart: int | None = None,
    dataset_name: str = "processed_dataset",
    dataset_format: str = "parquet",
    split: str = "train",
    cache_latents_dir: str = None,
):
    """
    Safely loads a Hugging Face dataset across multiple FSDP ranks.
    Only rank 0 writes to disk. All ranks load from disk.
    """
    import re
    if actual_datapart is None:
        match = re.match(r"train_(\d+)\.parquet", file_path)
        if match:
            actual_datapart = int(match.group(1)) 
    # if cache_latents_dir is None:
    #     # Resolve path to parent of 'SiD' directory
    #     BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    #     cache_latents_dir = os.path.join(BASE_DIR, "data", "tmp")
    #     os.makedirs(cache_latents_dir, exist_ok=True)
    if cache_latents_dir is None:
        # Resolve path to parent of 'SiD' directory
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        # Use the last part of run_dir to name the cache directory
        run_name = os.path.basename(os.path.normpath(run_dir))
        cache_latents_dir = os.path.join(BASE_DIR, "data", "tmp", run_name)

    os.makedirs(cache_latents_dir, exist_ok=True)

    assert torch.distributed.is_initialized(), "torch.distributed must be initialized"
    rank = get_rank()

    shared_path = os.path.join(cache_latents_dir, f"{dataset_name}_{actual_datapart}")
    lock_path = shared_path + ".lock"

    if rank == 0:
        with FileLock(lock_path):
            if not os.path.exists(shared_path):
                print(f"[Rank 0] Loading and saving dataset to {shared_path}...")
                hf_dataset = load_dataset(
                    path=dataset_format,
                    data_files={split: file_path},
                    split=split,
                )
                hf_dataset.save_to_disk(shared_path)
            else:
                print(f"[Rank 0] Dataset already exists at {shared_path}")
    torch.distributed.barrier()
    hf_dataset = load_from_disk(shared_path)
    torch.distributed.barrier()
    return hf_dataset

       
#----------------------------------------------------------------------------

def stream_data_sampler(
    datapart,
    text_image_pair_path,
    cache_latents_dir,
    #dataset_name,
    #text_encoding_pipeline,
    text_encoders,
    tokenizers,
    vae,
    resolution,
    train_diffusiongan,
    uncond_embeds,
    #uncond_attention_mask,
    uncond_pooled_embeds,
    batch_gpu,
    seed,
    data_loader_kwargs,
    precomputed_latents_path = None,
    text_feature_dtype=None,
):
    """Define data samplers and iterators for training.
    
    Args:
        datapart: Current data partition number
        text_image_pair_path: Path to text-image pair dataset
        cache_latents_dir: Directory to cache latents
        precompute_latents: Whether to precompute latents
        dataset_name: Name of the dataset
        text_encoding_pipeline: Pipeline for text encoding
        vae: VAE model
        resolution: Image resolution
        train_diffusiongan: Whether training diffusion GAN
        uncond_embeds: Unconditional embeddings
        uncond_attention_mask: Unconditional attention mask
        batch_gpu: Batch size per GPU
        seed: Random seed
        data_loader_kwargs: Additional data loader arguments
        
    Returns:
        tuple: (dataset_latents_iterator, dataset_latents_iterator_text, hf_dataset)
    """

    
    dist.print0(f"Switching to new data part: {datapart}")
    actual_datapart = datapart % 124
    pt_file = f"train_{str(actual_datapart).zfill(3)}.parquet"
    file_path = os.path.join(text_image_pair_path, pt_file) 

    # Explicitly delete any previous dataset objects and run garbage collection to free GPU memory
    gc.collect()
    torch.cuda.empty_cache()


    # Only load hf_dataset if we need it (i.e., if we don't already have precomputed latents)
    hf_dataset = None
    if precomputed_latents_path is None or not os.path.exists(precomputed_latents_path):
        hf_dataset = load_dataset_distributed_safe(
            file_path=file_path,
            actual_datapart=actual_datapart,
            cache_latents_dir=cache_latents_dir,
        )
    if precomputed_latents_path is not None:
        dist.print0("precomputed_latents_path", precomputed_latents_path)
    else:
        dist.print0("precomputed_latents_path:None")
    if precomputed_latents_path is not None and os.path.exists(precomputed_latents_path) is not None:
        dist.print0("os.path.exists(precomputed_latents_path)", os.path.exists(precomputed_latents_path))
    else:
        dist.print0("os.path.exists(precomputed_latents_path):None")

    if precomputed_latents_path is not None and not os.path.exists(precomputed_latents_path):
        # rank_device = torch.device(f"cuda:{get_rank() % dist.get_world_size()}")
        # context_manager_vae = (
        #     temporary_model_on_device(vae, rank_device)
        #     if train_diffusiongan else
        #     contextlib.nullcontext(vae)
        # )
        # with temporary_model_on_device(text_encoding_pipeline, rank_device):
        #     with context_manager_vae:
        #process_and_save_latents_with_pipeline_multigpu(
        process_and_save_latents_with_pipeline_multigpu_sd3(
            hf_dataset=hf_dataset,
            #text_encoding_pipeline=text_encoding_pipeline,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            vae=vae,
            latents_path=precomputed_latents_path,
            resolution=resolution,
            world_size=dist.get_world_size(),
            rank=get_rank(),
            train_diffusiongan=train_diffusiongan,
            batch_size=int(batch_gpu*2.5) if train_diffusiongan else batch_gpu*12,
            uncond_embeds=uncond_embeds,
            uncond_pooled_embeds=uncond_pooled_embeds,
            use_sd3=True
        )
        # All ranks wait for the final file before proceeding
        torch.distributed.barrier()
        gc.collect()
        torch.cuda.empty_cache()

    dataset_latents_iterator = None
    dataset_latents_obj = None
    dataset_latents_sampler = None

    if train_diffusiongan:
        dataset_latents_obj = Text2ImageDataset(
            hf_dataset=hf_dataset,
            resolution=resolution,
            precomputed_latents_path=precomputed_latents_path,
            sd3=True,
            text_feature_dtype=text_feature_dtype,
        )
        dataset_latents_sampler = misc.InfiniteSampler(
            dataset=dataset_latents_obj,
            rank=get_rank(),
            num_replicas=dist.get_world_size(),
            seed=seed
        )
        dataset_latents_iterator = iter(torch.utils.data.DataLoader(
            dataset=dataset_latents_obj,
            sampler=dataset_latents_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs
        ))

    dataset_latents_obj_text = TextDataset(
        hf_dataset=hf_dataset,
        resolution=resolution,
        precomputed_latents_path=precomputed_latents_path,
        sd3=True,
        text_feature_dtype=text_feature_dtype,
    )
    dataset_latents_sampler_text = misc.InfiniteSampler(
        dataset=dataset_latents_obj_text,
        rank=get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed
    )
    dataset_latents_iterator_text = iter(torch.utils.data.DataLoader(
        dataset=dataset_latents_obj_text,
        sampler=dataset_latents_sampler_text,
        batch_size=batch_gpu,
        **data_loader_kwargs
    ))

    dist.print0(f"Loaded dataset from: {file_path}, dataset size: {len(dataset_latents_obj_text)}")
    dist.print0(f"Switched to dataset part: {datapart}")
    torch.distributed.barrier()

    # Try to free up memory from any temporary objects
    del dataset_latents_obj
    del dataset_latents_sampler
    del dataset_latents_obj_text
    del dataset_latents_sampler_text
    gc.collect()
    torch.cuda.empty_cache()

    return dataset_latents_iterator, dataset_latents_iterator_text, hf_dataset

#def load_checkpoint(resume_training, fake_score, fake_score_optimizer_kwargs, G, g_optimizer_kwargs, FSDP, fsdp_kwargs, dnnlib, dist,true_score=None,sid_model=None):
def load_checkpoint(resume_training, fake_score_optimizer_kwargs, g_optimizer_kwargs, FSDP, fsdp_kwargs, dnnlib, dist,true_score,dtype=torch.float32,sid_model=None):
    import torch
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullOptimStateDictConfig
    device=true_score.device if true_score is not None else fake_score.device
    def convert_to_dtype(state_dict):
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and (v.dtype != dtype):
                state_dict[k] = v.to(dtype=dtype)
        return state_dict

    def convert_optimizer_state_to_dtype(opt_state):
        for state in opt_state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and (v.dtype != dtype):
                    state[k] = v.to(dtype=dtype)

    # def convert_to_dtype(state_dict,dtype=torch.float32,device=None): #,device='cuda'):
    #     for k, v in state_dict.items():
    #         if isinstance(v, torch.Tensor):
    #             state_dict[k] = v.to(dtype=dtype) if device is None else v.to(dtype=dtype, device=device)
    #     return state_dict

    # def convert_optimizer_state_to_dtype(opt_state,dtype=torch.float32,device=None): #,device='cuda'):
    #     for state in opt_state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.to(dtype=dtype) if device is None else v.to(dtype=dtype, device=device)
    
    def move_and_cast_optimizer_state(optimizer, device, dtype):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, dtype=dtype)

    

    if resume_training is None:
        # fake_score: The auxiliary model to be trained to learn the score of the generator (we alternate between training the generator and the fake score net)
    # Deep copy the teacher model to initialize the fake_score network, ensuring it starts with the same weights.
        if 1:
            fake_score = copy.deepcopy(true_score).train().requires_grad_(True).to(device=device)
            #fake_score = type(true_score)(**true_score.config).train().requires_grad_(True).to(device=device)
            #fake_score.load_state_dict(true_score.state_dict(), strict=False)
        else:
            true_score = FSDP(true_score, **fsdp_kwargs)
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            #full_optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(
                true_score,
                StateDictType.FULL_STATE_DICT,
                full_state_dict_config,
                #full_optim_state_dict_config
            ):
                state_dict= true_score.state_dict()
            #fake_score = copy.deepcopy(true_score).train().requires_grad_(True).to(device=device)
            fake_score = type(true_score.module)(**true_score.module.config).train().requires_grad_(True).to(device=device)
            fake_score.load_state_dict(state_dict, strict=False)
        fake_score_fsdp = FSDP(fake_score, **fsdp_kwargs)
       

        # G: Generator model, either initialized from the teacher or loaded from a checkpoint.
        if sid_model is None:
            if 1:
                G = copy.deepcopy(true_score).train().requires_grad_(True).to(device=device)
            else:
                G = type(true_score.module)(**true_score.module.config).train().requires_grad_(True).to(device=device)
                G.load_state_dict(state_dict, strict=False)
            
            #G = type(true_score)(**true_score.config).train().requires_grad_(True).to(device=device)   
            #G.load_state_dict(true_score.state_dict(), strict=False)
        else:
            dist.print0(f'Loading network from "{sid_model}"...')
            with dnnlib.util.open_url(sid_model, verbose=(get_rank() == 0)) as f:
                # Load the EMA (exponential moving average) weights for the generator.
                # Note: while the name suggests the use of EMA, we actually disabled it when running FSDP, as in this code.
                loaded_net = pickle.load(f)
                if 'ema' in loaded_net:
                    G = loaded_net['ema'].to(device=device, dtype=dtype)
                elif 'G' in loaded_net:
                    G = loaded_net['G'].to(device=device, dtype=dtype)
                else:
                    raise KeyError("Neither 'ema' nor 'G' found in the loaded checkpoint.")
                G.train().requires_grad_(True)
                del loaded_net
            dist.print0(f'Loaded network from "{sid_model}"...')
        G_fsdp = FSDP(G, **fsdp_kwargs)
        if 0:
            del state_dict
        else:
            true_score = FSDP(true_score, **fsdp_kwargs)
        # Barrier to synchronize all processes
        torch.distributed.barrier()

        #fake_score_fsdp = FSDP(fake_score, **fsdp_kwargs)
        #G_fsdp = FSDP(G, **fsdp_kwargs)
        #fake_score_optimizer = dnnlib.util.construct_class_by_name(params=fake_score.parameters(), **fake_score_optimizer_kwargs)
        #g_optimizer = dnnlib.util.construct_class_by_name(params=G.parameters(), **g_optimizer_kwargs)

        #fake_score_optimizer = dnnlib.util.construct_class_by_name(params=fake_score_fsdp.parameters(), **fake_score_optimizer_kwargs)
        #g_optimizer = dnnlib.util.construct_class_by_name(params=G_fsdp.parameters(), **g_optimizer_kwargs)

        fake_score_optimizer = torch.optim.Adam(
                fake_score_fsdp.parameters(),
                lr=fake_score_optimizer_kwargs.lr,
                betas=tuple(fake_score_optimizer_kwargs.betas),
                eps=fake_score_optimizer_kwargs.eps
        )

        g_optimizer = torch.optim.Adam(
            G_fsdp.parameters(),
            lr=g_optimizer_kwargs.lr,
            betas=tuple(g_optimizer_kwargs.betas),
            eps=g_optimizer_kwargs.eps
        )


    else:
        dist.print0('checkpoint path:', resume_training)
        rank_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

        model_checkpoint = torch.load(resume_training, map_location=torch.device('cpu'), weights_only=False)
        if true_score is not None:
            fake_score = type(true_score)(**true_score.config).train().requires_grad_(True).to(device=device,dtype=dtype)
        if model_checkpoint.get('fake_score_state') is None:
            resume_training_fake = resume_training + '_fake'
            model_checkpoint = torch.load(resume_training_fake, map_location=torch.device('cpu'), weights_only=False)
            #model_checkpoint = torch.load(resume_training_fake, map_location=device, weights_only=False)
        fake_score.load_state_dict(convert_to_dtype(model_checkpoint['fake_score_state']))
        #fake_score.load_state_dict(model_checkpoint['fake_score_state']) #,device=device))
        #fake_score.to(device=rank_device, dtype=dtype)
        fake_score_fsdp = FSDP(fake_score, **fsdp_kwargs)

        if model_checkpoint.get('fake_score_optimizer_state') is None:
            resume_training_fake_optim = resume_training + '_fake_optim'
            model_checkpoint = torch.load(resume_training_fake_optim, map_location=torch.device('cpu'), weights_only=False)
            #model_checkpoint = torch.load(resume_training_fake_optim, map_location=device, weights_only=False)

        FSDP.set_state_dict_type(
            fake_score_fsdp,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
        )
        
        #fake_score_optimizer = dnnlib.util.construct_class_by_name(params=fake_score_fsdp.parameters(), **fake_score_optimizer_kwargs)


        fake_score_optimizer = torch.optim.Adam(
                fake_score_fsdp.parameters(),
                lr=fake_score_optimizer_kwargs.lr,
                betas=tuple(fake_score_optimizer_kwargs.betas),
                eps=fake_score_optimizer_kwargs.eps
        )

        

        optim_state_dict = FSDP.optim_state_dict_to_load(
            fake_score_fsdp, fake_score_optimizer, model_checkpoint['fake_score_optimizer_state']
        )
        fake_score_optimizer.load_state_dict(optim_state_dict)
        convert_optimizer_state_to_dtype(fake_score_optimizer.state)
        #move_and_cast_optimizer_state(fake_score_optimizer, device=rank_device, dtype=dtype)

        #del optim_state_dict
        
        if true_score is not None:
            G = type(true_score)(**true_score.config).to(device=device,dtype=dtype).train().requires_grad_(True)

        if model_checkpoint.get('G_state') is None:
            resume_training_G = resume_training + '_G'
            model_checkpoint = torch.load(resume_training_G, map_location=torch.device('cpu'), weights_only=False)
            #model_checkpoint = torch.load(resume_training_G, map_location=device, weights_only=False)
        G.load_state_dict(convert_to_dtype(model_checkpoint['G_state']))
        #G.load_state_dict(model_checkpoint['G_state'])
        #G.to(device=rank_device, dtype=dtype)
        G_fsdp = FSDP(G, **fsdp_kwargs)

        if model_checkpoint.get('g_optimizer_state') is None:
            resume_training_G_optim = resume_training + '_G_optim'
            model_checkpoint = torch.load(resume_training_G_optim, map_location=torch.device('cpu'), weights_only=False)
            #model_checkpoint = torch.load(resume_training_G_optim, map_location=device, weights_only=False)

        FSDP.set_state_dict_type(
            G_fsdp,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
        )
        #g_optimizer = dnnlib.util.construct_class_by_name(params=G_fsdp.parameters(), **g_optimizer_kwargs)

        g_optimizer = torch.optim.Adam(
            G_fsdp.parameters(),
            lr=g_optimizer_kwargs.lr,
            betas=tuple(g_optimizer_kwargs.betas),
            eps=g_optimizer_kwargs.eps
        )

        optim_state_dict = FSDP.optim_state_dict_to_load(
            G_fsdp, g_optimizer, model_checkpoint['g_optimizer_state']
        )
        g_optimizer.load_state_dict(optim_state_dict)
        convert_optimizer_state_to_dtype(g_optimizer.state)
        #convert_optimizer_state_to_dtype(g_optimizer.state,dtype=dtype,device=device)
        #move_and_cast_optimizer_state(g_optimizer, device=rank_device, dtype=dtype)
        #del optim_state_dict

        #del model_checkpoint

        for optimizer in [fake_score_optimizer, g_optimizer]:
            # Ensure all optimizer states are on the correct device and dtype
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p not in optimizer.state:
                        continue
                    state = optimizer.state[p]
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            # Align device
                            if v.device != p.device:
                                state[k] = v.to(p.device)
                            # Align dtype
                            if v.dtype != p.dtype:
                                state[k] = state[k].to(dtype=p.dtype)



        true_score = FSDP(true_score, **fsdp_kwargs)

        torch.distributed.barrier()
        dist.print0('Loading checkpoint completed')




    true_score.eval().requires_grad_(False) #.to(device)  

    return fake_score_fsdp, fake_score_optimizer, G_fsdp, g_optimizer,true_score

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    fake_score_optimizer_kwargs   = {},       # Options for fake score network optimizer.
    g_optimizer_kwargs    = {},     # Options for generator optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 2000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 0,      # Half-life of the exponential moving average (EMA) of model weights.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    loss_scaling_G      = 100,       # Loss scaling factor of G for reducing FP16 under/overflows.
    kimg_per_tick       = 2,       # Interval of progress prints.
    snapshot_ticks      = 25,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 25,      # How often to dump training state, None = disable.
    resume_training     = None,     # Resume training from the given network snapshot.
    resume_kimg         = 0,        # Start from the given training progress.
    alpha               = 1,         # loss = L2-alpha*L1
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    metrics             = None,
    init_timestep       = 999,
    metric_pt_path      = None,
    metric_open_clip_path            = 'clipvitg14.pkl',
    metric_clip_path                 = None,
    pretrained_model_name_or_path    = "Efficient-Large-Model/Sana_600M_512px_diffusers",
    pretrained_vae_model_name_or_path= "Efficient-Large-Model/Sana_600M_512px_diffusers",
    dataset_prompt_text_kwargs       = {},
    cfg_train_fake                   = 4.5,
    cfg_eval_fake                    = 4.5,
    cfg_eval_real                    = 4.5,
    num_steps                        = 4,
    resolution                       = None,
    loss_scaling_D                   = 1,
    loss_scaling_G_gan               = 1,
    dataset_latents_kwargs           = None,
    sid_model                        = None,
    pooling_type                     = 'spatial',
    text_device                      = torch.device('cuda'),
    vae_device                       = torch.device('cuda'),
    cpu_offload                      = False,
    time_scale                       = 1,
    gradient_checkpointing           = True,
    text_image_pair_path             = None,
    save_best_and_last               = True,
    weighting_scheme                 = "1_minus_sigma",  # Weighting scheme for loss computation
    noise_type                       = "fresh",          # Noise type: "fresh", "fixed", "ddim"
    train_diffusiongan               = False,
    precompute_latents               = False, 
    dataset_name           = "processed_dataset",
    use_sd3_shift            = False,
    text_encoders_dtype = None
):
    #cpu_offload  = True
    
    num_nodes = int(os.environ['WORLD_SIZE']) // int(os.environ['LOCAL_WORLD_SIZE'])
    if text_image_pair_path is not None:
        precompute_latents = True
        dist.print0("precompute latents is set to True")
        #precompute latents as of now only supports text-image pair dataset
    #if num_nodes>1:
    #    precompute_latents = False
    #precompute_latents = False
    print("precompute_latents", precompute_latents)
    

    #BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    BASE_DIR = os.environ.get("IRISCTL_SHARED_ARTIFACT_DIR")
    dist.print0(BASE_DIR)
    if BASE_DIR is None or BASE_DIR == '':
        print('>'*100, 'no shared dir env found')
        BASE_DIR = "/mnt/task_runtime/shared"

    # Use the last part of run_dir to name the cache directory
    run_name = os.path.basename(os.path.normpath(run_dir))
    cache_latents_dir = os.path.join(BASE_DIR, "data", "tmp", run_name)

    cache_latents_dir = re.sub(r'[^A-Za-z0-9_/.\-]', '_', cache_latents_dir)

    os.makedirs(cache_latents_dir, exist_ok=True)

    if save_best_and_last:
        if get_rank() == 0:
            previous_pt_filename = None #for saving the last checkpoint
            best_fid = float('inf')
            best_clip_score = float('-inf')
            current_fid = float('inf')
            current_clip_score = float('-inf')
            best_fid_step = None
            best_clip_step = None
    

    num_steps_eval = num_steps
    guidance_scale = 0
 
    use_fsdp = True
    use_ddp = not use_fsdp
    
    true_score_fsdp = True
    
    if ema_halflife_kimg==0:
        ema_device=device
    
    if resume_training is not None:
        seed = (seed+resume_kimg)% (1 << 31)

    cur_nimg = resume_kimg * 1000
        
    use_context_dropout_train_fake = False


    use_train_eval = False #whether switching between train and eval modes for G and fake score net
    use_grad_turn_on_off=True #we can keep train modes for both G and fake score net, but turning on and off gradients instead
    #use_grad_turn_on_off=False 

   
    if network_kwargs.use_fp16:
        dtype= torch.float16
    else:
        dtype=torch.bfloat16 if network_kwargs.use_bf16 else torch.float32
        
    use_autocast = True if dtype==torch.float32 else False
    dtype_autocast=torch.bfloat16 if network_kwargs.autocast_bf16 else torch.float16

    dist.print0(use_autocast,'use_autocast')
    dist.print0(dtype,'dtype')
    dist.print0(dtype_autocast,'dtype_autocast')


    if text_encoders_dtype is None:
        text_encoders_dtype = dtype
    elif isinstance(text_encoders_dtype, str):
        if text_encoders_dtype.lower() == 'float16' or text_encoders_dtype.lower() == 'fp16':
            text_encoders_dtype = torch.float16
        elif text_encoders_dtype.lower() == 'bfloat16' or text_encoders_dtype.lower() == 'bf16':
            text_encoders_dtype = torch.bfloat16
        elif text_encoders_dtype.lower() == 'float32' or text_encoders_dtype.lower() == 'fp32':
            text_encoders_dtype = torch.float32
        else:
            raise ValueError(f"Invalid text_encoders_dtype: {text_encoders_dtype}")

    text_feature_dtype = dtype

    fsdp_kwargs = {
        "cpu_offload": CPUOffload(offload_params=cpu_offload),
        "auto_wrap_policy": size_based_auto_wrap_policy,
        'device_id': torch.cuda.current_device(), 
    }
    num_nodes = int(os.environ['WORLD_SIZE']) // int(os.environ['LOCAL_WORLD_SIZE'])
    # if num_nodes>1:

    #     fsdp_kwargs = {
    #         "cpu_offload": CPUOffload(offload_params=cpu_offload),
    #         "auto_wrap_policy": size_based_auto_wrap_policy,
    #         "device_id": torch.cuda.current_device(), 
    #         "forward_prefetch": False,                       # leave params resharded immediately after forward
    #         "backward_prefetch": BackwardPrefetch.BACKWARD_POST,  # overlap comm in backward without doubling buffers
    #         "limit_all_gathers": True,                       # reduce number of NCCL calls
    #         #"sync_module_states": False,                     # only broadcast once at init
    #         # you can also add "bucket_cap_mb": 16 if you want smaller gradient buckets
    #     }

    

    if fsdp_kwargs["cpu_offload"].offload_params:
        fsdp_kwargs["sync_module_states"] = False
    else:
        fsdp_kwargs["sync_module_states"] = True

        fsdp_kwargs["use_orig_params"] = True
        use_grad_turn_on_off=True

    #fsdp_kwargs["sync_module_states"] = True
    #fsdp_kwargs["use_orig_params"] = True

    
    

    if (use_autocast and dtype_autocast==torch.float16) or dtype==torch.float16:
        scaler =ShardedGradScaler()
        scaler_G = ShardedGradScaler()
    
    if get_rank() == 0:
        # vae, dit, noise_scheduler, text_encoding_pipeline = load_dit(
        #     pretrained_model_name_or_path=pretrained_model_name_or_path, 
        #     weight_dtype=dtype, 
        #     num_steps=num_steps,
        #     train_diffusiongan=train_diffusiongan,
        #     device=device,
        # ) 
        vae, dit, noise_scheduler, tokenizers, text_encoders = load_dit(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            weight_dtype=dtype, 
            num_steps=num_steps,
            train_diffusiongan=train_diffusiongan,
            device=device,
            text_encoders_dtype=text_encoders_dtype,
        )
    
    # Ensure rank 0 finishes loading before others continue
    torch.distributed.barrier()

    # Now all ranks load from cache
    if get_rank() != 0:
        # vae, dit, noise_scheduler, text_encoding_pipeline = load_dit(
        #     pretrained_model_name_or_path=pretrained_model_name_or_path, 
        #     weight_dtype=dtype, 
        #     num_steps=num_steps,
        #     train_diffusiongan=train_diffusiongan,
        #     device=device,
        # ) 
        vae, dit, noise_scheduler, tokenizers, text_encoders = load_dit(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            weight_dtype=dtype, 
            num_steps=num_steps,
            train_diffusiongan=train_diffusiongan,
            device=device,
            text_encoders_dtype=text_encoders_dtype,
        )            
    
    if precompute_latents: #num_nodes == 1:
        text_device                      = 'cpu'
        vae_device                       = 'cpu'
    if text_device == 'cpu':
        #text_encoding_pipeline=text_encoding_pipeline.to(torch.device('cpu'))
        text_encoders = [text_encoder.to(torch.device('cpu')) for text_encoder in text_encoders]
        #tokenizers = [tokenizer.to(torch.device('cpu')) for tokenizer in tokenizers]
    if vae_device == 'cpu':
        vae=vae.to(torch.device('cpu'))


    
    if not precompute_latents:
        rank_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        # local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # fsdp_vae_kwargs = {
        #     "cpu_offload": CPUOffload(offload_params=cpu_offload),
        #     "auto_wrap_policy": size_based_auto_wrap_policy,
        #     "device_id": local_rank,
        #     "sync_module_states": not cpu_offload,
        # }

        # # FSDP kwargs for text encoders (GPU‑only)
        # fsdp_te_kwargs = {
        #     "auto_wrap_policy": size_based_auto_wrap_policy,
        #     "device_id": local_rank,
        #     "sync_module_states": True,
        # }

        # # wrap VAE
        # vae = vae.to(rank_device)
        # vae = FSDP(vae, **fsdp_vae_kwargs)

        # # wrap text encoders without cpu_offload
        # text_encoders = [te.to(rank_device) for te in text_encoders]
        # text_encoders = [FSDP(te, **fsdp_te_kwargs) for te in text_encoders]


    torch.distributed.barrier()

    negative_prompt = [""] *batch_gpu
    # uncond_embeds, uncond_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
    #     prompt=negative_prompt,
    #     complex_human_instruction=False,
    #     do_classifier_free_guidance=False,
    # )
    rank_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    context_manager_text_encoding_pipeline = (
        contextlib.nullcontext(None)
        if not precompute_latents else
        temporary_model_on_device(text_encoders, rank_device)
    )
    dist.print0("precompute_latents0", precompute_latents)
    with context_manager_text_encoding_pipeline:

    #with temporary_model_on_device(text_encoders, device):
        with torch.autocast(device_type="cuda",dtype=dtype_autocast,enabled=use_autocast):
            uncond_embeds, uncond_pooled_embeds = encode_prompt(
                    text_encoders,
                    tokenizers,
                    negative_prompt,
                    max_sequence_length=77,
                    device=device,
                    num_images_per_prompt=1,
                    text_feature_dtype=text_feature_dtype,
                )
    
    grid_size = None
    grid_z = None
    grid_c = None

    # latent_img_channels = 32
    # latent_img_resolution = resolution//32
    latent_img_channels = 16
    latent_img_resolution = resolution//8
    
    # Load dataset.
    dist.print0('Loading dataset...')
    # if precompute_latents:
    #     dataset_kwargs["text_encoding_pipeline"] = text_encoding_pipeline
    #     dataset_kwargs["uncond_embeds"] = uncond_embeds
    #     dataset_kwargs["uncond_attention_mask"] = uncond_attention_mask
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    resolution=dataset_obj.resolution
    
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))


    if use_fsdp or get_rank() == 0: 
        torch.manual_seed(2024)
        grid_size, images_real, contexts = setup_snapshot_image_grid(training_set=dataset_obj)
        grid_z = torch.randn([images_real.shape[0], latent_img_channels, latent_img_resolution, latent_img_resolution], device=device, dtype=dtype)
        grid_z = grid_z.split(batch_gpu)
        grid_c = split_list(contexts, batch_gpu)
        for c in grid_c:
            dist.print0(c)


    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    

    iteration = 0
    
    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    

    if text_image_pair_path is None:
        # Data-free setting: use any prompts you wish (e.g., for distillation from text only)
        # if precompute_latents:
        #     dataset_prompt_text_kwargs["text_encoding_pipeline"] = text_encoding_pipeline
        #     dataset_prompt_text_kwargs["uncond_embeds"] = uncond_embeds
        #     dataset_prompt_text_kwargs["uncond_attention_mask"] = uncond_attention_mask
        dataset_prompt_text_obj = dnnlib.util.construct_class_by_name(**dataset_prompt_text_kwargs) # subclass of training.dataset.Dataset
        dataset_prompt_text_sampler = misc.InfiniteSampler(dataset=dataset_prompt_text_obj, rank=get_rank(), num_replicas=dist.get_world_size(), seed=seed)
        dataset_prompt_text_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_prompt_text_obj, sampler=dataset_prompt_text_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    else:
        #if train_DiffusionGAN or Use_RealImagePrompts_Train_G:
        # Using prompts from real/synthetic text-image pairs
        # If GAN is used, we will also use the images to train the fake score net (part of it is reused as discriminator)
        # We follow SANA to use the following (synthetic) dataset:
        # https://huggingface.co/datasets/brivangl/midjourney-v6-llava
        
        total_images = 986000
        images_per_zip = total_images / 50
        datapart = cur_nimg // images_per_zip

        actual_datapart = datapart % 124
        if precompute_latents:
            precomputed_latents_path = os.path.join(
                    cache_latents_dir, f"{dataset_name}_{actual_datapart}.{'text_image_latents' if train_diffusiongan else 'text_latents'}.pt"
                )
        else:
            precomputed_latents_path = None
        #rank_device = torch.device(f"cuda:{get_rank() % dist.get_world_size()}")
        rank_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        
        context_manager_vae = (
            contextlib.nullcontext(None) 
            if not precompute_latents else
            temporary_model_on_device(vae, rank_device)
            if train_diffusiongan and (precomputed_latents_path is not None and not os.path.exists(precomputed_latents_path)) else
            contextlib.nullcontext(vae)
        )
        # context_manager_text_encoding_pipeline = (
        #     temporary_model_on_device(text_encoding_pipeline, rank_device)
        #     if precomputed_latents_path is not None and not os.path.exists(precomputed_latents_path) else
        #     contextlib.nullcontext(text_encoding_pipeline)
        # )
        if precomputed_latents_path is not None:
            dist.print0("precomputed_latents_path", precomputed_latents_path)
        else:
            dist.print0("precomputed_latents_path:None")
        if precomputed_latents_path is not None and os.path.exists(precomputed_latents_path) is not None:
            dist.print0("os.path.exists(precomputed_latents_path)", os.path.exists(precomputed_latents_path))
        else:
            dist.print0("os.path.exists(precomputed_latents_path):None")

        context_manager_text_encoding_pipeline = (
            contextlib.nullcontext(None)
            if not precompute_latents else
            temporary_model_on_device(text_encoders, rank_device)
            if precomputed_latents_path is not None and not os.path.exists(precomputed_latents_path) else
            contextlib.nullcontext(text_encoders)
        )
        dist.print0("precompute_latents0", precompute_latents)
        with context_manager_text_encoding_pipeline:
            with context_manager_vae:
                dataset_latents_iterator, dataset_latents_iterator_text, hf_dataset = stream_data_sampler(
                    datapart=int(datapart),
                    text_image_pair_path=text_image_pair_path,
                    cache_latents_dir=cache_latents_dir,
                    #dataset_name=processed_dataset_name,
                    #text_encoding_pipeline=text_encoding_pipeline,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    vae=vae,
                    resolution=resolution,
                    train_diffusiongan=train_diffusiongan,
                    uncond_embeds=uncond_embeds,
                    #uncond_attention_mask=uncond_attention_mask,
                    uncond_pooled_embeds=uncond_pooled_embeds,
                    batch_gpu=batch_gpu,
                    seed=seed,
                    data_loader_kwargs=data_loader_kwargs,
                    precomputed_latents_path=precomputed_latents_path,
                    text_feature_dtype=text_feature_dtype,
                )

        dist.print0("sanity check for text-image pair dataset")
        torch.distributed.barrier()
        
        if train_diffusiongan:
            batch = next(dataset_latents_iterator)
            dist.print0(batch["text"])
            latents = batch["image"].to(dtype=vae.dtype).to(device)


            #rank_device = torch.device(f"cuda:{get_rank() % dist.get_world_size()}")
            rank_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
            context_manager_vae = (
                contextlib.nullcontext(None) 
                if not precompute_latents else
                temporary_model_on_device(vae, rank_device)
                #if train_diffusiongan and not precompute_latents else
                if train_diffusiongan else
                contextlib.nullcontext(vae)
            )
            with context_manager_vae:
                if not precompute_latents:
                    dist.print0(latents.shape)
                    with torch.no_grad():
                        #latents = vae.encode(latents).latent*vae.config.scaling_factor #0.41407
                        latents = vae.encode(latents).latent_dist.sample()
                        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                dist.print0(latents.shape)
            
                if get_rank() == 0:    
                    images_data = []
                    images = []
                    for z, c in zip(grid_z, grid_c):
                        batch = next(dataset_latents_iterator)
                        actual_batch_size = batch["image"].shape[0]
                        if len(c) > actual_batch_size:
                            dist.print0(f"Warning: batch too small. Requested {len(c)}, got {actual_batch_size}. Truncating.")
                            c = c[:actual_batch_size]
                        dist.print0(f"texts: {batch['text'][:len(c)]}")
                        image = batch["image"][:len(c)].to(dtype=vae.dtype).to(device)
                        #dist.print0(f"[DEBUG] image shape: {image.shape}, dtype: {image.dtype}")
                        if not precompute_latents:
                            with torch.no_grad():
                                #encoded = vae.encode(image).latent * vae.config.scaling_factor
                                encoded = vae.encode(image).latent_dist.sample()
                                encoded = (encoded - vae.config.shift_factor) * vae.config.scaling_factor
                            #dist.print0(f"[DEBUG] encoded shape (after vae.encode): {encoded.shape}, dtype: {encoded.dtype}")
                        else:
                            encoded = image
                            #dist.print0(f"[DEBUG] encoded shape (precomputed): {encoded.shape}, dtype: {encoded.dtype}")
                        #decoded = vae.decode(encoded / vae.config.scaling_factor, return_dict=False)[0]
                        decoded =vae.decode((encoded / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
                        #dist.print0(f"[DEBUG] decoded shape: {decoded.shape}, dtype: {decoded.dtype}")
                        
                        images_data.append(image)
                        images.append(decoded)
                    images_data = torch.cat(images_data).to(torch.float32).cpu().numpy()
                    images = torch.cat(images).to(torch.float32).cpu().numpy()
                    #dist.print0(f"[DEBUG] images_data (cat) shape: {images_data.shape}, images (cat) shape: {images.shape}")
                    if not precompute_latents:
                        save_image_grid(img=images_data, fname=os.path.join(run_dir, 'true_mj.png'), drange=[-1,1], grid_size=grid_size)
                    save_image_grid(img=images, fname=os.path.join(run_dir, 'true_mj_vae_encode_decode.png'), drange=[-1,1], grid_size=grid_size)
                
             
           
    for _i in range(2):
        dist.print0(f'prompt for SiD:{_i}')
        if text_image_pair_path is not None:
            contexts = next(dataset_latents_iterator_text)["text"]
        else:
            _, contexts  = next(dataset_prompt_text_iterator)
        dist.print0(contexts)
    

    # true_score: The teacher model (frozen, used for distillation targets).
    true_score = dit
    true_score.eval().requires_grad_(False).to(device)  
    if 1:
        fake_score_fsdp, fake_score_optimizer, G_fsdp, g_optimizer,true_score = load_checkpoint(resume_training, fake_score_optimizer_kwargs, g_optimizer_kwargs, FSDP, fsdp_kwargs, dnnlib, dist,true_score=true_score,dtype=dtype,sid_model=sid_model)
        
    if 0:
        # fake_score: The auxiliary model to be trained to learn the score of the generator (we alternate between training the generator and the fake score net)
        # Deep copy the teacher model to initialize the fake_score network, ensuring it starts with the same weights.
        #fake_score = copy.deepcopy(true_score).train().requires_grad_(True).to(device=device)
        #fake_score = type(true_score)(**true_score.config).train().requires_grad_(True).to(device=device)
        #fake_score.load_state_dict(true_score.state_dict(), strict=False)
        if resume_training is None:
            true_score = FSDP(true_score, **fsdp_kwargs)
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            full_optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(
                true_score,
                StateDictType.FULL_STATE_DICT,
                full_state_dict_config,
                full_optim_state_dict_config
            ):
                state_dict= true_score.state_dict()
            #fake_score = copy.deepcopy(true_score).train().requires_grad_(True).to(device=device)
            fake_score = type(true_score.module)(**true_score.module.config).train().requires_grad_(True).to(device=device)
            fake_score.load_state_dict(state_dict, strict=False)
            fake_score_fsdp = FSDP(fake_score, **fsdp_kwargs)
        else:
            fake_score = None

        # G: Generator model, either initialized from the teacher or loaded from a checkpoint.
        if sid_model is None:
            if resume_training is None:
                #G = copy.deepcopy(true_score).train().requires_grad_(True).to(device=device)
                G = type(true_score.module)(**true_score.module.config).train().requires_grad_(True).to(device=device)
                G.load_state_dict(state_dict, strict=False)
            else:
                G = None
            #G = type(true_score)(**true_score.config).train().requires_grad_(True).to(device=device)   
            #G.load_state_dict(true_score.state_dict(), strict=False)
        else:
            dist.print0(f'Loading network from "{sid_model}"...')
            with dnnlib.util.open_url(sid_model, verbose=(get_rank() == 0)) as f:
                # Load the EMA (exponential moving average) weights for the generator.
                # Note: while the name suggests the use of EMA, we actually disabled it when running FSDP, as in this code.
                loaded_net = pickle.load(f)
                if 'ema' in loaded_net:
                    G = loaded_net['ema'].to(device=device, dtype=dtype)
                elif 'G' in loaded_net:
                    G = loaded_net['G'].to(device=device, dtype=dtype)
                else:
                    raise KeyError("Neither 'ema' nor 'G' found in the loaded checkpoint.")
                G.train().requires_grad_(True)
                del loaded_net
            dist.print0(f'Loaded network from "{sid_model}"...')
        if resume_training is None:
            G_fsdp = FSDP(G, **fsdp_kwargs)
            del state_dict
        
        # Barrier to synchronize all processes
        torch.distributed.barrier()
        
        # Wrap the true_score model with FSDP for distributed training.
        
        if resume_training is None:
            #true_score = FSDP(true_score, **fsdp_kwargs)
            
            #fake_score_optimizer = dnnlib.util.construct_class_by_name(params=fake_score.parameters(), **fake_score_optimizer_kwargs)
            #g_optimizer = dnnlib.util.construct_class_by_name(params=G.parameters(), **g_optimizer_kwargs)
            
            fake_score_optimizer = dnnlib.util.construct_class_by_name(params=fake_score_fsdp.parameters(), **fake_score_optimizer_kwargs)
            g_optimizer = dnnlib.util.construct_class_by_name(params=G_fsdp.parameters(), **g_optimizer_kwargs)


        else:
            fake_score_fsdp, fake_score_optimizer, G_fsdp, g_optimizer = load_checkpoint(resume_training, fake_score, fake_score_optimizer_kwargs, G, g_optimizer_kwargs, FSDP, fsdp_kwargs, dnnlib, dist,true_score)
            true_score = FSDP(true_score, **fsdp_kwargs)
            

    if use_ddp:
        if use_train_eval:
            fake_score_ddp.eval().requires_grad_(False)
            G_ddp.eval().requires_grad_(False)
        else:
            fake_score_ddp.train().requires_grad_(True)
            G_ddp.train().requires_grad_(True)                                                  
    else:         
        G_ema = None                                  
        if use_train_eval:
            fake_score_fsdp.eval().requires_grad_(False)
            G_fsdp.eval().requires_grad_(False)
        else:
            true_score.eval().requires_grad_(False)
            fake_score_fsdp.train().requires_grad_(True)
            G_fsdp.train().requires_grad_(True)      
            
    torch.distributed.barrier()
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    stats_metrics = dict()
    
    torch.distributed.barrier()
    
    if 1: #resume_training is None:
        if use_fsdp or get_rank() == 0:
            dist.print0('Exporting sample images...')
            
            # dist.print0(images_real[0])
            if get_rank() == 0:
                save_image_grid(img=images_real, fname=os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
                del images_real
        
            dist.print0(grid_c[0])
            #rank_device = torch.device(f"cuda:{get_rank() % dist.get_world_size()}")
            rank_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
            #with temporary_model_on_device(text_encoding_pipeline, rank_device):

            context_manager_vae = (
                contextlib.nullcontext(None) 
                if not precompute_latents else
                temporary_model_on_device(vae, rank_device)
            )

            context_manager_text_encoding_pipeline = (
                contextlib.nullcontext(None)
                if not precompute_latents else
                temporary_model_on_device(text_encoders, rank_device)
            )
            
            with context_manager_text_encoding_pipeline:
                with context_manager_vae:
            #with temporary_model_on_device(text_encoders, rank_device):
                #with temporary_model_on_device(vae, rank_device):
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda",dtype=dtype_autocast,enabled=use_autocast):
                            images = [sid_dit_generate(dit=G_fsdp,latents=z,contexts=c,
                                        init_timesteps=init_timestep * torch.ones((len(c),), device=device, dtype=torch.long),
                                        noise_scheduler=noise_scheduler,
                                        #text_encoding_pipeline = text_encoding_pipeline,
                                        text_encoders=text_encoders,
                                        tokenizers=tokenizers,
                                        resolution=resolution,dtype=dtype,return_images=True, vae=vae,num_steps=num_steps,train_sampler=False,num_steps_eval=num_steps,
                                        time_scale=time_scale,noise_type=noise_type,use_sd3_shift=use_sd3_shift,text_feature_dtype=text_feature_dtype) for z, c in zip(grid_z, grid_c)]
                            
            if use_ddp:
                if G_ema.device!=ema_device:
                    G_ema=G_ema.to(ema_device)
            if get_rank() == 0:
                images = torch.cat(images).to(torch.float32).cpu().numpy()
                dist.print0(contexts[0])
                #print(images[0])
                save_image_grid(img=images, fname=os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
            del images
        
        
    
    # Barrier to synchronize all processes
    torch.distributed.barrier()

    
    if use_ddp:
        if use_train_eval:
            fake_score_ddp.train().requires_grad_(True)
            G_ddp.eval().requires_grad_(False)
            
        if use_grad_turn_on_off:
            G_ddp.requires_grad_(False)
            fake_score_ddp.requires_grad_(True)
            
        if gradient_checkpointing:
            G_ddp.module.enable_gradient_checkpointing()
            fake_score_ddp.module.enable_gradient_checkpointing()                                             
    else:                     
        #dist.print0(f"Iteration{iteration}  Prepare 1")
        if use_train_eval:
            fake_score_fsdp.train().requires_grad_(True)
            G_fsdp.eval().requires_grad_(False)
            
        #dist.print0(f"Iteration{iteration}  Prepare 2")
        if use_grad_turn_on_off:
            G_fsdp.requires_grad_(False)
            fake_score_fsdp.requires_grad_(True)
                
        #dist.print0(f"Iteration{iteration}  Prepare 3")
        if gradient_checkpointing:
            G_fsdp.enable_gradient_checkpointing()
            fake_score_fsdp.enable_gradient_checkpointing()
        #dist.print0(f"Iteration{iteration}  Prepare 4")                                             
    if use_ddp:
        if G_ema.device!=ema_device:
            G_ema=G_ema.to(ema_device)
        
    #full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    #full_optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)

    # with FSDP.state_dict_type(
    #     fake_score_fsdp,
    #     StateDictType.FULL_STATE_DICT,
    #     full_state_dict_config,
    #     full_optim_state_dict_config
    # ):
    #     fake_score_state = fake_score_fsdp.state_dict()
    #     fake_score_optimizer_state_dict = FSDP.optim_state_dict(fake_score_fsdp, fake_score_optimizer)

    # if not fake_score_optimizer_state_dict:
    #     dist.print0("Optimizer state dictionary is empty.")
    # else:
    #     dist.print0("Optimizer state dictionary contains data.")



    torch.distributed.barrier() 
    torch.cuda.empty_cache()
    while True:
        if (train_diffusiongan or text_image_pair_path is not None) and cur_nimg > images_per_zip:
            new_datapart = int(cur_nimg // images_per_zip) 
            if datapart < new_datapart:
                datapart = new_datapart
                
                # Clear any pending operations
                torch.cuda.synchronize()
                
                # Delete old iterators and dataset
                if dataset_latents_iterator is not None:
                    del dataset_latents_iterator
                if dataset_latents_iterator_text is not None:
                    del dataset_latents_iterator_text
                if hf_dataset is not None:
                    hf_dataset.cleanup_cache_files()
                    hf_dataset.set_format(None)
                    del hf_dataset
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                
                if precompute_latents:
                    precomputed_latents_path = os.path.join(
                            cache_latents_dir, f"{dataset_name}_{datapart}.{'text_image_latents' if train_diffusiongan else 'text_latents'}.pt"
                        )
                else:
                    precomputed_latents_path = None
                #rank_device = torch.device(f"cuda:{get_rank() % dist.get_world_size()}")
                rank_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
                context_manager_vae = (
                    contextlib.nullcontext(None) 
                    if not precompute_latents else
                    temporary_model_on_device(vae, rank_device)
                    if train_diffusiongan and (precomputed_latents_path is not None and not os.path.exists(precomputed_latents_path)) else
                    contextlib.nullcontext(vae)
                )
                context_manager_text_encoding_pipeline = (
                    contextlib.nullcontext(None)
                    if not precompute_latents else
                    #temporary_model_on_device(text_encoding_pipeline, rank_device)
                    temporary_model_on_device(text_encoders, rank_device)
                    if precomputed_latents_path is not None and not os.path.exists(precomputed_latents_path) else
                    #contextlib.nullcontext(text_encoding_pipeline)
                    contextlib.nullcontext(text_encoders)
                )
                print("precomputed_latents_path1", precomputed_latents_path)
                with context_manager_text_encoding_pipeline:
                    with context_manager_vae:
                        dataset_latents_iterator, dataset_latents_iterator_text, hf_dataset = stream_data_sampler(
                            datapart=datapart,
                            text_image_pair_path=text_image_pair_path,
                            cache_latents_dir=cache_latents_dir,
                            #dataset_name=processed_dataset_name,
                            #text_encoding_pipeline=text_encoding_pipeline,
                            text_encoders=text_encoders,
                            tokenizers=tokenizers,
                            vae=vae,
                            resolution=resolution,
                            train_diffusiongan=train_diffusiongan,
                            uncond_embeds=uncond_embeds,
                            #uncond_attention_mask=uncond_attention_mask,
                            uncond_pooled_embeds=uncond_pooled_embeds,
                            batch_gpu=batch_gpu,
                            seed=seed,
                            data_loader_kwargs=data_loader_kwargs,
                            precomputed_latents_path=precomputed_latents_path,
                            text_feature_dtype=text_feature_dtype,
                        )
                dist.print0(f"Switched to dataset part: {datapart}, cur_nimg: {cur_nimg}")
                
                # Ensure all processes are synchronized
                torch.distributed.barrier()
                
                # Clear memory again after loading
                gc.collect()
                torch.cuda.empty_cache()
                

                
        num_steps_random = torch.randint(1, num_steps+1, (1,)).item() 
        
        # Ensure all ranks use the same random number of steps per iteration
        if get_rank() == 0:
            num_steps_random_tensor = torch.randint(1, num_steps + 1, (1,), device="cuda", dtype=torch.int)
        else:
            num_steps_random_tensor = torch.empty(1, device="cuda", dtype=torch.int)
        torch.distributed.broadcast(num_steps_random_tensor, src=0)
        num_steps_random = num_steps_random_tensor.item()
        torch.distributed.barrier()
        
        #dist.print0(f"Iteration{iteration}  num_steps_random: {num_steps_random}")
        
        #torch.cuda.empty_cache()
        iteration =iteration+1
        #torch.distributed.barrier() 
        # We only wrap true score with FSDP in FSDP and true_score_fsdp mode 
       
        if use_ddp:           
            if fake_score_ddp.device != device:
                fake_score_ddp.to(device=device)
        

        if use_ddp:
            if use_grad_turn_on_off:
                G_ddp.train()
                fake_score_ddp.train()
                G_ddp.requires_grad_(False)
                fake_score_ddp.requires_grad_(True)
                
        else:
            
            if use_grad_turn_on_off:
                G_fsdp.train()
                fake_score_fsdp.train()
                G_fsdp.requires_grad_(False)
                fake_score_fsdp.requires_grad_(True)
                
        fake_score_optimizer.zero_grad(set_to_none=True)        

        Len_noise=0

        #rank_device = torch.device(f"cuda:{get_rank() % dist.get_world_size()}")
        rank_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        context_manager_vae = (
            contextlib.nullcontext(None) 
            if not precompute_latents else
            temporary_model_on_device(vae, rank_device)
            if train_diffusiongan and not precompute_latents else
            contextlib.nullcontext(vae)
        )
        context_manager_text_encoding_pipeline = (
            contextlib.nullcontext((None))
            if not precompute_latents else
            #temporary_model_on_device(text_encoding_pipeline, rank_device)
            temporary_model_on_device(text_encoders, rank_device)
            if precomputed_latents_path is not None and not os.path.exists(precomputed_latents_path) else
            #contextlib.nullcontext(text_encoding_pipeline)
            contextlib.nullcontext(text_encoders)
        )
        with context_manager_text_encoding_pipeline:
            with context_manager_vae:
                for round_idx in range(num_accumulation_rounds):
                    with torch.autocast(device_type="cuda",dtype=dtype_autocast,enabled=use_autocast):
                        #dist.print0(f"Iteration{iteration}  Prepare 7")
                        if train_diffusiongan:
                            if 0: # use_lmdb:
                                images_real,contexts_real = next(dataset_latents_iterator)
                                images_real=images_real.to(device) #.to(dtype)
                                images_real,contexts_real = next(dataset_latents_iterator)
                                images_real=images_real.to(device) #.to(dtype)
                            else:
                                #
                                batch = next(dataset_latents_iterator)
                                #images_real = batch["image"].to(dtype=torch.float32, device=device)
                                images_real = batch["image"].to(dtype=dtype, device=device)
                                contexts_real = batch["text"]
                                if not precompute_latents:
                                    with torch.no_grad():
                                        #images_real = vae.encode(images_real).latent*vae.config.scaling_factor
                                        images_real = vae.encode(images_real).latent_dist.sample()
                                        images_real = (images_real - vae.config.shift_factor) * vae.config.scaling_factor
                                    images_real=images_real.to(device)
                                    prompt_embeds_real = None
                                    #prompt_attention_mask_real = None
                                    pooled_prompt_embeds_real = None
                                else:
                                    prompt_embeds_real = batch["prompt_embeds"].to(device=device)
                                    #prompt_attention_mask_real = batch["prompt_attention_mask"].to(device=device)
                                    pooled_prompt_embeds_real = batch["pooled_prompt_embeds"].to(device=device)
                            batch= next(dataset_latents_iterator_text)
                            contexts = batch["text"]
                            if precompute_latents:
                                prompt_embeds = batch["prompt_embeds"].to(device=device)
                                #prompt_attention_mask = batch["prompt_attention_mask"].to(device=device)
                                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=device)
                            else:
                                prompt_embeds = None
                                #prompt_attention_mask = None
                                pooled_prompt_embeds = None
                                
                        else:
                            if text_image_pair_path is not None:
                                batch = next(dataset_latents_iterator_text)
                                contexts = batch["text"]
                                prompt_embeds = batch["prompt_embeds"].to(device=device) if precompute_latents else None
                                #prompt_attention_mask = batch["prompt_attention_mask"].to(device=device) if precompute_latents else None
                                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=device) if precompute_latents else None

                            else:
                                _, contexts  = next(dataset_prompt_text_iterator)
                                

                        if use_context_dropout_train_fake:
                            bool_tensor = torch.rand(batch_gpu) < 0.1
                            if train_diffusiongan:
                                contexts_real = ["" if flag else caption for flag, caption in zip(bool_tensor.tolist(), contexts_real)]
                                contexts = ["" if flag else caption for flag, caption in zip(bool_tensor.tolist(), contexts)]
                            else:
                                contexts = ["" if flag else caption for flag, caption in zip(bool_tensor.tolist(), contexts)]

                        # initialize latents z and noise
                        z = torch.randn([batch_gpu, latent_img_channels, latent_img_resolution, latent_img_resolution], device=device, dtype=dtype)
                        noise = torch.randn_like(z)

                        init_timesteps = init_timestep * torch.ones((batch_gpu,), device=device, dtype=torch.long)

                        # if use_ddp:
                        #     sync_context = misc.ddp_sync
                        #     G_fsdp=G_ddp
                        # else:
                        #     sync_context = fsdp_sync

                        
                        
                        #dist.print0(f"Iteration{iteration}  Start FSDP computation 0")
                        with fsdp_sync(fake_score_fsdp, (round_idx == num_accumulation_rounds - 1)):
                            with torch.no_grad():
                                images = sid_dit_generate(
                                    dit=G_fsdp,
                                    latents=z,
                                    contexts=contexts,
                                    init_timesteps=init_timesteps,
                                    noise_scheduler=noise_scheduler,
                                    #text_encoding_pipeline=text_encoding_pipeline,
                                    text_encoders=text_encoders,
                                    tokenizers=tokenizers,
                                    resolution=resolution,
                                    dtype=dtype,
                                    return_images=False,
                                    vae=None,
                                    num_steps=num_steps,
                                    train_sampler=False,
                                    num_steps_eval=num_steps_random,
                                    time_scale=time_scale,
                                    uncond_embeds=uncond_embeds,
                                    #uncond_attention_mask=uncond_attention_mask,
                                    uncond_pooled_embeds=uncond_pooled_embeds,
                                    noise_type=noise_type,
                                    prompt_embeds=prompt_embeds,
                                    #prompt_attention_mask=prompt_attention_mask,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    use_sd3_shift=use_sd3_shift,
                                    text_feature_dtype=text_feature_dtype,
                                )
                                    
                            if 0: #use_sd3_shift:
                                logit_mean = 0.0
                                logit_std = 1.0
                                shift = 3.0
                                
                                timesteps = torch.nn.functional.sigmoid(
                                        torch.normal(mean=logit_mean, std=logit_std, size=(batch_gpu,), device=device)
                                    ).to(dtype=dtype)
                                shift=3.0
                                timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
                                sigmas=timesteps
                                
                            else:
                                logit_mean = torch.log(torch.tensor(2.0))
                                logit_std = 1.6
                        
                                timesteps = torch.nn.functional.sigmoid(
                                        torch.normal(mean=logit_mean, std=logit_std, size=(batch_gpu,), device=device)
                                    ).to(dtype=dtype)
                                sigmas=timesteps    
                                
                            
                            if train_diffusiongan:
                                target = noise-images.detach()
                                
                                
                                output, logit_fake = sid_dit_denoise(
                                    dit=fake_score_fsdp,
                                    images=images.detach(),
                                    noise=noise,
                                    contexts=contexts,
                                    timesteps=timesteps,
                                    noise_scheduler=noise_scheduler,
                                    #text_encoding_pipeline=text_encoding_pipeline,
                                    text_encoders=text_encoders,
                                    tokenizers=tokenizers,
                                    resolution=resolution,
                                    dtype=dtype,
                                    predict_x0=False,
                                    guidance_scale=cfg_train_fake,
                                    time_scale=time_scale,
                                    uncond_embeds=uncond_embeds,
                                    #uncond_attention_mask=uncond_attention_mask,
                                    uncond_pooled_embeds=uncond_pooled_embeds,
                                    return_flag='encoder_decoder',
                                    prompt_embeds=prompt_embeds,
                                    #prompt_attention_mask=prompt_attention_mask,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    text_feature_dtype=text_feature_dtype,
                                )
                                
                            else:
                                
                                target = noise-images.detach()
                                output = sid_dit_denoise(
                                    dit=fake_score_fsdp,
                                    images=images.detach(),
                                    noise=noise,
                                    contexts=contexts,
                                    timesteps=timesteps,
                                    noise_scheduler=noise_scheduler,
                                    #text_encoding_pipeline=text_encoding_pipeline,
                                    text_encoders=text_encoders,
                                    tokenizers=tokenizers,
                                    resolution=resolution,
                                    dtype=dtype,
                                    predict_x0=False,
                                    guidance_scale=cfg_train_fake,
                                    time_scale=time_scale,
                                    uncond_embeds=uncond_embeds,
                                    #uncond_attention_mask=uncond_attention_mask,
                                    uncond_pooled_embeds=uncond_pooled_embeds,
                                    prompt_embeds=prompt_embeds,
                                    #prompt_attention_mask=prompt_attention_mask,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    text_feature_dtype=text_feature_dtype,
                                )


                            nan_mask = torch.isnan(output).flatten(start_dim=1).any(dim=1)
                            nan_mask = nan_mask | torch.isnan(target).flatten(start_dim=1).any(dim=1)

                            if train_diffusiongan:
                                nan_mask = nan_mask | torch.isnan(logit_fake).flatten(start_dim=1).any(dim=1)

                            # Check if there are any NaN values present
                            if nan_mask.any():
                                # Invert the nan_mask to get a mask of samples without NaNs
                                non_nan_mask = ~nan_mask
                                # Filter out samples with NaNs from y_real and y_fake
                                target = target[non_nan_mask]
                                output = output[non_nan_mask]
                                if train_diffusiongan:
                                    logit_fake = logit_fake[non_nan_mask]
                                del non_nan_mask,nan_mask
                            
                            if use_autocast:
                                loss = torch.nn.functional.mse_loss(target, output, reduction="sum")  
                            else:
                                loss = torch.nn.functional.mse_loss(target.to(torch.float32), output.to(torch.float32), reduction="sum")
                            loss=loss.sum().mul(loss_scaling / batch_gpu_total)

                            del images, contexts

                            if train_diffusiongan:
                                if noise.shape != images_real.shape:
                                    dist.print0(f"Warning: noise shape {noise.shape} does not match images_real shape {images_real.shape}. Regenerating noise.")
                                    noise = torch.randn_like(images_real)
                                
                                logit_real = sid_dit_denoise(
                                    dit=fake_score_fsdp,
                                    images=images_real,
                                    noise=noise,
                                    contexts=contexts_real,
                                    timesteps=timesteps,
                                    noise_scheduler=noise_scheduler,
                                    #text_encoding_pipeline=text_encoding_pipeline,
                                    text_encoders=text_encoders,
                                    tokenizers=tokenizers,
                                    resolution=resolution,
                                    dtype=dtype,
                                    predict_x0=False,
                                    guidance_scale=cfg_train_fake,
                                    time_scale=time_scale,
                                    uncond_embeds=uncond_embeds,
                                    #uncond_attention_mask=uncond_attention_mask,
                                    uncond_pooled_embeds=uncond_pooled_embeds,
                                    return_flag='encoder',
                                    prompt_embeds=prompt_embeds_real,
                                    #prompt_attention_mask=prompt_attention_mask_real,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    text_feature_dtype=text_feature_dtype,
                                )
                                real_labels = torch.ones_like(logit_real)
                                fake_labels = torch.zeros_like(logit_fake)

                                sigmas=timesteps
                                #weight = 1/(sigmas**2)
                                #weight = weight.view(-1,1,1,1)
                                #weight = ((1-sigmas)**2).clip(min=0.00001).view(-1,1,1,1) 
                                weight = 1

                                bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
                                if not use_autocast:
                                    logit_real = logit_real.to(dtype=torch.float32)
                                    logit_fake = logit_fake.to(dtype=torch.float32)

                                loss_real = bce_loss(logit_real.clamp(-10, 10), real_labels)
                                loss_fake = bce_loss(logit_fake.clamp(-10, 10), fake_labels)
                                loss_real = loss_real.view(loss_real.shape[0], -1).mean(dim=1, keepdim=True)
                                loss_fake = loss_fake.view(loss_fake.shape[0], -1).mean(dim=1, keepdim=True) 
                                loss_D = weight*(loss_real + loss_fake) / 2
                                loss_D = loss_D *images_real.shape[1]*images_real.shape[2]*images_real.shape[3]
                                loss_D = loss_D.sum().mul(loss_scaling_D / batch_gpu_total)
                                
                                #del images_real, contexts_real, prompt_embeds_real, prompt_attention_mask_real
                                del images_real, contexts_real, prompt_embeds_real, pooled_prompt_embeds_real


                            Len_noise = Len_noise+len(noise)
                            loss_fake_score_print=0
                            lossD_print=0
                            if len(noise) > 0:
                                if (use_autocast and dtype_autocast==torch.float16) or dtype==torch.float16:
                                    if train_diffusiongan:
                                        scaler.scale(loss+loss_D).backward()
                                    else:
                                        scaler.scale(loss).backward()
                                else:
                                    if train_diffusiongan:
                                        (loss+loss_D).backward()
                                    else:
                                        loss.backward()
                        

                loss_fake_score_print = loss.item()
                lossD_print = loss_D.item() if train_diffusiongan else 0

             
        training_stats.report('fake_score_Loss/loss', loss_fake_score_print)
        training_stats.report('D_Loss/loss', lossD_print)
            
        if use_ddp:
            fake_score_fsdp=fake_score_ddp.module
        
        if Len_noise>0:
            if (use_autocast and dtype_autocast==torch.float16) or dtype==torch.float16:
                scaler.unscale_(fake_score_optimizer)
                
            #for param in fake_score.parameters():
            for param in fake_score_fsdp.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                    #torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            
            if not use_autocast:
                torch.nn.utils.clip_grad_value_(fake_score_fsdp.parameters(), 1)
            
        if (use_autocast and dtype_autocast==torch.float16) or dtype==torch.float16:
            scaler.step(fake_score_optimizer)
            scaler.update()
        else:
            fake_score_optimizer.step()

        try:
            del z, noise,  loss
        except NameError as e:
            dist.print0(f"No such variable to delete: {e}")
        
        torch.cuda.empty_cache()
        
        fake_score_optimizer.zero_grad(set_to_none=True)   

        # for p in fake_score_fsdp.parameters():
        #     if p.grad is not None:
        #         del p.grad  # or: p.grad = None

        if use_train_eval:
            if use_ddp:
                G_ddp.train().requires_grad_(True)
                fake_score_ddp.eval().requires_grad_(False)
            else:
                G_fsdp.train().requires_grad_(True)
                fake_score_fsdp.eval().requires_grad_(False)
        #----------------------------------------------------------------------------------------------
        # Update One-Step Generator Network
        #if use_train_eval:              
        #    G_ddp.train().requires_grad_(True)

        if use_grad_turn_on_off:
            if use_ddp:
                fake_score_ddp.requires_grad_(False)
                G_ddp.requires_grad_(True)                                          
            else:                                         
                fake_score_fsdp.requires_grad_(False)
                G_fsdp.requires_grad_(True)
#                     if guidance_scale==100:
#                         fake_score_fsdp.module.apply(disable_bn_dropout)
#                         G_fsdp.module.apply(lambda m: restore_bn_dropout(m, name=str(m)))

            # for p in fake_score_ddp.module.parameters():
            #     p.requires_grad = False    
            # for p in G_ddp.module.parameters():
            #     p.requires_grad = True

        g_optimizer.zero_grad(set_to_none=True)
        Len_noise=0


        #rank_device = torch.device(f"cuda:{get_rank() % dist.get_world_size()}")
        rank_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        context_manager_vae = (
            contextlib.nullcontext(None) 
            if not precompute_latents else
            temporary_model_on_device(vae, rank_device)
            if train_diffusiongan and not precompute_latents else
            contextlib.nullcontext(vae)
        )
        context_manager_text_encoding_pipeline = (
            contextlib.nullcontext(None)
            if not precompute_latents else
            #temporary_model_on_device(text_encoding_pipeline, rank_device)
            temporary_model_on_device(text_encoders, rank_device)
            if precomputed_latents_path is not None and not os.path.exists(precomputed_latents_path) else
            #contextlib.nullcontext(text_encoding_pipeline)
            contextlib.nullcontext(text_encoders)
        )
        with context_manager_text_encoding_pipeline:
            for round_idx in range(num_accumulation_rounds):
                with torch.autocast(device_type="cuda",dtype=dtype_autocast,enabled=use_autocast):
                    
                    if train_diffusiongan:
                        batch = next(dataset_latents_iterator_text)
                        contexts_real = batch["text"]
                        contexts=contexts_real
                        prompt_embeds_real = batch["prompt_embeds"].to(device=device) if precompute_latents else None
                        #prompt_attention_mask_real = batch["prompt_attention_mask"].to(device=device) if precompute_latents else None
                        pooled_prompt_embeds_real = batch["pooled_prompt_embeds"].to(device=device) if precompute_latents else None
                        prompt_embeds = prompt_embeds_real
                        #prompt_attention_mask = prompt_attention_mask_real
                        pooled_prompt_embeds = pooled_prompt_embeds_real
                    else:
                        if text_image_pair_path is not None:
                            batch = next(dataset_latents_iterator_text)
                            contexts = batch["text"]
                            prompt_embeds = batch["prompt_embeds"].to(device=device) if precompute_latents else None
                            #prompt_attention_mask = batch["prompt_attention_mask"].to(device=device) if precompute_latents else None
                            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=device) if precompute_latents else None
                        else:
                            _, contexts  = next(dataset_prompt_text_iterator)           
                    
                    z = torch.randn([batch_gpu, latent_img_channels, latent_img_resolution, latent_img_resolution], device=device, dtype=dtype)
                    noise = torch.randn_like(z)

                    # initialize timesteps
                    init_timesteps = init_timestep * torch.ones((batch_gpu,), device=device, dtype=torch.long)
                    #timesteps = torch.randint(20, 980, (batch_gpu,), device=device, dtype=torch.long)
                        
                    

                    
                    if 0: #use_sd3_shift:
                        logit_mean = 0.0
                        logit_std = 1.0
                        shift = 3.0
                        
                        timesteps = torch.nn.functional.sigmoid(
                                torch.normal(mean=logit_mean, std=logit_std, size=(batch_gpu,), device=device)
                            ).to(dtype=dtype)
                        shift=3.0
                        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
                        sigmas=timesteps
                        
                    else:
                        logit_mean = torch.log(torch.tensor(2.0))
                        logit_std = 1.6
                
                        timesteps = torch.nn.functional.sigmoid(
                                torch.normal(mean=logit_mean, std=logit_std, size=(batch_gpu,), device=device)
                            ).to(dtype=dtype)
                        sigmas=timesteps

                    
                    
                    with fsdp_sync(G_fsdp,False):
                        #latent_model_input, t, prompt_embeds, prompt_attention_mask, latents = sid_dit_generate(
                        latent_model_input, t, prompt_embeds, pooled_prompt_embeds, latents = sid_dit_generate(
                            dit=G_fsdp,
                            latents=z,
                            contexts=contexts,
                            init_timesteps=init_timesteps,
                            noise_scheduler=noise_scheduler,
                            # text_encoding_pipeline=text_encoding_pipeline,
                            text_encoders=text_encoders,
                            tokenizers=tokenizers,
                            resolution=resolution,
                            dtype=dtype,
                            return_images=False,
                            vae=None,
                            num_steps=num_steps,
                            train_sampler=True,
                            num_steps_eval=num_steps_random,
                            time_scale=time_scale,
                            uncond_embeds=uncond_embeds,
                            #uncond_attention_mask=uncond_attention_mask,
                            uncond_pooled_embeds=uncond_pooled_embeds,
                            noise_type=noise_type,
                            prompt_embeds=prompt_embeds,
                            #prompt_attention_mask=prompt_attention_mask,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            use_sd3_shift=use_sd3_shift,
                            text_feature_dtype=text_feature_dtype,
                        )
                        latent_model_input = latent_model_input.detach()
                        latents = latents.detach()

                    with fsdp_sync(G_fsdp, (round_idx == num_accumulation_rounds - 1)):
                        
                        images = sid_dit_generate(
                            dit=G_fsdp,
                            latents=latents,
                            contexts=contexts,
                            init_timesteps=init_timesteps,
                            noise_scheduler=noise_scheduler,
                            #text_encoding_pipeline=text_encoding_pipeline,
                            text_encoders=text_encoders,
                            tokenizers=tokenizers,
                            resolution=resolution,
                            dtype=dtype,
                            return_images=False,
                            vae=None,
                            num_steps=num_steps,
                            train_sampler=True,
                            num_steps_eval=num_steps_random,
                            time_scale=time_scale,
                            uncond_embeds=uncond_embeds,
                            #uncond_attention_mask=uncond_attention_mask,
                            uncond_pooled_embeds=uncond_pooled_embeds,
                            latent_model_input=latent_model_input,
                            prompt_embeds=prompt_embeds,
                            #prompt_attention_mask=prompt_attention_mask,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            t=t,   
                            noise_type=noise_type,
                            use_sd3_shift=use_sd3_shift,
                            text_feature_dtype=text_feature_dtype,
                        )                                        
                            
                    
                        if train_diffusiongan:
                        
                            

                            y_fake,y_D = sid_dit_denoise(
                                    dit=fake_score_fsdp,
                                    images=images,
                                    noise=noise,
                                    contexts=contexts,
                                    timesteps=timesteps,
                                    noise_scheduler=noise_scheduler,
                                    #text_encoding_pipeline=text_encoding_pipeline,
                                    text_encoders=text_encoders,
                                    tokenizers=tokenizers,
                                    resolution=resolution,
                                    dtype=dtype,
                                    guidance_scale=cfg_eval_fake,
                                    time_scale=time_scale,
                                    uncond_embeds=uncond_embeds,
                                    #uncond_attention_mask=uncond_attention_mask,
                                    uncond_pooled_embeds=uncond_pooled_embeds,
                                    return_flag='encoder_decoder',
                                    prompt_embeds=prompt_embeds,
                                    #prompt_attention_mask=prompt_attention_mask,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    text_feature_dtype=text_feature_dtype,
                                )
                        else:
                            
                            y_fake = sid_dit_denoise(
                                    dit=fake_score_fsdp,
                                    images=images,
                                    noise=noise,
                                    contexts=contexts,
                                    timesteps=timesteps,
                                    noise_scheduler=noise_scheduler,
                                    #text_encoding_pipeline=text_encoding_pipeline,
                                    text_encoders=text_encoders,
                                    tokenizers=tokenizers,
                                    resolution=resolution,
                                    dtype=dtype,
                                    guidance_scale=cfg_eval_fake,
                                    time_scale=time_scale,
                                    uncond_embeds=uncond_embeds,
                                    #uncond_attention_mask=uncond_attention_mask,
                                    uncond_pooled_embeds=uncond_pooled_embeds,
                                    prompt_embeds=prompt_embeds,
                                    #prompt_attention_mask=prompt_attention_mask,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    text_feature_dtype=text_feature_dtype,
                                )
                        
                        y_real = sid_dit_denoise(
                                dit=true_score,
                                images=images,
                                noise=noise,
                                contexts=contexts,
                                timesteps=timesteps,
                                noise_scheduler=noise_scheduler,
                                #text_encoding_pipeline=text_encoding_pipeline,
                                text_encoders=text_encoders,
                                tokenizers=tokenizers,
                                resolution=resolution,
                                dtype=dtype,
                                guidance_scale=cfg_eval_real,
                                time_scale=time_scale,
                                uncond_embeds=uncond_embeds,
                                #uncond_attention_mask=uncond_attention_mask,
                                uncond_pooled_embeds=uncond_pooled_embeds,
                                prompt_embeds=prompt_embeds,
                                #prompt_attention_mask=prompt_attention_mask,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                text_feature_dtype=text_feature_dtype,
                            )
                        

                        nan_mask_images = torch.isnan(images).flatten(start_dim=1).any(dim=1)
                        nan_mask_y_real = torch.isnan(y_real).flatten(start_dim=1).any(dim=1)
                        nan_mask_y_fake = torch.isnan(y_fake).flatten(start_dim=1).any(dim=1)
                        nan_mask = nan_mask_images | nan_mask_y_real | nan_mask_y_fake
                        if train_diffusiongan:
                            nan_mask_y_D = torch.isnan(y_D).flatten(start_dim=1).any(dim=1)
                            nan_mask = nan_mask  | nan_mask_y_D


                        # Check if there are any NaN values present
                        if nan_mask.any():
                            # Invert the nan_mask to get a mask of samples without NaNs
                            non_nan_mask = ~nan_mask
                            # Filter out samples with NaNs from y_real and y_fake
                            images = images[non_nan_mask]
                            y_real = y_real[non_nan_mask]
                            y_fake = y_fake[non_nan_mask]
                            sigmas = sigmas[non_nan_mask]
                            if train_diffusiongan:
                                y_D = y_D[non_nan_mask]

                        with torch.no_grad():
                            if not use_autocast:
                                sigmas = sigmas.to(dtype=torch.float32)

                            if weighting_scheme == "sid_legacy":
                                if images is None or y_real is None:
                                    raise ValueError("images and y_real required for sid_legacy weighting_scheme")
                                weight_factor = abs(images.to(torch.float32) - y_real.to(torch.float32)).mean(dim=[1, 2, 3], keepdim=True).clip(min=0.00001)
                                scale_factor = 1/weight_factor
                            elif weighting_scheme == "snr_sqrt":
                                # SNR_sqrt
                                weight_factor = ((sigmas/(1-sigmas))).clip(min=0.00001).view(-1,1,1,1)
                                scale_factor = 1/weight_factor
                            elif weighting_scheme == "snr":
                                # SNR
                                weight_factor = ((sigmas/(1-sigmas))**2).clip(min=0.00001).view(-1,1,1,1)
                                scale_factor = 1/weight_factor
                            elif weighting_scheme == "1_over_sigma2":
                                # 1/sigma^2
                                weight_factor = ((sigmas)**2).clip(min=0.00001).view(-1,1,1,1)
                                scale_factor = 1/weight_factor
                            elif weighting_scheme == "1_over_sigma":
                                # 1/sigma
                                weight_factor = sigmas.clip(min=0.00001).view(-1,1,1,1)
                                scale_factor = 1/weight_factor
                            elif weighting_scheme == "1_minus_sigma_squared":
                                # (1-sigma)^2
                                #weight_factor = (1/(1-sigmas)**2).clip(min=0.00001).view(-1,1,1,1)
                                scale_factor = ((1-sigmas)**2).view(-1,1,1,1)
                            elif weighting_scheme == "1_minus_sigma":
                                # (1-sigma) - default for SANA distillation
                                #weight_factor = (1/(1-sigmas)).clip(min=0.00001).view(-1,1,1,1)
                                scale_factor = (1-sigmas).view(-1,1,1,1)
                            else:
                                raise ValueError(f"Unknown weighting weighting_scheme: {weighting_scheme}. Available: sid_legacy, snr_sqrt, snr, 1_over_sigma2, 1_over_sigma, 1_minus_sigma_squared, 1_minus_sigma")
                        if not use_autocast:
                            y_real = y_real.to(dtype=torch.float32)
                            y_fake = y_fake.to(dtype=torch.float32)
                            images = images.to(dtype=torch.float32)
                            scale_factor = scale_factor.to(dtype=torch.float32)
                        if alpha==1:
                            loss = (y_real - y_fake) * (y_fake - images) *scale_factor #/ weight_factor
                        else:
                            loss = (y_real - y_fake) * ((y_real - images) - alpha * (y_real - y_fake)) *scale_factor #/ weight_factor
                        loss=loss.sum().mul(loss_scaling_G / batch_gpu_total)
                        lossG_print = loss.item()
                        loss_gan=0
                        lossG_gan_print=0    

                        if train_diffusiongan:
                            y_D_labels = torch.ones_like(y_D)
                            bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
                            loss_gan = bce_loss(y_D.clamp(-10, 10),y_D_labels) #if use_autocast else bce_loss(y_D.to(dtype=torch.float32).clamp(-10, 10),y_D_labels) 
                            loss_gan = loss_gan.view(loss_gan.shape[0], -1).mean(dim=1, keepdim=True)*scale_factor.view(loss_gan.shape[0], -1)
                            loss_gan = loss_gan*images.shape[1]*images.shape[2]*images.shape[3]
                            loss_gan = loss_gan.sum().mul(loss_scaling_G_gan / batch_gpu_total)
                            lossG_gan_print = loss_gan.item()

                        if len(y_real) > 0:
                            Len_noise=Len_noise+len(y_real)
                            if train_diffusiongan:
                                #if cur_nimg>200*1000:
                                if cur_nimg>40*1000:
                                    if (use_autocast and dtype_autocast==torch.float16) or dtype==torch.float16:
                                        scaler_G.scale(0.5*loss+0.5*loss_gan).backward()
                                    else:
                                        (0.5*loss+0.5*loss_gan).backward()
                                else:
                                    if (use_autocast and dtype_autocast==torch.float16) or dtype==torch.float16:
                                        scaler_G.scale(loss).backward()
                                    else:
                                        loss.backward()
                            else:
                                if (use_autocast and dtype_autocast==torch.float16) or dtype==torch.float16:
                                    scaler_G.scale(loss).backward()
                                else:
                                    loss.backward()                            

                        

        training_stats.report('G_Loss/loss', lossG_print)
        training_stats.report('G_gan_Loss/loss', lossG_gan_print)
        
        if use_ddp:
            G_fsdp=G_ddp.module
                                                
        
        if ((not train_diffusiongan) or cur_nimg>20*1000) and Len_noise > 0:
            if (use_autocast and dtype_autocast==torch.float16) or dtype==torch.float16:
                #if cur_nimg>100*1000 or sid_model is None:  
                scaler_G.unscale_(g_optimizer)
        
            for param in G_fsdp.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            
            if not use_autocast:
                torch.nn.utils.clip_grad_value_(G_fsdp.parameters(), 1)

            if (use_autocast and dtype_autocast==torch.float16) or dtype==torch.float16:
                scaler_G.step(g_optimizer)
                scaler_G.update()
            else:
                g_optimizer.step()
        
        try:
            del z,images,noise,y_fake,y_real,loss,loss_gan,sigmas, weight_factor, scale_factor 
        except NameError:
            pass
        torch.cuda.empty_cache()
        
        g_optimizer.zero_grad(set_to_none=True)    
        # for p in G_fsdp.parameters():
        #     if p.grad is not None:
        #         del p.grad  # or: p.grad = None

        if use_train_eval:
            fake_score_fsdp.train().requires_grad_(True)
            G_fsdp.eval().requires_grad_(False)

        if use_grad_turn_on_off:
            G_fsdp.requires_grad_(False)
            fake_score_fsdp.requires_grad_(True)
            
        if use_ddp:
            G_ema = G 
        
        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)

        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        fields += [f"loss_fake_score {training_stats.report0('fake_score_Loss/loss', loss_fake_score_print):<6.2f}"]
        fields += [f"loss_G {training_stats.report0('G_Loss/loss', lossG_print):<6.2f}"]
        fields += [f"loss_G_gan {training_stats.report0('G_gan_Loss/loss', lossG_gan_print):<6.2f}"]
        fields += [f"loss_D {training_stats.report0('D_Loss/loss', lossD_print):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0 or cur_tick in [5,10,50,100]):

            if  use_ddp or not true_score_fsdp:
                if ema_device!=device:    
                    true_score=true_score.to(device=ema_device)
            
            
         
            G_ema = None

            #rank_device = torch.device(f"cuda:{get_rank() % dist.get_world_size()}")
            rank_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
            #with temporary_model_on_device(text_encoding_pipeline, rank_device):

            context_manager_vae = (
                contextlib.nullcontext(None) 
                if not precompute_latents else
                temporary_model_on_device(vae, rank_device)
            )
            
            context_manager_text_encoding_pipeline = (
                contextlib.nullcontext(None)
                if not precompute_latents else
                temporary_model_on_device(text_encoders, rank_device)
            )
            #dist.print0("precompute_latents0", precompute_latents)
            with context_manager_text_encoding_pipeline:
                with context_manager_vae:

            #with temporary_model_on_device(text_encoders, rank_device):
                #with temporary_model_on_device(vae, rank_device):
                    
                    torch.distributed.barrier()
                    dist.print0('Exporting sample images...')
                    if (1 and get_rank() == 0) or (use_fsdp): # and dtype!=torch.float32:
                        # images = [sdxl_sample(unet=G_ema, latents=z, contexts=c, timesteps=init_timestep * torch.ones((1,), device=device, dtype=torch.long), 
                        #                       noise_scheduler=noise_scheduler, return_images=True) for z, c in zip(grid_z, grid_c)]
                        #for num_steps_eval in [1,2,3,4]:
                        List = [1,num_steps_eval] if num_steps_eval>1 else [1]
                        
                        for num_steps_sampler in List:
                        #if 1:
                            with torch.no_grad():
                                with torch.autocast(device_type="cuda",dtype=dtype_autocast,enabled=use_autocast):
                                    # Clear memory before generating images
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    
                                    images = []
                                    for z, c in zip(grid_z, grid_c):
                                        # Generate images in smaller batches to save memory
                                        batch_images = sid_dit_generate(
                                            dit=G_fsdp,
                                            latents=z,
                                            contexts=c,
                                            init_timesteps=init_timestep * torch.ones((len(c),), device=device, dtype=torch.long),
                                            noise_scheduler=noise_scheduler,
                                            #text_encoding_pipeline=text_encoding_pipeline,
                                            text_encoders=text_encoders,
                                            tokenizers=tokenizers,
                                            resolution=resolution,
                                            dtype=dtype,
                                            return_images=True,
                                            vae=vae,
                                            num_steps=num_steps_sampler,
                                            train_sampler=False,
                                            num_steps_eval=num_steps_sampler,
                                            time_scale=time_scale,
                                            uncond_embeds=uncond_embeds,
                                            #uncond_attention_mask=uncond_attention_mask,
                                            uncond_pooled_embeds=uncond_pooled_embeds,
                                            noise_type=noise_type,
                                            use_sd3_shift=use_sd3_shift,
                                            text_feature_dtype=text_feature_dtype,
                                        )
                                        images.append(batch_images)
                                        
                                        # Clean up intermediate tensors
                                        del batch_images
                                        torch.cuda.empty_cache()
                                    
                                if get_rank() == 0:
                                    # Concatenate and process images on rank 0
                                    images = torch.cat(images).to(torch.float32).cpu().numpy()
                                    save_image_grid(img=images, fname=os.path.join(run_dir, f'fakes_{alpha:03f}_{cur_nimg//1000:06d}_{num_steps_sampler:d}.png'), drange=[-1,1], grid_size=grid_size)
                                    
                                # Clean up after saving
                                del images
                                
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                        
                    if cur_tick>0:    
                        dist.print0('Evaluating metrics...')
                        dist.print0(metric_pt_path)
                        dist.print0(dist.get_world_size())
                        dist.print0(get_rank())
                        dist.print0('Evaluating metrics...')
                        #List = [num_steps] 
                        #for num_steps_sampler in List:
                        num_steps_sampler = num_steps
                        if metrics is not None:
                            for metric in metrics:
                                # Clear memory before each metric calculation
                                gc.collect()
                                torch.cuda.empty_cache()
                                
                                with torch.no_grad():
                                    with torch.autocast(device_type="cuda",dtype=dtype_autocast,enabled=use_autocast):
                                        result_dict = metric_main.calc_metric(
                                            metric=metric,
                                            metric_pt_path=metric_pt_path,
                                            metric_open_clip_path=metric_open_clip_path,
                                            metric_clip_path=metric_clip_path,
                                            G=partial(
                                                sid_dit_generate,
                                                dit=G_fsdp,
                                                noise_scheduler=noise_scheduler,
                                                #text_encoding_pipeline=text_encoding_pipeline,
                                                text_encoders=text_encoders,
                                                tokenizers=tokenizers,
                                                resolution=resolution,
                                                dtype=dtype,
                                                return_images=True,
                                                vae=vae,
                                                num_steps=num_steps_sampler,
                                                train_sampler=False,
                                                num_steps_eval=num_steps_sampler,
                                                time_scale=time_scale,
                                                uncond_embeds=uncond_embeds,
                                                #uncond_attention_mask=uncond_attention_mask,
                                                uncond_pooled_embeds=uncond_pooled_embeds,
                                                noise_type=noise_type,
                                                use_sd3_shift=use_sd3_shift,
                                                text_feature_dtype=text_feature_dtype,
                                            ),
                                            init_timestep=init_timestep,
                                            dataset_kwargs=dataset_kwargs,
                                            num_gpus=dist.get_world_size(),
                                            rank=get_rank(),
                                            local_rank=dist.get_local_rank(),
                                            device=device,
                                            latent_img_channels=latent_img_channels,
                                            latent_img_resolution=latent_img_resolution
                                        )
                                        current_fid = result_dict.results.fid10k_full
                                        current_clip_score = result_dict.results.open_clipscore_10k
                                        
                                # Clean up after each metric
                                
                                

                                if get_rank() == 0:
                                    dist.print0(result_dict.results)
                                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'fakes_{alpha:03f}_{cur_nimg//1000:06d}_{num_steps_sampler}step.png', alpha=alpha, num_steps_eval=num_steps_sampler)    
                                stats_metrics.update(result_dict.results)
                                del result_dict
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                    if G_ema is not None:
                        if G_ema.device!=ema_device:
                            G_ema=G_ema.to(ema_device)
                            torch.cuda.empty_cache()
                    
                    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    full_optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(
                        G_fsdp,
                        StateDictType.FULL_STATE_DICT,
                        full_state_dict_config,
                        full_optim_state_dict_config
                    ):
                        state_dict= G_fsdp.state_dict()
                    if get_rank() == 0:
                        if use_ddp:
                            data = dict(ema=G_ema)
                        else:
                            #G_ema_fsdp = type(G)(**G.config).eval().requires_grad_(False)  
                            G_ema_fsdp= type(G_fsdp.module)(**G_fsdp.module.config)
                            G_ema_fsdp.eval().requires_grad_(False)
                            #G_ema_fsdp.load_state_dict(state_dict)
                            G_ema_fsdp.load_state_dict(state_dict, strict=False)
                            data = dict(ema = G_ema_fsdp)
                            del state_dict

                        for key, value in data.items():
                            if isinstance(value, torch.nn.Module):
                                with torch.no_grad():
                                    # 1) Remember the original device (likely GPU)
                                    original_device = next(value.parameters()).device

                                    # 2) Move the *original* module to CPU to avoid GPU overhead during deepcopy
                                    value.cpu()

                                    # 3) Deepcopy on CPU
                                    value_cpu_copy = copy.deepcopy(value).to(torch.float16)
                                    
                                    #value_cpu_copy = value.detach().clone().to(torch.float16)
                                    
                                    #value.detach().clone().to(torch.float16)

                                    # 4) Finalize the copy for inference (eval mode + no grad)
                                    value_cpu_copy.eval().requires_grad_(False)

                                    # 5) Move original back to GPU (or whatever device it was on)
                                    value.to(original_device)

                                    # 6) Store the CPU-side snapshot in data
                                    data[key] = value_cpu_copy

                            # Optionally, delete the local variable to free references.
                            del value
                        
                        if not save_best_and_last:
                            save_data(data=data, fname=os.path.join(run_dir, f'network-snapshot-{alpha:03f}-{cur_nimg//1000:06d}.pkl'))
                        else:
                            step_str = f'{alpha:03f}-{cur_nimg//1000:06d}'
                            checkpoint_path = os.path.join(run_dir, f'network-snapshot-{step_str}.pkl')

                            # Save best FID
                            if current_fid < best_fid:
                                # Save new checkpoint
                                save_data(data=data, fname=checkpoint_path)
                        
                                # Remove previous best FID checkpoint if it's not also best CLIP
                                if best_fid_step is not None and best_fid_step != best_clip_step:
                                    old_fid_path = os.path.join(run_dir, f'network-snapshot-{best_fid_step}.pkl')
                                    if os.path.exists(old_fid_path):
                                        try:
                                            os.remove(old_fid_path)
                                        except OSError as e:
                                            dist.print0(f"Error removing old FID checkpoint: {e}")
                        
                                # Update best FID
                                best_fid = current_fid
                                best_fid_step = step_str
                        
                            # Save best CLIP
                            if current_clip_score > best_clip_score:
                                # Save new checkpoint
                                save_data(data=data, fname=checkpoint_path)
                        
                                # Remove previous best CLIP checkpoint if it's not also best FID
                                if best_clip_step is not None and best_clip_step != best_fid_step:
                                    old_clip_path = os.path.join(run_dir, f'network-snapshot-{best_clip_step}.pkl')
                                    if os.path.exists(old_clip_path):
                                        try:
                                            os.remove(old_clip_path)
                                        except OSError as e:
                                            dist.print0(f"Error removing old CLIP checkpoint: {e}")
                        
                                # Update best CLIP
                                best_clip_score = current_clip_score
                                best_clip_step = step_str
                        try:
                            del data,G_ema_fsdp,G_ema
                            gc.collect()
                            torch.cuda.empty_cache()
                        except:
                            pass
                        
        

        if use_ddp or not true_score_fsdp:    
            if true_score.device!=device:
                true_score=true_score.to(device=device) #,dtype=dtype)
        
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0 or cur_tick==2) and cur_tick != 0:
           
            
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            
            if use_ddp:
                if get_rank() == 0:
                    dist.print0(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_G')
                    save_pt(pt=dict(G_state=G_ddp.module.state_dict() if use_ddp else G_state, g_optimizer_state=g_optimizer.state_dict() if use_ddp else g_optimizer_state), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_G'))#,dtype=torch.float16)
                    dist.print0(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_fake')
                    save_pt(pt=dict(fake_score_state=fake_score_ddp.module.state_dict() if use_ddp else fake_score_state, fake_score_optimizer_state=fake_score_optimizer.state_dict() if use_ddp else fake_score_optimizer_state), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_fake'))#,dtype=torch.float16)
                    dist.print0(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt')
                    save_pt(pt=dict(G_ema_state=G_ema.state_dict()), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))#,dtype=torch.float16)
            else:
                torch.distributed.barrier()
                
                if 0:
                    torch.distributed.barrier()


                    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    full_optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

                    with FSDP.state_dict_type(
                        G_fsdp,
                        StateDictType.FULL_STATE_DICT,
                        full_state_dict_config,
                        full_optim_state_dict_config
                    ):
                        G_state = G_fsdp.state_dict()
                        g_optimizer_state_dict = FSDP.optim_state_dict(G_fsdp, g_optimizer)


                    if get_rank() == 0:
                        dist.print0(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_G')
                        save_pt(pt=dict(G_state= G_state), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_G')) #,dtype=torch.float16)

                    if get_rank()==(1%dist.get_world_size()):
                        print(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_G_optim')
                        save_pt(pt=dict(g_optimizer_state=g_optimizer_state_dict), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_G_optim')) #,dtype=torch.float16)
                    torch.distributed.barrier()
                    del G_state, g_optimizer_state_dict
                    torch.cuda.empty_cache()


                    with FSDP.state_dict_type(
                        fake_score_fsdp,
                        StateDictType.FULL_STATE_DICT,
                        full_state_dict_config,
                        full_optim_state_dict_config
                    ):
                        fake_score_state = fake_score_fsdp.state_dict()
                        fake_score_optimizer_state_dict = FSDP.optim_state_dict(fake_score_fsdp, fake_score_optimizer)

                    # if not fake_score_optimizer_state_dict:

                    #     dist.print0("Optimizer state dictionary is empty.")

                    # else:

                    #     dist.print0("Optimizer state dictionary contains data.")

                    if get_rank() == (2%dist.get_world_size()):
                        print(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_fake')
                        save_pt(pt=dict(fake_score_state=fake_score_state), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_fake'))#,dtype=torch.float16)
                        
                    if get_rank() == (3%dist.get_world_size()):
                        print(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_fake_optim')
                        save_pt(pt=dict(fake_score_optimizer_state=fake_score_optimizer_state_dict), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_fake_optim'))#,dtype=torch.float16)
                    torch.distributed.barrier()
                    del fake_score_state, fake_score_optimizer_state_dict 
                    torch.cuda.empty_cache()
                else:
                    torch.distributed.barrier()
                    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    full_optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(
                        G_fsdp,
                        StateDictType.FULL_STATE_DICT,
                        full_state_dict_config,
                        full_optim_state_dict_config
                    ):
                        G_state = G_fsdp.state_dict()
                        #g_optimizer_state_dict = FSDP.optim_state_dict(G_fsdp, g_optimizer)

                    if get_rank() == 0:
                        dist.print0(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_G')
                        save_pt(pt=dict(G_state= G_state), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_G')) #,dtype=torch.float16)

                    # if get_rank()==(1%dist.get_world_size()):
                    #     print(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_G_optim')
                    #     save_pt(pt=dict(g_optimizer_state=g_optimizer_state_dict), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_G_optim')) #,dtype=torch.float16)
                    torch.distributed.barrier()
                    del G_state
                    torch.cuda.empty_cache()

                    with FSDP.state_dict_type(
                        G_fsdp,
                        StateDictType.FULL_STATE_DICT,
                        full_state_dict_config,
                        full_optim_state_dict_config
                    ):
                        #G_state = G_fsdp.state_dict()
                        g_optimizer_state_dict = FSDP.optim_state_dict(G_fsdp, g_optimizer)



                    if get_rank()==0: #(1%dist.get_world_size()):
                        print(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_G_optim')
                        save_pt(pt=dict(g_optimizer_state=g_optimizer_state_dict), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_G_optim')) #,dtype=torch.float16)
                    torch.distributed.barrier()
                    del g_optimizer_state_dict
                    torch.cuda.empty_cache()

                    



                    with FSDP.state_dict_type(
                        fake_score_fsdp,
                        StateDictType.FULL_STATE_DICT,
                        full_state_dict_config,
                        full_optim_state_dict_config
                    ):
                        fake_score_state = fake_score_fsdp.state_dict()
                        #fake_score_optimizer_state_dict = FSDP.optim_state_dict(fake_score_fsdp, fake_score_optimizer)



                    # if not fake_score_optimizer_state_dict:

                    #     dist.print0("Optimizer state dictionary is empty.")

                    # else:

                    #     dist.print0("Optimizer state dictionary contains data.")



                                    

                    if get_rank() == 0: #(2%dist.get_world_size()):
                        print(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_fake')
                        save_pt(pt=dict(fake_score_state=fake_score_state), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_fake'))#,dtype=torch.float16)


                    # if get_rank() == (3%dist.get_world_size()):

                    #     print(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_fake_optim')

                    #     save_pt(pt=dict(fake_score_optimizer_state=fake_score_optimizer_state_dict), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_fake_optim'))#,dtype=torch.float16)

                    torch.distributed.barrier()

                    del fake_score_state #, fake_score_optimizer_state_dict 





                    with FSDP.state_dict_type(

                        fake_score_fsdp,

                        StateDictType.FULL_STATE_DICT,

                        full_state_dict_config,

                        full_optim_state_dict_config

                    ):

                        #fake_score_state = fake_score_fsdp.state_dict()

                        fake_score_optimizer_state_dict = FSDP.optim_state_dict(fake_score_fsdp, fake_score_optimizer)



                    # if not fake_score_optimizer_state_dict:

                    #     dist.print0("Optimizer state dictionary is empty.")

                    # else:

                    #     dist.print0("Optimizer state dictionary contains data.")



                                    

                    # if get_rank() == (2%dist.get_world_size()):

                    #     print(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_fake')

                    #     save_pt(pt=dict(fake_score_state=fake_score_state), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_fake'))#,dtype=torch.float16)

                        

                    if get_rank() == 0: #(3%dist.get_world_size()):

                        print(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt_fake_optim')

                        save_pt(pt=dict(fake_score_optimizer_state=fake_score_optimizer_state_dict), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt_fake_optim'))#,dtype=torch.float16)

                    torch.distributed.barrier()

                    del fake_score_optimizer_state_dict 

                    torch.cuda.empty_cache()

                    

            

            


                

                if get_rank() == 0:
                    dist.print0(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt')
                    save_pt(pt=dict(G_ema_state=None), #G_ema.state_dict()), 
                                            fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt')) #,dtype=torch.float16)
                #del G_state
                
                if save_best_and_last is True and get_rank() == 0:
                    try:
                        if previous_pt_filename is not None:
                            #if os.path.exists(previous_pt_filename):
                            #    os.remove(previous_pt_filename)
                            # Define checkpoint suffixes
                            suffixes = ["", "_fake", "_fake_optim", "_G", "_G_optim"]
                            # Construct full paths
                            filenames = [f"{previous_pt_filename}{suffix}" for suffix in suffixes]
                            # Remove any existing checkpoint files
                            for filename in filenames:
                                if os.path.exists(filename):
                                    os.remove(filename)
                                    dist.print0(f"Removed: {filename}")
                                else:
                                    dist.print0(f"Not found (skipped): {filename}")

                    except OSError as e:
                        dist.print0(f"Error removing previous checkpoint: {e}")
        
                    previous_pt_filename = os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt')
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize() 
                

            
        # Update logs.
        training_stats.default_collector.update()
        if get_rank() == 0:
            if stats_jsonl is None:
                append_line(jsonl_line=json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n', fname=os.path.join(run_dir, f'stats_{alpha:03f}.jsonl'))

        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0('Exiting...')

#----------------------------------------------------------------------------