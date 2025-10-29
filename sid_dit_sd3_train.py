#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

# Example: Copy and paste this command to launch distributed training (4 GPUs):

"""
Examples command (reduce batch-gpu to fit your GPU memory):\
    # --data is required for COCO-2014 metric evaluation. It is not required for data-free training (set --metrics to None)
    # For data-free training, specify either --data_prompt_text or --text_image_pair_path.
    # To enable Diffusion GAN loss, --text_image_pair_path must be provided.
    # --noise_type can be fresh (ddpm styles, injecting noise at every step), fixed, or ddim
    # --weighting_scheme can be sid_legend, snr_sqrt, snr, 1_over_sigma2, 1_over_sigma, 1_minus_sigma_squared, 1_minus_sigma, or any custom function (we suggest 1_minus_sigma as the default)
    # --train_diffusiongan can be 0 or 1 to enable or disable Diffusion GAN loss
    # --nosubdir can be 0 or 1 to enable or disable creating a subdirectory for results
    # --duration is the training duration in millions of training images
    # --ls is the loss scaling factor for the fake score network
    # --lsg is the loss scaling factor for the generator
    # --metrics is the metric (fid and clip) to evaluate the training progress
    # --noise_type is the noise type for the training data
    # --init_timestep defines the intitial time to start the reverse process
    # --num_steps is the number of generation steps
    # --fp16=False, --bf16=False, --autocast_bf16=True: Autocast with bf16 mixed precision
    # --gradient_checkpointing = 1 enables gradient checkpointing that reduces memory usage at the cost of slower training speed

    torchrun --standalone --nproc_per_node=8 sid_dit_train.py \ 
    --outdir data/image_experiment/sid_dit \
    --resume data/image_experiment/sid_dit \
    --data data/datasets/MS-COCO-256/val \
    --data_prompt_text data/datasets/aesthetics_6_plus \
    --text_image_pair_path data/datasets/midjourney-v6-llava/data \
    --dit_model Efficient-Large-Model/Sana_600M_512px_diffusers \
    --optimizer adam \
    --resolution 512 \
    --batch 256 \
    --batch-gpu 16 \
    --lr 0.000005 \
    --glr 0.000005 \
    --cfg_train_fake 4.5 \
    --cfg_eval_fake 4.5 \
    --cfg_eval_real 4.5 \
    --alpha 1.0 \
    --init_timestep 999 \
    --num_steps 4 \
    --fp16 0 \
    --bf16 0 \
    --autocast_bf16 1 \
    --gradient_checkpointing 1 \
    --tick 2 \
    --snap 25 \
    --dump 25 \
    --duration 2 \
    --ls 1 \
    --lsg 100 \
    --metrics fid10k_full \
    --noise_type fresh \
    --weighting_scheme 1_minus_sigma \
    --train_diffusiongan 0 \
    --nosubdir

"""


"""Distill DiT-based diffusion/flow-matching models using the SiD few-step techniques described in the
paper "Score Distillation of Flow"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import sid_dit_sd3_training_loop as training_loop

# --- Helper Classes & Functions ---
def parse_int_list(s):
    """Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    """
    if isinstance(s, list): 
        return s
    
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def find_latest_checkpoint(directory):
    """Find the latest training state checkpoint file in a directory and its subdirectories.
    
    Args:
        directory: The path to the directory to search in.
        
    Returns:
        Tuple of (latest_file_path, latest_number). Returns (None, -1) if no checkpoint found.
    """
    latest_file = None
    latest_number = -1
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("training-state-") and file.endswith(".pt"):
                # Extract the number from the file name
                number_part = file[len("training-state-"):-len(".pt")]
                try:
                    number = int(number_part)
                    if number > latest_number:
                        latest_number = number
                        latest_file = os.path.join(root, file)
                except ValueError:
                    # If the number part is not an integer, ignore this file
                    continue
    
    return latest_file, latest_number

class CommaSeparatedList(click.ParamType):
    """Click parameter type for comma-separated lists."""
    name = 'list'
    
    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()

# Main options
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=False)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)

# Hyperparameters
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=10, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--lr',            help='Learning rate of fake score estimation network', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1e-5, show_default=True)
@click.option('--glr',           help='Learning rate of fake data generator', metavar='FLOAT',      type=click.FloatRange(min=0, min_open=True), default=1e-5, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=float, default=0, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='FLOAT',                    type=float, default=0.0, show_default=True)

# Performance-related
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--bf16',          help='Enable bf16', metavar='BOOL',                                type=bool, default=False, show_default=True)
@click.option('--autocast_bf16', help='Enable bf16', metavar='BOOL',                                type=bool, default=True, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=0), default=1, show_default=True)
@click.option('--gradient_checkpointing', help='gradient_checkpointing', metavar='BOOL',            type=bool, default=True, show_default=True)

# I/O-related
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=25, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=25, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

# Adapted from Diff-Instruct
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full,fid30k_full,fid10k_full]',      default=None, type=CommaSeparatedList())
@click.option('--dit_model',     help='edm_model', type=str)
@click.option('--resolution',    help='Clip images in data to resolution', metavar='INT',          type=int, default=256, show_default=True)

# Parameters for SiD
@click.option('--init_timestep', help='Noise standard deviation that is fixed during distillation and generation', metavar='INT', type=int, default=50, show_default=True)
@click.option('--alpha',         help='L2-alpha*L1', metavar='FLOAT',                              type=click.FloatRange(min=-1000, min_open=True), default=1.0, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--lsg',           help='Loss scaling G', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=100, show_default=True)
@click.option('--lsd',           help='Loss scaling D', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--lsg_gan',       help='Loss scaling G_gan', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--num_steps',     help='Number of generation steps (NFEs)', metavar='INT', type=int, default=4, show_default=True)

@click.option('--optimizer',     help='Optimizer',     metavar='adam|adamw', type=str, default='adam', show_default=True)

# FID metric PT path
@click.option('--metric_pt_path',  help='Where the metric pt locates on',     metavar='DIR', type=str, default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt', show_default=True)
@click.option('--metric_clip_path',  help='Where the metric clip file locates on ',     metavar='DIR', type=str)
@click.option('--metric_open_clip_path',  help='Where the metric open clip file locates',     metavar='DIR', type=str,default='clipvitg14.pkl')

# Guidance scales
@click.option('--cfg_train_fake', help='Guidance scale in training fake. Default value is 1.0.', metavar='FLOAT', type=float, default=1, show_default=True)
@click.option('--cfg_eval_fake',  help='Guidance scale in evaluating fake. Default value is 1.0.', metavar='FLOAT', type=float, default=1, show_default=True)
@click.option('--cfg_eval_real',  help='Guidance scale in evaluating fake. Default value is 1.0.', metavar='FLOAT', type=float, default=1, show_default=True)

# Data free options
@click.option('--data_prompt_text', help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
# Image-text pair options
@click.option('--text_image_pair_path', help='Path to image latents', metavar='ZIP|DIR',                     type=str, default=None, required=False)

# Initialization options
@click.option('--sid_model',     help='sid_model', type=str,default=None, show_default=True)

@click.option('--pooling_type',  help='channel|spatial', type=str,default='spatial', show_default=True)
@click.option('--vae_model',     help='vae_model',default=None, type=str)

### FSDP options
@click.option('--cpu_offload',   help='Offload states to cpu?', metavar='BOOL',            type=bool, default=False)

# SANA specific
@click.option('--time_scale',    help='scaling t by 1, 1000 (sana sprint distill code on regular sana teacher), or 0.0001 (sana distill on sana sprint teacher)', metavar='FLOAT', type=float, default=None, show_default=True)

# Ablation options
@click.option('--noise_type',    help='Noise type for generation: fresh, fixed, or ddim', metavar='STR', type=click.Choice(['fresh', 'fixed', 'ddim']), default='fresh', show_default=True)
@click.option('--weighting_scheme', help='Loss weighting scheme: sid_legacy, snr_sqrt, snr, 1_over_sigma2, 1_over_sigma, 1_minus_sigma_squared, 1_minus_sigma', metavar='STR', type=click.Choice(['sid_legacy', 'snr_sqrt', 'snr', '1_over_sigma2', '1_over_sigma', '1_minus_sigma_squared', '1_minus_sigma']), default='1_minus_sigma', show_default=True)
@click.option('--train_diffusiongan', help='Train DiffusionGAN (debug)', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--use_sd3_shift', help='Use SD3 shift', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--text_encoders_dtype', help='Text feature dtype', metavar='STR', type=click.Choice(['float16','bfloat16','float32','fp16','bf16','fp32']), default=None, show_default=True)

def main(**kwargs):
    """Main entry point for SiD-DiT training."""
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict
    c = dnnlib.EasyDict()
    
    if opts.metrics is not None:
        c.dataset_kwargs = dnnlib.EasyDict(
            class_name='training.mscoco_dataset.ImageDataset', 
            path=opts.data, 
            resolution=opts.resolution, 
            random_flip=opts.xflip
        )
    
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2 if opts.workers > 0 else None)

    c.dataset_prompt_text_kwargs = dnnlib.EasyDict(
        class_name='training.aesthetics_dataset.ImageDataset', 
        path=opts.data_prompt_text, 
        resolution=opts.resolution, 
        random_flip=opts.xflip, 
        prompt_only=True
    )
    
    c.network_kwargs = dnnlib.EasyDict()
    #c.loss_kwargs = dnnlib.EasyDict()

    c.fake_score_optimizer_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam', 
        lr=opts.lr, 
        betas=[0.0, 0.999], 
        eps=1e-8 if (not opts.fp16 and not opts.bf16) else 1e-4
    )
    c.g_optimizer_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam', 
        lr=opts.glr, 
        betas=[0.0, 0.999], 
        eps=1e-8 if (not opts.fp16 and not opts.bf16) else 1e-4
    )
      
    c.init_timestep = opts.init_timestep

    # Validate dataset options
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        data_max_size = len(dataset_obj)  # be explicit about dataset size
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
        
    # Image-text pair dataset configuration
    if opts.text_image_pair_path is not None:
        c.dataset_latents_kwargs = dnnlib.EasyDict(
            class_name='training.sd_latents_dataset.SDImageDatasetLMDB', 
            dataset_path=opts.text_image_pair_path, 
            #local_path=opts.local_path, 
            #dataset_cache=opts.cache_path, 
            resolution=opts.resolution, 
            random_flip=opts.xflip
        )
    else:
        c.dataset_latents_kwargs = None
    
    c.text_image_pair_path = opts.text_image_pair_path
    c.metrics = opts.metrics

    c.network_kwargs.update(use_fp16=opts.fp16, use_bf16=opts.bf16, autocast_bf16=opts.autocast_bf16)

    # Training options
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    
    # Batch and performance settings
    c.update(
        batch_size=opts.batch, 
        batch_gpu=opts.batch_gpu,
        cudnn_benchmark=opts.bench,
        kimg_per_tick=opts.tick, 
        snapshot_ticks=opts.snap, 
        state_dump_ticks=opts.dump,
        gradient_checkpointing=opts.gradient_checkpointing,
    )
    c.train_diffusiongan = opts.train_diffusiongan
    
    # Loss scaling parameters
    c.update(
        loss_scaling=opts.ls, 
        loss_scaling_G=opts.lsg,
        loss_scaling_D=opts.lsd, 
        loss_scaling_G_gan=opts.lsg_gan
    )
    
    # SiD-specific parameters
    c.alpha = opts.alpha
    c.noise_type = opts.noise_type
    c.weighting_scheme = opts.weighting_scheme

    # Random seed
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    if opts.resume is not None:
        if os.path.isdir(opts.resume):
            # Get the latest checkpoint file and number from the directory
            latest_file, latest_number = find_latest_checkpoint(opts.resume)
            if latest_file is None:
                # No checkpoint found in the directory
                c.resume_training = None
                c.resume_kimg = 0
            else:
                # Set the latest checkpoint file and the corresponding kimg number
                c.resume_training = latest_file
                c.resume_kimg = latest_number
        else:
            # If opts.resume is a file, validate its name and existence
            match = re.fullmatch(r'training-state-(\d+)\.pt', os.path.basename(opts.resume))
            if not match or not os.path.isfile(opts.resume):
                # Invalid file, reset resume options
                c.resume_training = None
                c.resume_kimg = 0
            else:
                # Valid checkpoint file, extract kimg number from filename
                c.resume_training = opts.resume
                c.resume_kimg = int(match.group(1))

    # Description string
    cond_str = 'text_cond'
    if c.network_kwargs.use_fp16:
        dtype_str = 'fp16'
    else:
        dtype_str = 'bf16' if c.network_kwargs.use_bf16 else 'fp32'
    
    desc = f'{dataset_name:s}-{cond_str:s}-glr{opts.glr}-lr{opts.lr}-initsigma{opts.init_timestep}-gpus{dist.get_world_size():d}-alpha{c.alpha}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    if opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    c.metric_pt_path = opts.metric_pt_path
    c.metric_open_clip_path = opts.metric_open_clip_path
    c.metric_clip_path = opts.metric_clip_path

    c.pretrained_model_name_or_path = opts.dit_model
    c.pretrained_vae_model_name_or_path = opts.dit_model if opts.vae_model is None else opts.vae_model

    c.cfg_train_fake = opts.cfg_train_fake
    c.cfg_eval_fake = opts.cfg_eval_fake
    c.cfg_eval_real = opts.cfg_eval_real
    c.num_steps = opts.num_steps
    
    c.resolution = opts.resolution
    
    c.sid_model = opts.sid_model
    c.pooling_type = opts.pooling_type
    c.cpu_offload = opts.cpu_offload
    
    if opts.time_scale is not None:    
        c.time_scale = opts.time_scale
    else:
        c.time_scale = 1 if 'sprint' in opts.dit_model.lower() else 1000
    c.use_sd3_shift = opts.use_sd3_shift
    c.text_encoders_dtype = opts.text_encoders_dtype

    # Print options
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Dataset length:          {data_max_size}')
    dist.print0(f'Class-conditional:       text_cond')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0(f'alpha:                   {c.alpha}')
    dist.print0(f'precision:               {dtype_str}')
    dist.print0(f'metric_pt_path:          {c.metric_pt_path}')
    dist.print0(f'pretrained_model_name_or_path: {c.pretrained_model_name_or_path}')
    dist.print0(f'pretrained_vae_model_name_or_path: {c.pretrained_vae_model_name_or_path}')
    dist.print0(f'resolution: {c.resolution}')
    dist.print0()

    # Dry run check
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory
    dist.print0('Creating output directory...')
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Start training
    training_loop.training_loop(**c)
    
# --- Script Entry Point ---
if __name__ == "__main__":
    main()