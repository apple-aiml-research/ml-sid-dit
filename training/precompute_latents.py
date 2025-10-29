#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
import torchvision.transforms as T
from PIL import Image
import io
import numpy as np
import gc

from training.sid_dit_sd3_util import encode_prompt    

class RGBConvert:
    """Picklable transform that converts PIL images to RGB."""
    def __call__(self, img):
        return img.convert("RGB")

class Text2ImageDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading text-image pairs, either from raw HuggingFace dataset
    or from precomputed latents.

    Args:
        hf_dataset: Raw HuggingFace dataset (required if not using precomputed latents)
        resolution: Image resolution for transforms (default 1024)
        precomputed_latents_path: Path to .pt file with precomputed latents (if using)
    """

    def __init__(self, hf_dataset, resolution=1024, precomputed_latents_path=None,sd3=False, text_feature_dtype=None):
        self.precomputed_latents_path = precomputed_latents_path
        self.sd3 = sd3
        self.text_feature_dtype = text_feature_dtype
        if precomputed_latents_path is not None:
            self.precomputed_latents = torch.load(precomputed_latents_path)
            # Clear memory after loading latents
            torch.cuda.empty_cache()
            gc.collect()
        else:
            self.dataset = hf_dataset
            self.transform = self.get_image_transform(resolution)
            # self.transform = T.Compose(
            #     [
            #         RGBConvert(),              # Safe alternative to Lambda
            #         T.Resize(resolution),
            #         T.CenterCrop(resolution),
            #         T.ToTensor(),
            #         T.Normalize([0.5], [0.5]),
            #     ]
            # )

    def __del__(self):
        # Clean up when dataset is deleted
        if hasattr(self, 'precomputed_latents'):
            del self.precomputed_latents
        if hasattr(self, 'dataset'):
            del self.dataset
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def get_image_transform(resolution=1024):
        return T.Compose([
            RGBConvert(),
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        if hasattr(self, "precomputed_latents_path") and self.precomputed_latents_path is not None:
            return len(self.precomputed_latents)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.precomputed_latents_path is not None:
            item = self.precomputed_latents[idx]
            if self.sd3:
                prompt_embeds = item["prompt_embeds"]
                pooled_prompt_embeds = item["pooled_prompt_embeds"]
                if self.text_feature_dtype is not None:
                    prompt_embeds = prompt_embeds.to(dtype=self.text_feature_dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.text_feature_dtype)
                return {
                    "text": item["text"],
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "image": item["image"],
                }
            else:
                prompt_embeds = item["prompt_embeds"]
                if self.text_feature_dtype is not None:
                    prompt_embeds = prompt_embeds.to(dtype=self.text_feature_dtype)
                prompt_attention_mask = item["prompt_attention_mask"].to(dtype=torch.bool)
                return {
                    "text": item["text"],
                    "prompt_embeds": prompt_embeds,
                    "prompt_attention_mask": prompt_attention_mask,
                    "image": item["image"],
                }
        else:
            item = self.dataset[int(idx) if isinstance(idx, np.integer) else idx]
            text = item["llava"]
            image_bytes = item["image"]

            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            image_tensor = self.transform(image)

            return {"text": text, "image": image_tensor}

    
class TextDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading text-image pairs from a HuggingFace dataset.
    """

    def __init__(self, hf_dataset, resolution=1024, precomputed_latents_path=None, sd3=False, text_feature_dtype=None):
        self.precomputed_latents_path = precomputed_latents_path
        self.sd3 = sd3
        self.text_feature_dtype = text_feature_dtype
        if precomputed_latents_path is not None:
            self.precomputed_latents = torch.load(precomputed_latents_path)
            # Clear memory after loading latents
            torch.cuda.empty_cache()
            gc.collect()
        else:
            self.dataset = hf_dataset
        
    def __del__(self):
        # Clean up when dataset is deleted
        if hasattr(self, 'precomputed_latents'):
            del self.precomputed_latents
        if hasattr(self, 'dataset'):
            del self.dataset
        gc.collect()
        torch.cuda.empty_cache()

    def __len__(self):
        if hasattr(self, "precomputed_latents_path") and self.precomputed_latents_path is not None:
            return len(self.precomputed_latents)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.precomputed_latents_path is not None:
            item = self.precomputed_latents[idx]
            # Print shapes before squeeze
            prompt_embeds = item["prompt_embeds"]
            if self.sd3:
                pooled_prompt_embeds = item["pooled_prompt_embeds"]
            else:
                prompt_attention_mask = item["prompt_attention_mask"]
            # Print shapes after squeeze
            if self.sd3:
                return {
                    "text": item["text"],
                    "prompt_embeds": prompt_embeds if self.text_feature_dtype is None else prompt_embeds.to(dtype=self.text_feature_dtype),
                    "pooled_prompt_embeds": pooled_prompt_embeds if self.text_feature_dtype is None else pooled_prompt_embeds.to(dtype=self.text_feature_dtype),
                }
            else:
                return {
                    "text": item["text"],
                    "prompt_embeds": prompt_embeds,
                    "prompt_attention_mask": prompt_attention_mask.to(dtype=torch.bool),
                }
            
        else:
            item = self.dataset[int(idx) if isinstance(idx, np.integer) else idx]
            text = item["llava"]
            return {"text": text}

 

def process_and_save_latents_with_pipeline_multigpu(
    hf_dataset,
    text_encoding_pipeline,
    vae,
    latents_path,
    resolution,
    world_size,
    rank,
    uncond_embeds=None,
    uncond_attention_mask=None,
    train_diffusiongan=False,
    batch_size=16,
    use_sd3=False
):
    import torch
    from tqdm import tqdm
    from PIL import Image
    import io
    from training.sid_dit_util import COMPLEX_HUMAN_INSTRUCTION
    import os
    import gc

    def ensure_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, list):
            return [ensure_cpu(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: ensure_cpu(v) for k, v in obj.items()}
        else:
            return obj

    gc.collect()
    torch.cuda.empty_cache()

    vae.eval()
    image_transform = Text2ImageDataset.get_image_transform(resolution)
    N = len(hf_dataset)
    indices = list(range(rank, N, world_size))

    results = []
    for start in tqdm(range(0, len(indices), batch_size), desc=f"Rank {rank} encoding latents"):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]
        batch_prompts = []
        image_tensors = []
        raw_images = []

        for idx in batch_indices:
            item = hf_dataset[idx]
            batch_prompts.append(item["llava"])
            if train_diffusiongan:
                image = Image.open(io.BytesIO(item["image"])).convert("RGB")
                raw_images.append(image)
                image_tensor = image_transform(image)
                image_tensors.append(image_tensor)

        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                batch_prompts,
                complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION,
                do_classifier_free_guidance=False,
            )

            prompt_embeds = prompt_embeds.cpu()
            prompt_attention_mask = prompt_attention_mask.cpu()

            if uncond_embeds is not None and uncond_attention_mask is not None:
                for i, p in enumerate(batch_prompts):
                    if not p.strip():
                        prompt_embeds[i] = uncond_embeds[i]
                        prompt_attention_mask[i] = uncond_attention_mask[i]

        if train_diffusiongan:
            image_tensor_batch = torch.stack(image_tensors).to(vae.device)
            with torch.no_grad():
                if use_sd3:
                    #image_latents = vae.encode(image_tensor_batch).latent_dist.sample() * vae.config.scaling_factor
                    #bug fixed july 29, 2025
                    image_latents = vae.encode(image_tensor_batch.to(vae.dtype)).latent_dist.sample()
                    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                else:
                    image_latents = vae.encode(image_tensor_batch.to(vae.dtype)).latent * vae.config.scaling_factor
            image_latents = image_latents.cpu()
            del image_tensor_batch
            for img in raw_images:
                del img
        else:
            image_latents = [None] * len(batch_indices)

        for i, idx in enumerate(batch_indices):
            results.append({
                "text": batch_prompts[i],
                "prompt_embeds": ensure_cpu(prompt_embeds[i]),
                "prompt_attention_mask": ensure_cpu(prompt_attention_mask[i]),
                "image": ensure_cpu(image_latents[i]) if train_diffusiongan else "",
                "original_idx": idx,
            })

        del prompt_embeds, prompt_attention_mask, image_latents
        gc.collect()
        torch.cuda.empty_cache()

    part_path = latents_path + f".rank{rank}"
    torch.save(results, part_path)
    del results
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier()

    if rank == 0:
        all_latents = []
        for r in range(world_size):
            rank_results = torch.load(latents_path + f".rank{r}")
            all_latents.extend(rank_results)
            del rank_results
            os.remove(latents_path + f".rank{r}")
            gc.collect()
            torch.cuda.empty_cache()

        all_latents.sort(key=lambda x: x["original_idx"])
        for latent in all_latents:
            latent.pop("original_idx", None)
        all_latents = [ensure_cpu(item) for item in all_latents]
        torch.save(all_latents, latents_path)
        del all_latents
        gc.collect()
        torch.cuda.empty_cache()

    torch.distributed.barrier()



def process_and_save_latents_with_pipeline_multigpu_sd3(
    hf_dataset,
    #text_encoding_pipeline,
    text_encoders,
    tokenizers,
    vae,
    latents_path,
    resolution,
    world_size,
    rank,
    uncond_embeds=None,
    #uncond_attention_mask=None,
    uncond_pooled_embeds=None,
    train_diffusiongan=False,
    batch_size=16,
    use_sd3=False
):
    import torch
    from tqdm import tqdm
    from PIL import Image
    import io
    #from training.sid_dit_util import COMPLEX_HUMAN_INSTRUCTION
    import os
    import gc

    def ensure_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, list):
            return [ensure_cpu(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: ensure_cpu(v) for k, v in obj.items()}
        else:
            return obj

    gc.collect()
    torch.cuda.empty_cache()

    vae.eval()
    image_transform = Text2ImageDataset.get_image_transform(resolution)
    N = len(hf_dataset)
    indices = list(range(rank, N, world_size))

    results = []
    for start in tqdm(range(0, len(indices), batch_size), desc=f"Rank {rank} encoding latents"):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]
        batch_prompts = []
        image_tensors = []
        raw_images = []

        for idx in batch_indices:
            item = hf_dataset[idx]
            batch_prompts.append(item["llava"])
            if train_diffusiongan:
                image = Image.open(io.BytesIO(item["image"])).convert("RGB")
                raw_images.append(image)
                image_tensor = image_transform(image)
                image_tensors.append(image_tensor)

        with torch.no_grad():
            #prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders,
                tokenizers,
                batch_prompts,
                max_sequence_length=77,
                device=text_encoders[0].device,
                num_images_per_prompt=1,
            )

            prompt_embeds = prompt_embeds.cpu()
            #prompt_attention_mask = prompt_attention_mask.cpu()
            pooled_prompt_embeds = pooled_prompt_embeds.cpu()

            # #if uncond_embeds is not None and uncond_pooled_embeds is not None:
            # if uncond_embeds is not None and uncond_pooled_embeds is not None:
            #     for i, p in enumerate(batch_prompts):
            #         if not p.strip():
            #             prompt_embeds[i] = uncond_embeds[i]
            #             #prompt_attention_mask[i] = uncond_attention_mask[i]
            #             pooled_prompt_embeds[i] = uncond_pooled_embeds[i]

        if train_diffusiongan:
            image_tensor_batch = torch.stack(image_tensors).to(vae.device)
            with torch.no_grad():
                if use_sd3:
                    #image_latents = vae.encode(image_tensor_batch).latent_dist.sample() * vae.config.scaling_factor
                    #bug fixed july 29, 2025
                    image_latents = vae.encode(image_tensor_batch.to(vae.dtype)).latent_dist.sample()
                    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                else:
                    image_latents = vae.encode(image_tensor_batch.to(vae.dtype)).latent * vae.config.scaling_factor
            image_latents = image_latents.cpu()
            del image_tensor_batch
            for img in raw_images:
                del img
        else:
            image_latents = [None] * len(batch_indices)

        for i, idx in enumerate(batch_indices):
            results.append({
                "text": batch_prompts[i],
                "prompt_embeds": ensure_cpu(prompt_embeds[i]),
                #"prompt_attention_mask": ensure_cpu(prompt_attention_mask[i]),
                "pooled_prompt_embeds": ensure_cpu(pooled_prompt_embeds[i]),
                "image": ensure_cpu(image_latents[i]) if train_diffusiongan else "",
                "original_idx": idx,
            })

        #del prompt_embeds, prompt_attention_mask, image_latents
        del prompt_embeds, pooled_prompt_embeds, image_latents
        gc.collect()
        torch.cuda.empty_cache()

    part_path = latents_path + f".rank{rank}"
    torch.save(results, part_path)
    del results
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier()

    if rank == 0:
        all_latents = []
        for r in range(world_size):
            rank_results = torch.load(latents_path + f".rank{r}")
            all_latents.extend(rank_results)
            del rank_results
            os.remove(latents_path + f".rank{r}")
            gc.collect()
            torch.cuda.empty_cache()

        all_latents.sort(key=lambda x: x["original_idx"])
        for latent in all_latents:
            latent.pop("original_idx", None)
        all_latents = [ensure_cpu(item) for item in all_latents]
        torch.save(all_latents, latents_path)
        del all_latents
        gc.collect()
        torch.cuda.empty_cache()

    torch.distributed.barrier()



def process_and_save_latents_with_pipeline_multigpu_flux(
    hf_dataset,
    #text_encoding_pipeline,
    text_encoders,
    tokenizers,
    vae,
    latents_path,
    resolution,
    world_size,
    rank,
    uncond_embeds=None,
    #uncond_attention_mask=None,
    uncond_pooled_embeds=None,
    train_diffusiongan=False,
    batch_size=16,
    use_flux=True
):
    from training.sid_dit_flux_util import encode_prompt as  encode_prompt_flux     

    import torch
    from tqdm import tqdm
    from PIL import Image
    import io
    #from training.sid_dit_util import COMPLEX_HUMAN_INSTRUCTION
    import os
    import gc

    def ensure_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, list):
            return [ensure_cpu(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: ensure_cpu(v) for k, v in obj.items()}
        else:
            return obj

    gc.collect()
    torch.cuda.empty_cache()

    vae.eval()
    image_transform = Text2ImageDataset.get_image_transform(resolution)
    N = len(hf_dataset)
    indices = list(range(rank, N, world_size))

    results = []
    for start in tqdm(range(0, len(indices), batch_size), desc=f"Rank {rank} encoding latents"):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]
        batch_prompts = []
        image_tensors = []
        raw_images = []

        for idx in batch_indices:
            item = hf_dataset[idx]
            batch_prompts.append(item["llava"])
            if train_diffusiongan:
                image = Image.open(io.BytesIO(item["image"])).convert("RGB")
                raw_images.append(image)
                image_tensor = image_transform(image)
                image_tensors.append(image_tensor)

        with torch.no_grad():
            #prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
            prompt_embeds, pooled_prompt_embeds,_ = encode_prompt_flux(
                text_encoders,
                tokenizers,
                batch_prompts,
                max_sequence_length=512,
                device=text_encoders[0].device,
                num_images_per_prompt=1,
            )

            prompt_embeds = prompt_embeds.cpu()
            #prompt_attention_mask = prompt_attention_mask.cpu()
            pooled_prompt_embeds = pooled_prompt_embeds.cpu()

            #if uncond_embeds is not None and uncond_pooled_embeds is not None:
            # if uncond_embeds is not None and uncond_pooled_embeds is not None:
            #     for i, p in enumerate(batch_prompts):
            #         if not p.strip():
            #             prompt_embeds[i] = uncond_embeds[i]
            #             #prompt_attention_mask[i] = uncond_attention_mask[i]
            #             pooled_prompt_embeds[i] = uncond_pooled_embeds[i]

        if train_diffusiongan:
            image_tensor_batch = torch.stack(image_tensors).to(vae.device)
            with torch.no_grad():
                if use_flux:
                    #image_latents = vae.encode(image_tensor_batch).latent_dist.sample() * vae.config.scaling_factor
                    #bug fixed july 29, 2025
                    image_latents = vae.encode(image_tensor_batch.to(vae.dtype)).latent_dist.sample()
                    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                else:
                    image_latents = vae.encode(image_tensor_batch.to(vae.dtype)).latent * vae.config.scaling_factor
            image_latents = image_latents.cpu()
            del image_tensor_batch
            for img in raw_images:
                del img
        else:
            image_latents = [None] * len(batch_indices)

        for i, idx in enumerate(batch_indices):
            results.append({
                "text": batch_prompts[i],
                "prompt_embeds": ensure_cpu(prompt_embeds[i]),
                #"prompt_attention_mask": ensure_cpu(prompt_attention_mask[i]),
                "pooled_prompt_embeds": ensure_cpu(pooled_prompt_embeds[i]),
                "image": ensure_cpu(image_latents[i]) if train_diffusiongan else "",
                "original_idx": idx,
            })

        #del prompt_embeds, prompt_attention_mask, image_latents
        del prompt_embeds, pooled_prompt_embeds, image_latents
        gc.collect()
        torch.cuda.empty_cache()

    part_path = latents_path + f".rank{rank}"
    torch.save(results, part_path)
    del results
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier()

    if rank == 0:
        all_latents = []
        for r in range(world_size):
            rank_results = torch.load(latents_path + f".rank{r}")
            all_latents.extend(rank_results)
            del rank_results
            os.remove(latents_path + f".rank{r}")
            gc.collect()
            torch.cuda.empty_cache()

        all_latents.sort(key=lambda x: x["original_idx"])
        for latent in all_latents:
            latent.pop("original_idx", None)
        all_latents = [ensure_cpu(item) for item in all_latents]
        torch.save(all_latents, latents_path)
        del all_latents
        gc.collect()
        torch.cuda.empty_cache()

    torch.distributed.barrier()

# def process_and_save_latents_with_pipeline_multigpu(
#     hf_dataset,
#     text_encoding_pipeline,
#     vae,
#     latents_path,
#     resolution,
#     world_size,
#     rank,
#     uncond_embeds=None,
#     uncond_attention_mask=None,
#     train_diffusiongan=False
# ):
#     """
#     Multi-GPU version: Each rank encodes a slice of hf_dataset, writes a part file.
#     Rank 0 merges parts, sorts, saves as latents_path.
#     """
#     import torch
#     from tqdm import tqdm
#     from PIL import Image
#     import io
#     from training.sid_dit_util import COMPLEX_HUMAN_INSTRUCTION
#     import os
#     import gc

#     def ensure_cpu(obj):
#         if isinstance(obj, torch.Tensor):
#             return obj.cpu()
#         elif isinstance(obj, list):
#             return [ensure_cpu(x) for x in obj]
#         elif isinstance(obj, dict):
#             return {k: ensure_cpu(v) for k, v in obj.items()}
#         else:
#             return obj

#     # Clear memory before starting
#     gc.collect()
#     torch.cuda.empty_cache()

#     vae.eval()
#     image_transform = Text2ImageDataset.get_image_transform(resolution)
#     N = len(hf_dataset)
#     indices = list(range(rank, N, world_size))

#     results = []
#     for idx in tqdm(indices, desc=f"Rank {rank} encoding latents"):
#         # Clear memory at start of each batch
#         if idx % 100 == 0:  # Every 100 items
#             gc.collect()
#             torch.cuda.empty_cache()
            
#         item = hf_dataset[idx]
#         # Text encoding (pipeline)
#         prompts = item["llava"]
#         max_sequence_length=300
#         with torch.no_grad():
#             prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
#                 prompts, complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION, do_classifier_free_guidance=False,
#             )
#             if (uncond_embeds is not None) and (uncond_attention_mask is not None):
#                 for i, p in enumerate(prompts):
#                     if not p.strip():
#                         prompt_embeds[i] = uncond_embeds[i]
#                         prompt_attention_mask[i] = uncond_attention_mask[i]
            
#             # Don't squeeze - we need to match uncond_embeds shape for concatenation later
#             prompt_embeds = prompt_embeds.squeeze(0).cpu()
#             prompt_attention_mask = prompt_attention_mask.squeeze(0).cpu()
            
#         # Image encoding
#         if train_diffusiongan:
#             image = Image.open(io.BytesIO(item["image"])).convert("RGB")
#             image_tensor = image_transform(image).unsqueeze(0) #.to(device)
#             with torch.no_grad():
#                 # VAE encode, get latents, scale by vae.scaling_factor to match training loop convention
#                 image_latent = vae.encode(image_tensor.to(vae.device)).latent * vae.config.scaling_factor
#                 image_latent = image_latent.cpu()[0]
                
#             # Clean up image tensors
#             del image
#             del image_tensor
#         else:
#             image_latent = None
            
#         results.append({
#             "text": prompts,
#             "prompt_embeds": ensure_cpu(prompt_embeds),
#             "prompt_attention_mask": ensure_cpu(prompt_attention_mask),
#             "image": ensure_cpu(image_latent),
#             "original_idx": idx,  # For correct merge order
#         })
        
#         # Clean up intermediate tensors
#         del prompt_embeds
#         del prompt_attention_mask
#         if image_latent is not None:
#             del image_latent
            
#     # Save this rank's part
#     part_path = latents_path + f".rank{rank}"
#     torch.save(results, part_path)
    
#     # Clean up results list
#     del results
#     gc.collect()
#     torch.cuda.empty_cache()
    
#     torch.distributed.barrier()

#     # Merge and finalize on rank 0
#     if rank == 0:
#         all_latents = []
#         for r in range(world_size):
#             # Load and immediately process each rank's results
#             rank_results = torch.load(latents_path + f".rank{r}")
#             all_latents.extend(rank_results)
#             del rank_results
#             os.remove(latents_path + f".rank{r}")
            
#             # Clean up after each rank's processing
#             gc.collect()
#             torch.cuda.empty_cache()
            
#         # Restore original dataset order
#         all_latents.sort(key=lambda x: x["original_idx"])
#         for latent in all_latents:
#             latent.pop("original_idx", None)
            
#         # Final device safety pass
#         all_latents = [ensure_cpu(item) for item in all_latents]
        
#         # Save final results
#         torch.save(all_latents, latents_path)
        
#         # Clean up
#         del all_latents
#         gc.collect()
#         torch.cuda.empty_cache()
        
#     torch.distributed.barrier()