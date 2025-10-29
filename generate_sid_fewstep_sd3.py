#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from functools import partial
from training.sid_dit_sd3_util import load_dit, sid_dit_generate, sid_dit_denoise, encode_prompt    


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def read_file_to_sentences(filename):
    # Initialize an empty list to store the sentences
    sentences = []

    # Open the file
    with open(filename, 'r', encoding='utf-8') as file:
        # Read each line from the file
        for line in file:
            # Strip newline and any trailing whitespace characters
            clean_line = line.strip()
            # Add the cleaned line to the list if it is not empty
            if clean_line:
                sentences.append(clean_line)
    
    return sentences

#----------------------------------------------------------------------------

def compress_to_npz(folder_path, num=50000):
    # Get the list of all files in the folder
    npz_path = f"{folder_path}.npz"
    file_names = os.listdir(folder_path)

    # Filter the list of files to include only images
    file_names = [file_name for file_name in file_names if file_name.endswith(('.png', '.jpg', '.jpeg'))]
    num = min(num, len(file_names))
    file_names = file_names[:num]

    # Initialize a dictionary to hold image arrays and their filenames
    samples = []

    # Iterate through the files, load each image, and add it to the dictionary with a progress bar
    for file_name in tqdm.tqdm(file_names, desc=f"Compressing images to {npz_path}"):
        # Create the full path to the image file
        file_path = os.path.join(folder_path, file_name)
        
        # Read the image using PIL and convert it to a NumPy array
        image = PIL.Image.open(file_path)
        image_array = np.asarray(image).astype(np.uint8)
        
        samples.append(image_array)
    samples = np.stack(samples)

    # Save the images as a .npz file
    np.savez(npz_path, arr_0=samples)
    print(f"Images from folder {folder_path} have been saved as {npz_path}")

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
@click.option('--outdir', help='Where to save the output images', metavar='DIR', type=str, required=True)
@click.option('--seeds', help='Random seeds (e.g. 1,2,5-10)', metavar='LIST', type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs', help='Create subdirectory for every 1000 seeds', is_flag=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT', type=click.IntRange(min=1), default=16, show_default=True)
@click.option('--num', 'num_fid_samples', help='Maximum num of images', metavar='INT', type=click.IntRange(min=1), default=30000, show_default=True)
@click.option('--init_timestep', 'init_timestep', help='t_init, in [0,999]', metavar='INT', type=click.IntRange(min=0), default=999, show_default=True)
@click.option('--text_prompts', 'text_prompts', help='captions filename; the default [prompts/captions.txt] consists of 30k COCO2014 prompts', metavar='PATH|URL', type=str, default='prompts/captions.txt', show_default=True)
@click.option('--pretrained_model_name_or_path', help='DiT model path', metavar='PATH|URL', type=str, default='Efficient-Large-Model/Sana_600M_512px_diffusers', show_default=True)
@click.option('--use_fp16', help='Enable mixed-precision training', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--enable_compress_npz', help='Enable compressive npz', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--num_steps_eval', 'num_steps_eval', help='Number of generation steps', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--custom_seed', help='Enable custom seed', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--noise_type', help='Noise type for generation: fresh, fixed, or ddim', metavar='STR', type=click.Choice(['fresh', 'fixed', 'ddim']), default='fresh', show_default=True)
@click.option('--resolution', help='Image resolution', metavar='INT', type=int, default=1024, show_default=True)

def main(network_pkl, outdir, subdirs, seeds, max_batch_size, num_fid_samples, init_timestep, text_prompts, 
         pretrained_model_name_or_path, device=torch.device('cuda'), use_fp16=False, enable_compress_npz=False, 
         num_steps_eval=4, custom_seed=False, noise_type='fresh', resolution=1024):
    """Generate random images using SiD-SANA (DiT-based few-step generation).
    
    Examples:
    
    # Generate example images with default settings:
    python generate_onestep.py --outdir='output/example_images' --seeds='1,2,3,4,5' --batch=4 --network='/path/to/sid_model.pkl' --text_prompts='prompts/example_captions.txt'
    
    

    # Generate 10k images for evaluation:
    torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='output/fid_images' --seeds=0-9999 --batch=16 --network='/path/to/sid_model.pkl' --text_prompts='prompts/captions.txt' --num_steps_eval=4 --noise_type=fresh
    """
    dist.init()

    dtype = torch.float16 if use_fp16 else torch.float32
    #dtype = torch.float32

    train_diffusiongan = False
    text_encoders_dtype = dtype
    num_steps = num_steps_eval
    num_steps_eval = 4
    
   
    captions = read_file_to_sentences(text_prompts)

    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    if not custom_seed:
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    else:
        seeds_idx = parse_int_list(f'0-{len(seeds)-1}')
        all_batches = torch.as_tensor(seeds_idx).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load the SiD model
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        loaded_net = pickle.load(f)
        if 'ema' in loaded_net:
            G_ema = loaded_net['ema'].to(device).to(dtype)
        elif 'G' in loaded_net:
            G_ema = loaded_net['G'].to(device).to(dtype)
        else:
            raise KeyError("Neither 'ema' nor 'G' found in the loaded checkpoint.")

    # Load the base DiT model components
    dist.print0(f'Loading base model from "{pretrained_model_name_or_path}"...')
    
    vae, _, noise_scheduler, tokenizers, text_encoders = load_dit(
        pretrained_model_name_or_path=pretrained_model_name_or_path, 
        weight_dtype=dtype, 
        num_steps=num_steps,
        train_diffusiongan=train_diffusiongan,
        device=device,
    )
         
    text_feature_dtype = dtype
    # Create the generator function
    G=partial(
                sid_dit_generate,
                dit=G_ema,
                noise_scheduler=noise_scheduler,
                #text_encoding_pipeline=text_encoding_pipeline,
                text_encoders=text_encoders,
                tokenizers=tokenizers,
                resolution=resolution,
                dtype=dtype,
                return_images=True,
                vae=vae,
                num_steps=num_steps_eval,
                train_sampler=False,
                num_steps_eval=num_steps_eval,
                time_scale=1000,
                uncond_embeds=None,
                uncond_pooled_embeds=None,
                noise_type=noise_type,
                use_sd3_shift=False,
                text_feature_dtype=text_feature_dtype,
            )

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    if num_steps_eval > 1:
        outdir = f'{outdir}_numstep{num_steps_eval}'

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        if not custom_seed:
            rnd = StackedRandomGenerator(device, batch_seeds)
        else:
            cseed = [seeds[i] for i in batch_seeds]
            rnd = StackedRandomGenerator(device, cseed)

        # Generate latents for DiT model (32x32 for 512px resolution)
        img_channels = 16
        img_resolution = resolution // 8
        latents = rnd.randn([batch_size, img_channels, img_resolution, img_resolution], device=device,dtype=dtype)

        c = [captions[i] for i in batch_seeds]  # Index captions using list comprehension

        with torch.no_grad():
            images = G(latents=latents, contexts=c, init_timesteps=init_timestep * torch.ones((len(c),), device=latents.device, dtype=torch.long))
                
        print(images.shape, images.dtype, images.min(), images.max())

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        del images
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        del images_np

    if enable_compress_npz:
        torch.distributed.barrier()
        if dist.get_rank() == 0:
            compress_to_npz(outdir, num_fid_samples)
        torch.distributed.barrier()
    
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
