#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import torch
import numpy as np
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast


from typing import List, Optional, Tuple, Union

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from training.transformer_with_encoder_flux import FluxTransformer2DModelWithEncoder
########################################################
# flux text processing utils
########################################################



def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")
    
    

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype

    with torch.autocast(device_type="cuda",dtype=dtype,enabled=dtype==torch.float32):
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")
        
    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    with torch.autocast(device_type="cuda",dtype=dtype,enabled=dtype==torch.float32):
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
    text_feature_dtype=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    #text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = torch.zeros(1, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    #text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    #text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)
    if text_feature_dtype is None:
        return prompt_embeds, pooled_prompt_embeds, text_ids
    else:
        return prompt_embeds.to(dtype=text_feature_dtype), pooled_prompt_embeds.to(dtype=text_feature_dtype), text_ids.to(dtype=text_feature_dtype)

########################################################
# flux latent utils
########################################################

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

   
def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


def prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        vae_scale_factor,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height =  2 * (int(height) // (vae_scale_factor * 2))
        width =  2 * (int(width) // (vae_scale_factor * 2))
        
        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = _pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids


########################################################
# sid flux utils
########################################################

def load_dit(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype,
    num_steps: int,
    train_diffusiongan: bool = True,
    device: torch.device = None,
    text_encoders_dtype = None
) -> tuple:
    """Load a DiT-based model and return (vae, dit, noise_scheduler, text_encoding_pipeline)."""
    print(f'pretrained_model_name_or_path: {pretrained_model_name_or_path}')
    pipeline = FluxPipeline.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=weight_dtype,
                revision=None,
                variant=None,
    )
    
    pipeline.set_progress_bar_config(disable=True)

    dit = pipeline.transformer
    tokenizer_one = pipeline.tokenizer
    tokenizer_two = pipeline.tokenizer_2
    


    text_encoder_one = pipeline.text_encoder
    text_encoder_two = pipeline.text_encoder_2
    

    noise_scheduler = pipeline.scheduler
    
    vae = pipeline.vae
    
    if not train_diffusiongan:
        dit = dit.to(device, dtype=weight_dtype)
    else:
        dit = FluxTransformer2DModelWithEncoder.from_pretrained(pretrained_model_name_or_path,subfolder="transformer",revision=None,variant=None)
        dit = dit.to(device, dtype=weight_dtype)


    dit.requires_grad_(False)
    vae.requires_grad_(False).to(device, weight_dtype)
    
    #text_encoder_one.requires_grad_(False).to(device, weight_dtype)
    #text_encoder_two.requires_grad_(False).to(device, weight_dtype)

    text_encoders = [text_encoder_one.requires_grad_(False).to(device, dtype=weight_dtype if text_encoders_dtype is None else text_encoders_dtype),
                     text_encoder_two.requires_grad_(False).to(device, dtype=weight_dtype if text_encoders_dtype is None else text_encoders_dtype)]
    
    tokenizers = [tokenizer_one, tokenizer_two]
    

    return vae, dit, noise_scheduler,  tokenizers, text_encoders  #[text_encoder_one, text_encoder_two]

import math
def time_shift( mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



    

def sid_dit_generate(
    dit, latents, contexts, init_timesteps, noise_scheduler, #text_encoding_pipeline,
    tokenizers, text_encoders,
    resolution, dtype=torch.float32, return_images=False, vae=None, guidance_scale=1,
    num_steps=1, train_sampler=True, num_steps_eval=1, time_scale=1,
    uncond_embeds=None, #uncond_attention_mask=None,    
    uncond_pooled_embeds=None,
    noise=None,
    model_input=None, prompt_embeds=None, pooled_prompt_embeds=None, #prompt_attention_mask=None,
    latent_model_input=None, t=None,
    noise_type='fresh',  # 'fixed', 'ddim'
    precondition_outputs = False,
    #use_sd3_shift = True,
    use_sd3_shift = False,
    text_feature_dtype=None,
    max_sequence_length=512, # sequence length for t5 tokenizer, deafault is 77, max is 256
    use_guidance_embeds=True,
    guidance=None,text_ids=None, latent_image_ids=None,height=None,width=None,vae_scale_factor=None
):
    """
    Generate samples using the multi-step SiD method, leveraging a DiT as the backbone model.

    This function supports different noise injection strategies for sampling:
      - 'fresh': DDPM-style, reinjecting new noise at every step.
      - 'fixed': Uses the same noise at every step (fixed noise).
      - 'ddim': DDIM-style, applies a single noise at the start and updates subsequent noise and latents deterministically.

    Args:
        dit: The DiT model used for denoising and flow prediction.
        latents: Initial latent representations to be denoised.
        contexts: List of text prompts or conditioning information.
        init_timesteps: Initial timesteps for the diffusion process.
        noise_scheduler: Scheduler controlling the noise schedule.
        text_encoding_pipeline: Pipeline for encoding text prompts.
        resolution: Image resolution for generation.
        dtype: Data type for computation (default: torch.float32).
        return_images: If True, decode and return images; otherwise, return latents.
        vae: VAE model for decoding latents to images (optional).
        guidance_scale: Classifier-free guidance scale.
        num_steps: Number of denoising steps for training.
        train_sampler: Whether to enable gradients for the sampler.
        num_steps_eval: Number of denoising steps for evaluation/sampling.
        time_scale: Scaling factor for timesteps.
        uncond_embeds: Unconditional text embeddings for classifier-free guidance.
        uncond_attention_mask: Attention mask for unconditional embeddings.
        noise: Optional precomputed noise tensor.
        model_input: Optional precomputed model input.
        prompt_embeds: Optional precomputed prompt embeddings.
        prompt_attention_mask: Optional precomputed prompt attention mask.
        latent_model_input: Optional precomputed latent model input.
        t: Optional precomputed timesteps.
        noise_type: Noise injection strategy ('fresh', 'fixed', or 'ddim').

    Returns:
        Generated samples as latents or images, depending on return_images.
    """
    

    device = latents.device 
    batch_size = len(contexts)
    
    

    # prepare latents
    #vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) 
    vae_scale_factor = 8
    num_channels_latents = 16
    height = 2 * (int(resolution) // (vae_scale_factor * 2))
    width = 2 * (int(resolution) // (vae_scale_factor * 2))
    shape = (batch_size, num_channels_latents, height, width)
    # latents = torch.randn(shape, device=device, dtype=torch.float16)
    if latent_image_ids is None:
        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                        latents.shape[0],
                        latents.shape[2] // 2,
                        latents.shape[3] // 2,
                        latents.device,
                        latents.dtype,
        )
    
    
    
        image_seq_len = height//2 * width//2
        if use_sd3_shift:
            mu = calculate_shift(
                image_seq_len,
                noise_scheduler.config.get("base_image_seq_len", 256),
                noise_scheduler.config.get("max_image_seq_len", 4096),
                noise_scheduler.config.get("base_shift", 0.5),
                noise_scheduler.config.get("max_shift", 1.15),
            )


    
    
    
    
    if isinstance(contexts, tuple):
        contexts = list(contexts)
    with torch.set_grad_enabled(train_sampler):
        #if model_input is None and latent_model_input is None:
        if model_input is None and guidance is None: #latents is None:
            if prompt_embeds is None:
                prompts = contexts
                with torch.no_grad():
                    # prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                    #     prompts, complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION, do_classifier_free_guidance=False
                    # )
                    # if (uncond_embeds is not None) and (uncond_attention_mask is not None):
                    #     # do not apply complex_human_instruction for empty prompts
                    #     for i, p in enumerate(prompts):
                    #         if not p.strip():
                    #             prompt_embeds[i] = uncond_embeds[i]
                    #             prompt_attention_mask[i] = uncond_attention_mask[i]
                    #text_input_ids_list = [tokenize_prompt(tokenizer, prompts) for tokenizer in tokenizers]
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                                text_encoders=text_encoders,
                                tokenizers=tokenizers,
                                prompt=prompts,
                                max_sequence_length=max_sequence_length,
                                text_feature_dtype=text_feature_dtype,
                    )
                    # encode prompt
                    # if (uncond_embeds is not None) and (uncond_pooled_embeds is not None):
                    #     # do not apply complex_human_instruction for empty prompts
                    #     for i, p in enumerate(prompts):
                    #         if not p.strip():
                    #             prompt_embeds[i] = uncond_embeds[i]
                    #             pooled_prompt_embeds[i] = uncond_pooled_embeds[i]
            else:
                text_ids = torch.zeros(1, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
            num_images_per_prompt = 1
            text_ids_repeated = text_ids.repeat(num_images_per_prompt, 1, 1)

            if text_ids.dim() == 3 and text_ids.shape[0] == 1:
                text_ids = text_ids.squeeze(0)

            D_x = torch.zeros_like(latents).to(latents.device)
            initial_latents = latents.clone() if noise_type == 'fixed' else None


            # main sampling loop
            

            for i in range(num_steps_eval):
                
                if noise_type == 'fresh':
                    noise = latents if i == 0 else torch.randn_like(latents).to(latents.device)
                elif noise_type=='ddim':
                    noise = latents if i == 0 else ((latents - (1.0 - t) * D_x) / t).detach()
                elif noise_type == 'fixed':
                    noise = initial_latents  # Use the initial, unmodified latents
                else:
                    raise ValueError(f"Unknown noise_type: {noise_type}")

                
                with torch.no_grad():
                    # Compute timestep t for current denoising step, normalized to [0, 1]
                    scalar_t = float(init_timesteps[0]) * (1.0 - float(i) / float(num_steps))
                    t_val = scalar_t / 999.0

                    t =  t_val 
                    

                    if use_sd3_shift:
                        t = time_shift(mu, 1.0, t)
                        #shift = 3.0
                        #t_val = shift * t_val / (1 + (shift - 1) * t_val)
                     
                    t = torch.full((latents.shape[0],), t_val, device=latents.device, dtype=latents.dtype)
                    t_flattern = t.flatten()
                    if t.numel() > 1:
                        t = t.view(-1, 1, 1, 1)

                

                latents = (1.0 - t) * D_x + t * noise
                latent_model_input = latents



                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                            latents.shape[0],
                            latents.shape[2] // 2,
                            latents.shape[3] // 2,
                            latents.device,
                            latents.dtype,
                )
                packed_latents = _pack_latents(
                            latents,
                            batch_size=latents.shape[0],
                            num_channels_latents=latents.shape[1],
                            height=latents.shape[2],
                            width=latents.shape[3],
                )

                #timesteps = t.view(-1) * 1000.0



                
                # flux uses guidance embeds by default
                if use_guidance_embeds:
                    guidance = torch.tensor([guidance_scale], device=device)
                    guidance = guidance.expand(latents.shape[0])

                # print(f"[Step {i}] Input dimensions:")
                # print(f"  packed_latents: {packed_latents.shape}")
                # print(f"  timestep: {t.view(-1).shape}")
                # print(f"  guidance: {guidance.shape}")
                # print(f"  pooled_projections: {pooled_prompt_embeds.shape}")
                # print(f"  encoder_hidden_states: {prompt_embeds.shape}")
                # print(f"  txt_ids: {text_ids.shape}")
                # print(f"  img_ids: {latent_image_ids.shape}")
                flow_pred = dit(
                    hidden_states=packed_latents,
                    timestep=t.view(-1), #timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                

                
                


                if i < num_steps_eval - 1 or (not train_sampler):
                    # if 1: #precompute_latents:
                    #     print(f"[Step {i}] latent_model_input shape: {latent_model_input.shape}")
                    #     print(f"[Step {i}] prompt_embeds shape: {prompt_embeds.shape}")
                    #     print(f"[Step {i}] prompt_attention_mask shape: {prompt_attention_mask.shape}")
                    #     print(f"[Step {i}] timestep shape: {(time_scale * t_flattern).shape}")
                    #     print(f"[Step {i}] Type of latent_model_input: {type(latent_model_input)}")
                    #     print(f"[Step {i}] Type of prompt_embeds: {type(prompt_embeds)}")
                    #     print(f"[Step {i}] Type of prompt_attention_mask: {type(prompt_attention_mask)}")
                    #     print(f"[Step {i}] Type of t_flattern: {type(t_flattern)}")
                    flow_pred = dit(
                        hidden_states=packed_latents,
                        # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                        timestep=t.view(-1), #timesteps / 1000.0,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]

                    flow_pred = _unpack_latents(
                        flow_pred,
                        height=height*vae_scale_factor,
                        width=width*vae_scale_factor,
                        vae_scale_factor=vae_scale_factor,
                    )
                    D_x = latents - t.view(-1, 1, 1, 1) * flow_pred

                    # flow_pred = dit(
                    #     hidden_states=latent_model_input,
                    #     encoder_hidden_states=prompt_embeds,
                    #     #encoder_attention_mask=prompt_attention_mask,
                    #     pooled_projections=pooled_prompt_embeds,
                    #     timestep=time_scale * t_flattern,
                    #     return_dict=False,
                    # )[0]
                    
                    #D_x = latents - (t * flow_pred if torch.numel(t) == 1 else t.view(-1, 1, 1, 1) * flow_pred)
                else:
                    #return latent_model_input, t, prompt_embeds, prompt_attention_mask, latents
                    #return latent_model_input, t, prompt_embeds, pooled_prompt_embeds, latents
                    return latents, t, guidance, prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids,height,width,vae_scale_factor
        else:
            # if 1: #precompute_latents:
            #     print(f"[ioiiii={i}] latent_model_input shape: {latent_model_input.shape}")
            #     print(f"[ioiiii={i}] prompt_embeds shape: {prompt_embeds.shape}")
            #     print(f"[ioiiii={i}] prompt_attention_mask shape: {prompt_attention_mask.shape}")
            #     print(f"[ioiiii={i}] timestep shape: {(time_scale * t.flatten()).shape}")
            #     print(f"[ioiiii={i}] Type of latent_model_input: {type(latent_model_input)}")
            #     print(f"[ioiiii={i}] Type of prompt_embeds: {type(prompt_embeds)}")
            #     print(f"[ioiiii={i}] Type of prompt_attention_mask: {type(prompt_attention_mask)}")
            #     print(f"[ioiiii={i}] Type of t_flattern: {type(t_flattern)}")
            text_ids_repeated = text_ids.repeat(1, 1, 1)
            if text_ids.dim() == 3 and text_ids.shape[0] == 1:
                text_ids = text_ids.squeeze(0)

            packed_latents = _pack_latents(
                            latents,
                            batch_size=latents.shape[0],
                            num_channels_latents=latents.shape[1],
                            height=latents.shape[2],
                            width=latents.shape[3],
                )

            flow_pred = dit(
                hidden_states=packed_latents,
                timestep=t.view(-1), #timesteps / 1000.0,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            flow_pred = _unpack_latents(
                flow_pred,
                height=height*vae_scale_factor,
                width=width*vae_scale_factor,
                vae_scale_factor=vae_scale_factor,
            )
            D_x = latents - t.view(-1, 1, 1, 1) * flow_pred

            # flow_pred = dit(
            #     hidden_states=latent_model_input,
            #     encoder_hidden_states=prompt_embeds,
            #     #encoder_attention_mask=prompt_attention_mask,
            #     pooled_projections=pooled_prompt_embeds,
            #     timestep=time_scale * t.flatten(),
            #     return_dict=False,
            # )[0]
            # D_x = latents - (t * flow_pred if torch.numel(t) == 1 else t.view(-1, 1, 1, 1) * flow_pred)
    if return_images:
        with torch.no_grad():
            images = vae.decode((D_x / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
        return images
    else:
        return D_x
    



def sid_dit_denoise(
    dit, images, noise, contexts, timesteps, noise_scheduler, #text_encoding_pipeline,
    text_encoders, tokenizers,
    resolution, predict_x0=True, dtype=torch.float32, guidance_scale=1.0, time_scale=1.0,
    uncond_embeds=None, #uncond_attention_mask=None,
    uncond_pooled_embeds=None,
    return_flag='decoder',
    prompt_embeds=None,pooled_prompt_embeds=None, #prompt_attention_mask=None
    text_feature_dtype=None,
    max_sequence_length=512, 
    use_guidance_embeds=True,
):
    """
    Denoise images using the DiT-based denoiser.
    """
    
    if isinstance(contexts, tuple):
        contexts = list(contexts)
    batch_size = len(contexts)
    prompts = contexts
    #sigmas = timesteps / 1000.0 ?
    sigmas = timesteps

        # prepare latents
    vae_scale_factor = 8
    num_channels_latents = 16
    height = 2 * (int(resolution) // (vae_scale_factor * 2))
    width = 2 * (int(resolution) // (vae_scale_factor * 2))


    latents = (1 - sigmas.view(-1, 1, 1, 1)) * images + sigmas.view(-1, 1, 1, 1) * noise

    latent_image_ids = _prepare_latent_image_ids(
                    latents.shape[0],
                    latents.shape[2] // 2,
                    latents.shape[3] // 2,
                    latents.device,
                    latents.dtype,
    )


    # todo check how guidance embeds combined with cfg
    if use_guidance_embeds:
        guidance = torch.tensor([guidance_scale], device=images.device)
        guidance = guidance.expand(latents.shape[0])

    
    latent_model_input = latents
    #latent_model_input = noise_scheduler.scale_model_input(latents, timesteps) 
    if prompt_embeds is None:
        # with torch.no_grad():
        #     prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
        #         prompts, complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION, do_classifier_free_guidance=False
        #     )
        #     for i, p in enumerate(prompts):
        #         if not p.strip():
        #             prompt_embeds[i] = uncond_embeds[i]
        #             prompt_attention_mask[i] = uncond_attention_mask[i]
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompts, max_sequence_length, text_feature_dtype=text_feature_dtype
            )
    else:
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=latents.device, dtype=dtype)
    text_ids_repeated = text_ids.repeat(1, 1, 1)

    if text_ids.dim() == 3 and text_ids.shape[0] == 1:
        text_ids = text_ids.squeeze(0)

    packed_latents = _pack_latents(
                    latents,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
    )

    if return_flag=='decoder':
        if isinstance(guidance_scale, (int, float)) and guidance_scale == 1:

            flow_pred = dit(
                hidden_states=packed_latents,
                timestep=timesteps.flatten(), #to verify
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

        elif isinstance(guidance_scale, (int, float)) and guidance_scale == 0:

            flow_pred = dit(
                hidden_states=packed_latents,
                timestep=timesteps.flatten(), #to verify
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

        else:
            prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([uncond_pooled_embeds, pooled_prompt_embeds], dim=0)
            #t = torch.cat([timesteps, timesteps])
            t = torch.cat([timesteps]*2,dim=0)
            guidance = torch.cat([guidance]*2,dim=0)
            latent_model_input = torch.cat([packed_latents] * 2)
            flow_pred = dit(
                hidden_states=latent_model_input,
                timestep=t.flatten(),
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids, #torch.cat([] * 2,dim=0),
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            flow_pred_uncond, flow_pred_text = flow_pred.chunk(2)
            flow_pred = flow_pred_uncond + guidance_scale * (flow_pred_text - flow_pred_uncond)

        flow_pred = _unpack_latents(
            flow_pred,
            height=height*vae_scale_factor,
            width=width*vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )
        if predict_x0:
            D_x = latents - sigmas.view(-1, 1, 1, 1) * flow_pred
            return D_x
        else:
            return flow_pred

    else:
        if isinstance(guidance_scale, (int, float)) and guidance_scale == 1:

            dit_output_dict = dit(
                hidden_states=packed_latents,
                timestep=timesteps.flatten(),
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids, #torch.cat([text_ids] * 2,dim=0),
                img_ids=latent_image_ids,
                return_flag=return_flag,
            )
            if return_flag!='encoder':
                flow_pred = dit_output_dict.sample
            logit = dit_output_dict.encoder_output
        else:
            if return_flag=='encoder':
                dit_output_dict = dit(
                    hidden_states=packed_latents,
                    timestep=timesteps.flatten(),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_flag=return_flag,
                )
                logit = dit_output_dict.encoder_output

            else:
            
                prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
                pooled_prompt_embeds = torch.cat([uncond_pooled_embeds, pooled_prompt_embeds], dim=0)
                #t = torch.cat([timesteps, timesteps])
                t = torch.cat([timesteps]*2,dim=0)
                guidance = torch.cat([guidance]*2,dim=0)
                latent_model_input = torch.cat([packed_latents] * 2)
                dit_output_dict = dit(
                    hidden_states=latent_model_input,
                    timestep=t.flatten(),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids, #torch.cat([text_ids] * 2),
                    img_ids=latent_image_ids,
                    return_flag=return_flag,
                )
                
                flow_pred = dit_output_dict.sample
                flow_pred_uncond, flow_pred_text = flow_pred.chunk(2)
                flow_pred = flow_pred_uncond + guidance_scale * (flow_pred_text - flow_pred_uncond)
                logit_dict = dit_output_dict.encoder_output
                logit_uncond,logit_text = logit_dict.chunk(2)
                logit = logit_text

        flow_pred = _unpack_latents(
            flow_pred,
            height=height*vae_scale_factor,
            width=width*vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )
        if predict_x0:
            if return_flag!='encoder':
                D_x = latents - sigmas.view(-1, 1, 1, 1) * flow_pred
                return D_x,logit
            else:
                return logit
        else:
            if return_flag!='encoder':
                return flow_pred,logit
            else:
                return logit
    



