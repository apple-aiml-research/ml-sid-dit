#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


"""
Utility functions for SiD distillation with DiT-based diffusion or flow-matching models.

These utilities are tailored for use with DiT models from the Sana pipeline.
To support DiT models from other pipelines, some adaptation may be required.

Functions in this module assist with model loading, text encoding, and other
common tasks needed for SiD distillation workflows.
"""

import torch
from diffusers import SanaPipeline
from training.transformer_with_encoder import SanaTransformer2DModelWithEncoder
import gc

#Sana pipeline uses Gemma2B as the text encoder, and complex human instruction for text encoding
COMPLEX_HUMAN_INSTRUCTION = [
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ]

def load_dit(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype,
    num_steps: int,
    train_diffusiongan: bool = False,
    device: torch.device = None
) -> tuple:
    """Load a DiT-based model and return (vae, dit, noise_scheduler, text_encoding_pipeline)."""
    # Clear memory before loading models
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f'pretrained_model_name_or_path: {pretrained_model_name_or_path}')
    pipe = SanaPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float32)
    
    # Clear memory after loading pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    pipe.to("cuda")
    # dit is the DiT model to be used for training and sampling
    dit = pipe.transformer
    # text_encoder and tokenizer are the text encoder and tokenizer to be used for training and sampling
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    # vae is the vae to be used for training and sampling
    vae = pipe.vae
    # noise_scheduler is the noise scheduler to be used for training and sampling
    noise_scheduler = pipe.scheduler
    
    # Clear references to pipe to free memory
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    # set the timesteps for the noise scheduler
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    noise_scheduler.timesteps = noise_scheduler.timesteps.to(device=device)
    # vae, dit, and text_encoder are moved to the device and dtype
    vae = vae.requires_grad_(False).to(device, dtype=weight_dtype)
    if not train_diffusiongan:
        dit = dit.to(device, dtype=weight_dtype)
    else:
        dit = SanaTransformer2DModelWithEncoder.from_pretrained(pretrained_model_name_or_path,subfolder="transformer",revision=None,variant=None)
        dit = dit.to(device, dtype=weight_dtype)
    # Gemma is suitable under bf16, so text_encoder is set to bf16  
    text_encoder = text_encoder.requires_grad_(False).to(dtype=torch.bfloat16)
    # Initialize a text encoding pipeline and keep it to CPU for now.
    # text_encoding_pipeline is the text encoding pipeline to be used for training and sampling.
    text_encoding_pipeline = SanaPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return vae, dit, noise_scheduler, text_encoding_pipeline

def sid_dit_generate(
    dit, latents, contexts, init_timesteps, noise_scheduler, text_encoding_pipeline,
    resolution, dtype=torch.float32, return_images=False, vae=None, guidance_scale=1,
    num_steps=1, train_sampler=True, num_steps_eval=1, time_scale=1,
    uncond_embeds=None, uncond_attention_mask=None, noise=None,
    model_input=None, prompt_embeds=None, prompt_attention_mask=None,
    latent_model_input=None, t=None,
    noise_type='fresh',  # 'fixed', 'ddim'
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
    
    if isinstance(contexts, tuple):
        contexts = list(contexts)
    with torch.set_grad_enabled(train_sampler):
        if model_input is None and latent_model_input is None:
            if prompt_embeds is None:
                prompts = contexts
                with torch.no_grad():
                    prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                        prompts, complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION, do_classifier_free_guidance=False
                    )
                    if (uncond_embeds is not None) and (uncond_attention_mask is not None):
                        # do not apply complex_human_instruction for empty prompts
                        for i, p in enumerate(prompts):
                            if not p.strip():
                                prompt_embeds[i] = uncond_embeds[i]
                                prompt_attention_mask[i] = uncond_attention_mask[i]
            

            D_x = torch.zeros_like(latents).to(latents.device)
            initial_latents = latents.clone() if noise_type == 'fixed' else None
            for i in range(num_steps_eval):
                
                if noise_type == 'fresh':
                    noise = latents if i == 0 else torch.randn_like(latents).to(latents.device)
                elif noise_type=='ddim':
                    noise = latents if i == 0 else ((latents - (1.0 - t) * D_x) / t).detach()
                elif noise_type == 'fixed':
                    noise = initial_latents  # Use the initial, unmodified latents
                else:
                    raise ValueError(f"Unknown noise_type: {noise_type}")

                

                # Compute timestep t for current denoising step, normalized to [0, 1]
                with torch.no_grad():
                    scalar_t = float(init_timesteps[0]) * (1.0 - float(i) / float(num_steps))
                    t_val = scalar_t / 999.0
                    t = torch.full((latents.shape[0],), t_val, device=latents.device, dtype=latents.dtype)
                    t_flattern = t.flatten()
                    if t.numel() > 1:
                        t = t.view(-1, 1, 1, 1)

                latents = (1.0 - t) * D_x + t * noise
                latent_model_input = latents
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
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=time_scale * t_flattern,
                        return_dict=False,
                    )[0]
                    D_x = latents - (t * flow_pred if torch.numel(t) == 1 else t.view(-1, 1, 1, 1) * flow_pred)
                else:
                    return latent_model_input, t, prompt_embeds, prompt_attention_mask, latents
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
            
            flow_pred = dit(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=time_scale * t.flatten(),
                return_dict=False,
            )[0]
            D_x = latents - (t * flow_pred if torch.numel(t) == 1 else t.view(-1, 1, 1, 1) * flow_pred)
    if return_images:
        images = vae.decode(D_x / vae.config.scaling_factor, return_dict=False)[0]
        return images
    else:
        return D_x

def sid_dit_denoise(
    dit, images, noise, contexts, timesteps, noise_scheduler, text_encoding_pipeline,
    resolution, predict_x0=True, dtype=torch.float32, guidance_scale=1.0, time_scale=1.0,
    uncond_embeds=None, uncond_attention_mask=None,return_flag='decoder',
    prompt_embeds=None,prompt_attention_mask=None
):
    """
    Denoise images using the DiT-based denoiser.
    """
    
    if isinstance(contexts, tuple):
        contexts = list(contexts)
    prompts = contexts
    sigmas = timesteps
    latents = (1 - sigmas.view(-1, 1, 1, 1)) * images + sigmas.view(-1, 1, 1, 1) * noise
    #latent_model_input = latents
    latent_model_input = noise_scheduler.scale_model_input(latents, timesteps) 
    if prompt_embeds is None:
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                prompts, complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION, do_classifier_free_guidance=False
            )
            for i, p in enumerate(prompts):
                if not p.strip():
                    prompt_embeds[i] = uncond_embeds[i]
                    prompt_attention_mask[i] = uncond_attention_mask[i]
    

    if return_flag=='decoder':
        if isinstance(guidance_scale, (int, float)) and guidance_scale == 1:
            flow_pred = dit(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=time_scale * timesteps.flatten(),
                return_dict=False,
            )[0]
        elif isinstance(guidance_scale, (int, float)) and guidance_scale == 0:
            flow_pred = dit(
                hidden_states=latent_model_input,
                encoder_hidden_states=uncond_embeds,
                encoder_attention_mask=uncond_attention_mask,
                timestep=time_scale * timesteps.flatten(),
                return_dict=False,
            )[0]
        else:
            prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([uncond_attention_mask, prompt_attention_mask], dim=0)
            t = torch.cat([timesteps, timesteps])
            latent_model_input = torch.cat([latents] * 2)
            flow_pred = dit(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=time_scale * t.flatten(),
                return_dict=False,
            )[0]
            flow_pred_uncond, flow_pred_text = flow_pred.chunk(2)
            flow_pred = flow_pred_uncond + guidance_scale * (flow_pred_text - flow_pred_uncond)
        if predict_x0:
            D_x = latents - sigmas.view(-1, 1, 1, 1) * flow_pred
            return D_x
        else:
            return flow_pred
    else:
        if isinstance(guidance_scale, (int, float)) and guidance_scale == 1:
            dit_output_dict = dit(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=time_scale * timesteps.flatten(),
                return_flag=return_flag
            )
            if return_flag!='encoder':
                flow_pred = dit_output_dict.sample
            logit = dit_output_dict.encoder_output
        else:
            if return_flag=='encoder':
                dit_output_dict = dit(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=time_scale * timesteps.flatten(),
                        return_flag=return_flag
                    )
                logit = dit_output_dict.encoder_output
            else:
                prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([uncond_attention_mask, prompt_attention_mask], dim=0)
                t = torch.cat([timesteps, timesteps])
                latent_model_input = torch.cat([latents] * 2)
                dit_output_dict = dit(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=time_scale * t.flatten(),
                    return_flag=return_flag
                )
                flow_pred = dit_output_dict.sample
                flow_pred_uncond, flow_pred_text = flow_pred.chunk(2)
                flow_pred = flow_pred_uncond + guidance_scale * (flow_pred_text - flow_pred_uncond)
                logit_dict = dit_output_dict.encoder_output
                logit_uncond,logit_text = logit_dict.chunk(2)
                logit = logit_text
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