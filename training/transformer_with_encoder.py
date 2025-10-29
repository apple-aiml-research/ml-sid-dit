#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

# Efficient GAN feature extraction from DiT: We apply spatial pooling after normalization and before the projection layer in the final transformer block.
# This approach leverages the hierarchical structure of transformers and is specifically designed for DiT architectures.
# For comparison, our earlier work (e.g., SiDA: https://arxiv.org/abs/2410.14919) used channel pooling at the UNet bottleneck for Diffusion GAN feature extraction.
# However, these methods are not directly comparable, as they are designed for different backbone architectures (DiT vs. UNet).



# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers,BaseOutput
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers import  SanaTransformer2DModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class Transformer2DModelOutputWithEncoder(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """
    encoder_output: torch.Tensor =None
    sample: torch.Tensor = None


class SanaTransformer2DModelWithEncoder(SanaTransformer2DModel):
    r"""
    A 2D Transformer model introduced in [Sana](https://huggingface.co/papers/2410.10629) family of models.

    Args:
        in_channels (`int`, defaults to `32`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `32`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `70`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `32`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of Transformer blocks to use.
        num_cross_attention_heads (`int`, *optional*, defaults to `20`):
            The number of heads to use for cross-attention.
        cross_attention_head_dim (`int`, *optional*, defaults to `112`):
            The number of channels in each head for cross-attention.
        cross_attention_dim (`int`, *optional*, defaults to `2240`):
            The number of channels in the cross-attention output.
        caption_channels (`int`, defaults to `2304`):
            The number of channels in the caption embeddings.
        mlp_ratio (`float`, defaults to `2.5`):
            The expansion ratio to use in the GLUMBConv layer.
        dropout (`float`, defaults to `0.0`):
            The dropout probability.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in the attention layer.
        sample_size (`int`, defaults to `32`):
            The base size of the input latent.
        patch_size (`int`, defaults to `1`):
            The size of the patches to use in the patch embedding layer.
        norm_elementwise_affine (`bool`, defaults to `False`):
            Whether to use elementwise affinity in the normalization layer.
        norm_eps (`float`, defaults to `1e-6`):
            The epsilon value for the normalization layer.
        qk_norm (`str`, *optional*, defaults to `None`):
            The normalization to use for the query and key.
        timestep_scale (`float`, defaults to `1.0`):
            The scale to use for the timesteps.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["SanaTransformerBlock", "PatchEmbed", "SanaModulatedNorm"]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load the pre-trained U-Net model using the parent class's from_pretrained method
        unet = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Change the class of the loaded model to CustomUNet
        unet.__class__ = cls
        return unet
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples: Optional[Tuple[torch.Tensor]] = None,
        return_dict: bool = True,
        return_flag: str = 'decoder', 
        pooling_type: str = 'spatial_pooling',
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)
        
        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        if 0: #torch.is_grad_enabled() and self.gradient_checkpointing:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        else:
                
            
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep,
                        post_patch_height,
                        post_patch_width,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep,
                        post_patch_height,
                        post_patch_width,
                    )
                    
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]
            

                        
        # 3. Normalization
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        
        
        if return_flag != "decoder": 
            #B, N, C = hidden_states.shape
            #Spatial pool
            if pooling_type == 'spatial_pooling': 
                encoder_output = hidden_states.mean(dim=1, keepdim=True) 
            elif pooling_type == 'channel_pooling':
                encoder_output = hidden_states.mean(dim=2, keepdim=True) 
            else:
                raise ValueError(f"Invalid pooling type: {pooling}")

            if return_flag == "encoder":
                if not return_dict:
                    return (encoder_output,)
                else: 
                    return Transformer2DModelOutputWithEncoder(encoder_output = encoder_output)
        
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, self.config.patch_size, self.config.patch_size, -1
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
            
        #sample=output
        if not return_dict:
            if return_flag=="encoder_decoder":
                return (output, encoder_output)
            else:
                return (output,)
        else:
            if return_flag=="encoder_decoder":
                return Transformer2DModelOutputWithEncoder(sample = output, encoder_output = encoder_output)
            else:
                return Transformer2DModelOutputWithEncoder(sample=output)


### example usage ###
def main():
    device = "cuda:0"
    weight_dtype = torch.float32
    pretrained_model_name_or_path = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    transformer = SanaTransformer2DModelWithEncoder.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        revision=None,
        variant=None,
        guidance_embeds=False,
        torch_dtype=weight_dtype,
        device_map=device,  # 
        low_cpu_mem_usage=True,
        return_dict = False,
    )
    transformer = transformer.to(device,dtype=weight_dtype)
    latents_in = torch.randn((1,32,16,16)).to(device,dtype=weight_dtype)
    prompt_embs = torch.randn((1,300,2304)).to(device,dtype=weight_dtype)
    prompt_attn_masks = torch.zeros((1,300)).to(device,dtype=weight_dtype)
    t_steps = torch.tensor([999], dtype =torch.long).flatten().to(device)
    flow_pred = transformer(
        hidden_states= latents_in,
        encoder_hidden_states=prompt_embs,
        encoder_attention_mask=prompt_attn_masks,
        timestep=t_steps, # .expand(batch_size),
        return_dict=False,
        return_flag ="encoder_decoder",
    )
    print(flow_pred[0].shape)
    print(flow_pred[1].shape)
    # decoder output dim (bsz, 32, 16,16) 512 res 1.6B
    # encoder output dim (bsz, 256 2240) 512 res 1.6B
    # 28 layers for 0.6M
    #hidden_states_shape = torch.Size([1, 256, 1152])
    # 20 layers for 0.6M
    #hidden_states_shape = torch.Size([1, 256, 2240])
if __name__ == "__main__":
    main()