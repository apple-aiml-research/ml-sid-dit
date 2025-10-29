# SiD-DiT: Score Identity Distillation for DiT-based Flow Matching Models
|![SiD-DiT](Examples/sid_dit_teaser.png "Four-Step Text-to-Image Generation")|
|:--:|
|*Four-Step Text-to-Image Generation samples from SiD-DiT*|

Welcome to the official implementation of **Score Identity Distillation (SiD)** for DiT-based diffusion and flow-matching models. This repository enables **fast, few-step text-to-image generation** via scalable, generalizable distillation techniques. The same set of hyperparameters work across **all major DiT-based Flow Matching models**, including:

- **SANA** (Rectified Flow and TrigFlow; 0.6B and 1.6B)
- **Stable Diffusion 3-Medium**
- **Stable Diffusion 3.5-Medium**
- **Stable Diffusion 3.5-Large**
- **FLUX.1-dev** (512×512 and 1024×1024)


The research paper can be found at [SiD-DiT](https://arxiv.org/abs/2509.25127).

---

## Installation

### Step 1: Environment Setup

```bash
conda env create -f sid_dit_environment.yml
conda init
source ~/.bashrc
conda activate sid_dit
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install \
  accelerate==1.8.1 blobfile==3.0.0 click==8.2.1 \
  datasets==2.19.0 diffusers==0.33.1 ftfy==6.3.1 \
  huggingface-hub==0.33.0 numpy==1.26.4 open-clip-torch==2.32.0 \
  pillow==10.3.0 requests==2.31.0 safetensors==0.5.3 \
  scipy==1.13.0 timm==0.9.16 tokenizers==0.21.1 tqdm==4.66.4 \
  transformers==4.52.4 wcwidth==0.2.13 protobuf==6.31.1 sentencepiece==0.2.0
```

---

## Dataset Setup

To train the model, we just need to prepare the training prompts as SiD is a data-free framework. By default, we use the text prompts from the [midjourney-v6-llava](https://huggingface.co/datasets) Hugging Face dataset.

We can also choose to use Aesthetic6+, Aesthetic6.25+, Aesthetic6.5+, or any other list of prompts, as long as they do not include COCO captions.

The prompts should  be saved save them in the following path:
`/data/datasets/aesthetics_6_plus/aesthetics_6_plus.txt.`

You may also change the data path, but change the path in `run_sid_dit.sh` accordingly.

To evaluate the metrics on the fly, we may prepare the validation set of MSCOCO. Please follow the setting of [SiD-LSG](https://github.com/mingyuanzhou/SiD-LSG) for the MSCOCO data preparation.

---

## Hugging Face Access

Login using your token:

```bash
huggingface-cli login --token <YOUR_HF_TOKEN>
```

Ensure your token has access to **SD3** and **SD3.5** models. This is not needed for using SANA.

---

## Launch Training

Run with (sd3-medium as an example):

```bash
sh run_sid_dit.sh sd3-medium 1_minus_sigma 8
```

- Check `run_sid_dit.sh` for all available model options and configurations.

---


## Training Configurations

Two FSDP training variants are supported:

- **AMP (Autocast + bf16) + FSDP**
- **Pure BF16 + FSDP**

Example: In `run_sid_dit_sd3.sh`, use the following flags:

**AMP + FSDP**
```bash
--fp16 0 \
--bf16 0 \
--autocast_bf16 1
```
- Uses: `lr = glr = 1e-6`, Adam `eps = 1e-8`

**Pure BF16 + FSDP**
```bash
--fp16 0 \
--bf16 1 \
--autocast_bf16 1
```
- Uses: `lr = glr = 1e-5`, Adam `eps = 1e-4`

> These hyperparameters are plug-and-play across all supported models.

All models—**except FLUX.1-dev at 1024×1024 resolution**—can be trained on a **single node with 8×80GB A100 or H100 GPUs**, with **rapid convergence** typically achieved within a few hours. Longer training can yield incremental gains, but improvements taper off after the initial convergence phase.

For **FLUX.1-dev at 1024×1024**, we **recommend using B100 GPUs**. Although it is technically possible to fit the model into eight 80GB GPUs using `cpu_offloading`, we’ve observed inconsistencies in FSDP gradient updates when `cpu_offloading` is enabled—leading to behavior that diverges from the non-offloaded baseline. While `cpu_offloading` is supported in the codebase, it **has not been fully debugged** and should be used with caution.

---

## Generate samples 
```
python generate_sid_fewstep_sd3.py --outdir=<out_dir_path> --seeds='1,2,3,4,5' \
  --batch=4 --network=<network path> --text_prompts='prompts/fig1-captions.txt' \
  --pretrained_model_name_or_path='stabilityaistable-diffusion-3-medium-diffusers'
```
Check `prompts` folder for text prompts to reproduce the results shown in the paper. 

---

## Features
**SiD-DiT** compresses large DiT-based models into **fast**, **few-step generators** with high visual fidelity and broad applicability.

Key features include:

- **Few-step generation** (default: 4 steps)
- **Flexible noise scheduling**:
  - `fresh` (default)
  - `fixed` and `ddim` — equally effective in practice, and often preferred in tasks requiring deterministic latent inputs
- **Configurable loss weighting schemes**:
  - Default: `1_minus_sigma`
  - Alternatives: `sid_default`, `1_over_sigma`, and other variants
  - Each weighting function biases the output differently (e.g., toward higher contrast or saturation). Choose based on aesthetic preference.
    - We favor `1_minus_sigma` for brighter, more "sunny" visuals.
- **Distributed training** via FSDP (Fully Sharded Data Parallel)
- **Support for AMP and BF16** training modes
- **Automatic FID & CLIP evaluation** on COCO-2014 for checkpoint selection  
  - These metrics are useful for tracking progress **within the same teacher**, but may not be reliable when comparing across different teachers or at higher resolutions.

> **Note**: SiD is **data-free by default**, requiring only **text prompts** for distillation. In this repository, the default configuration uses the [midjourney-v6-llava](https://huggingface.co/datasets) Hugging Face dataset, which provides synthetic text–image pairs. However, **only the prompts** are used under data-free settings.

For training with **adversarial losses**, the corresponding images are also utilized. Be aware that the synthetic images in midjourney-v6-llava are often of **lower quality than the outputs of SiD-distilled models** (e.g., from SD3, SD3.5, FLUX). As such, **we do not recommend enabling Diffusion GAN training (setting `--train_diffusiongan 1`) unless your provided image data is of demonstrably higher quality than your distilled model outputs**.

---

## Background & References
Please cite our work if you find it is helpful:
```bibtex
  @misc{zhou2025scoredistillationflowmatching,
      title={Score Distillation of Flow Matching Models}, 
      author={Mingyuan Zhou and Yi Gu and Huangjie Zheng and Liangchen Song and Guande He and Yizhe Zhang and Wenze Hu and Yinfei Yang},
      year={2025},
      eprint={2509.25127},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.25127}, 
}
  ```

SiD-DiT builds on the Score identity Distillation research series:

- **Few-Step Diffusion via Score Identity Distillation**  
  [arXiv:2505.12674](https://arxiv.org/abs/2505.12674)
  ```bibtex
  @misc{zhou2025fewstepdiffusionscoreidentity,
    title={Few-Step Diffusion via Score Identity Distillation},
    author={Mingyuan Zhou and Yi Gu and Zhendong Wang},
    year={2025},
    eprint={2505.12674},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
  ```

- **Guided Score Identity Distillation for Data-Free One-Step Text-to-Image Generation**  
  [arXiv:2406.01561](https://arxiv.org/abs/2406.01561)
  ```bibtex
  @inproceedings{zhou2025guided,
    title={Guided Score Identity Distillation for Data-Free One-Step Text-to-Image Generation},
    author={Zhou, Mingyuan and Wang, Zhendong and Zheng, Huangjie and Huang, Hai},
    booktitle={ICLR 2025},
    year={2025}
  }
  ```

- **Adversarial Score Identity Distillation: Rapidly Surpassing the Teacher in One Step**  
  [OpenReview](https://openreview.net/forum?id=lS2SGfWizd)
  ```bibtex
  @inproceedings{zhou2025adversarial,
    title={Adversarial Score Identity Distillation: Rapidly Surpassing the Teacher in One Step},
    author={Mingyuan Zhou and Huangjie Zheng and Yi Gu and Zhendong Wang and Hai Huang},
    booktitle={ICLR 2025}
  }
  ```

- **Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation**  
  [arXiv:2404.04057](https://arxiv.org/abs/2404.04057)
  ```bibtex
  @inproceedings{zhou2024score,
    title={Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation},
    author={Zhou, Mingyuan and Zheng, Huangjie and Wang, Zhendong and Yin, Mingzhang and Huang, Hai},
    booktitle={ICML 2024},
    year={2024}
  }
  ```

---
