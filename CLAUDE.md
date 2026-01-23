# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OSCAR (Optical-aware Semantic Control for Aleatoric Refinement in SAR-to-Optical Translation) is a research project for translating SAR (Synthetic Aperture Radar) images to optical imagery using DINOv3 and Stable Diffusion with ControlNet.

## Development Commands

### Encoder Training (2-Stage Knowledge Distillation)

**Stage 0 - Train Optical Baseline (Teacher):**
```bash
CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 dino_final.py \
    --dataset benv2 --stage 0 --data_type opt \
    --output_dir ./checkpoints/benv2/stage0_opt \
    --dinov3_repo /path/to/dinov3 \
    --dinov3_pretrained_weights ./dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
    --batch_size 72 --num_epochs 100
```

**Stage 1 - Train SAR Encoder with KD (Student):**
```bash
CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 dino_final.py \
    --dataset benv2 --stage 1 --data_type sar \
    --teacher_checkpoint ./checkpoints/benv2/stage0_opt/checkpoint_stage0_epoch100.pth \
    --output_dir ./checkpoints/benv2/stage1_sar \
    --batch_size 72 --num_epochs 100
```

### ControlNet Training

```bash
CUDA_VISIBLE_DEVICES="0,1" accelerate launch train_controlnet.py \
    --dataset benv2 \
    --pretrained_model_name_or_path ./stable-diffusion-2-1-base \
    --output_dir ./checkpoints/benv2/controlnet \
    --train_batch_size 8 --gradient_accumulation_steps 4 \
    --max_train_steps 100000 --mixed_precision bf16
```

### Evaluation

```bash
python test_controlnet.py \
    --dataset benv2 \
    --checkpoint_dir ./checkpoints/benv2/controlnet/checkpoint-100000 \
    --base_model_path ./stable-diffusion-2-1-base \
    --output_dir ./validation_results/benv2 --num_samples 1000
```

### Pre-configured Training Scripts

```bash
# BENv2 dataset
bash scripts/benv2/dino_stage_0.sh      # Stage 0 optical encoder
bash scripts/benv2/dino_stage_1.sh      # Stage 1 SAR encoder
bash scripts/benv2/train_controlnet.sh  # ControlNet training
bash scripts/benv2/test_controlnet.sh   # Evaluation

# SEN12MS dataset
bash scripts/sen12ms/dino_stage_0.sh
bash scripts/sen12ms/dino_stage1.sh
bash scripts/sen12ms/train_controlnet.sh
bash scripts/sen12ms/test_controlnet.sh
```

## Architecture

### Two-Stage Training Pipeline

1. **Optical-Aware SAR Encoder** (`dino_final.py`): DINOv3 ViT-L/16 with LoRA fine-tuning
   - Stage 0: Train on optical images to create teacher model
   - Stage 1: Train on SAR images with knowledge distillation from teacher

2. **Semantic-Grounded ControlNet** (`train_controlnet.py`): Stable Diffusion 2.1 with ControlNet
   - Uses SAR encoder features as conditioning
   - Generates optical images from SAR inputs

### Key Components

| Path | Purpose |
|------|---------|
| `dino_final.py` | Encoder training with 2-stage knowledge distillation |
| `train_controlnet.py` | ControlNet training for diffusion synthesis |
| `test_controlnet.py` | Unified evaluation for BENv2/SEN12MS |
| `models/controlnet.py` | ControlNet architecture |
| `models/unet_2d_condition.py` | UNet with image cross-attention |
| `pipelines/pipeline_seesr.py` | Inference pipeline |
| `utils/metrics.py` | Image quality metrics (PSNR, SSIM, LPIPS, FID, QNR, SAM) |
| `utils/transforms.py` | Data augmentation for SAR/Optical |
| `dataloaders/sen12ms_dataloader.py` | SEN12MS dataset loader |

## Datasets

- **BENv2** (BigEarthNet v2): LMDB format, 19-class land cover classification
- **SEN12MS**: Directory-based, 11-class multi-temporal Sentinel-1/2

Dataset paths are configured via command-line arguments:
- BENv2: `--dataset_images_lmdb`, `--dataset_metadata_parquet`
- SEN12MS: `--sen12ms_root_dir`

## Dependencies

- Python >= 3.9
- PyTorch >= 2.0 with CUDA >= 11.8
- Key packages: accelerate, diffusers, peft, transformers, configilm, torchmetrics, lpips, DISTS_pytorch, torch_fidelity

## Pre-trained Models Required

- DINOv3 ViT-L/16 (SAT-493M): `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
- Stable Diffusion 2.1 Base: from HuggingFace
