#!/usr/bin/env python
"""
Evaluation script for ControlNet SAR-to-Optical synthesis with PyTorch Lightning.

Usage:
    # Evaluate on BENv2
    python scripts/test_controlnet.py +experiment=benv2 \
        ++checkpoint.path=./checkpoints/controlnet/benv2/final

    # Evaluate on SEN12MS
    python scripts/test_controlnet.py +experiment=sen12ms \
        ++checkpoint.path=./checkpoints/controlnet/sen12ms/final

    # Override evaluation settings
    python scripts/test_controlnet.py +experiment=benv2 \
        ++checkpoint.path=./checkpoints/controlnet/benv2/final \
        validation.num_samples=1000 \
        validation.inference_steps=50
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Subset
import torch

from src.modules.controlnet_test_module import ControlNetTestModule
from src.datamodules.multimodal_datamodule import MultimodalDataModule
from src.callbacks.evaluation_metrics_callback import EvaluationMetricsCallback


@hydra.main(version_base=None, config_path="../configs/controlnet", config_name="default")
def main(cfg: DictConfig):
    """Main evaluation function using Lightning Trainer.predict()."""
    # Print config
    print("=" * 60)
    print("Evaluation Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Validate checkpoint
    checkpoint_path = cfg.checkpoint.get('path', None)
    if checkpoint_path is None:
        raise ValueError("checkpoint.path is required for evaluation. Use ++checkpoint.path=<path>")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Set seed
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Create output directory
    output_dir = cfg.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize datamodule
    datamodule = MultimodalDataModule(cfg)
    datamodule.setup(stage="predict")

    # Handle num_samples subset
    num_samples = cfg.validation.num_samples
    if num_samples and num_samples < len(datamodule.test_dataset):
        generator = torch.Generator().manual_seed(cfg.experiment.seed)
        indices = torch.randperm(len(datamodule.test_dataset), generator=generator)[:num_samples]
        datamodule.test_dataset = Subset(datamodule.test_dataset, indices.tolist())
        print(f"Using subset of {num_samples} samples")

    # Initialize model
    print(f"\nLoading models from checkpoint: {checkpoint_path}")
    model = ControlNetTestModule(cfg, checkpoint_path)

    # Setup logger (optional)
    logger = None
    if cfg.logging.wandb.enabled:
        logger = WandbLogger(
            project=cfg.logging.wandb.project,
            name=f"{cfg.experiment.name}_eval",
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["evaluation", cfg.data.dataset],
        )

    # Setup callbacks
    callbacks = [
        EvaluationMetricsCallback(
            output_dir=output_dir,
            compute_generative_metrics=True,
            save_images=True,
        ),
    ]

    # Initialize trainer for prediction
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=1,  # Single GPU for evaluation
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # Print info
    print(f"\nStarting evaluation:")
    print(f"  Dataset: {cfg.data.dataset}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Num samples: {len(datamodule.test_dataset)}")
    print(f"  Inference steps: {cfg.validation.inference_steps}")
    print(f"  Guidance scale: {cfg.validation.guidance_scale}")
    print()

    # Run prediction (evaluation)
    trainer.predict(model, datamodule=datamodule)

    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
