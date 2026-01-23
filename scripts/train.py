#!/usr/bin/env python
"""
Main training script for DINO Knowledge Distillation with PyTorch Lightning and Hydra.

Usage:
    # Stage 0: Train optical baseline on BENv2
    python scripts/train.py experiment=stage0_benv2

    # Stage 1: Train SAR with KD on BENv2
    python scripts/train.py experiment=stage1_benv2

    # Override config values
    python scripts/train.py experiment=stage0_benv2 training.optimizer.lr_base=5e-5

    # Multi-GPU training
    python scripts/train.py experiment=stage0_benv2 trainer.devices=4

    # Debug mode
    python scripts/train.py experiment=stage0_benv2 debug.fast_dev_run=true
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
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from src.modules.dino_kd_module import DINOKDModule
from src.datamodules.multimodal_datamodule import MultimodalDataModule
from src.callbacks.visualization_callback import VisualizationCallback


def validate_config(cfg: DictConfig):
    """Validate configuration before training."""
    # Stage validation
    if cfg.experiment.stage == 0:
        if cfg.distillation.teacher_checkpoint is not None:
            raise ValueError("Stage 0 (optical baseline) should not have teacher_checkpoint")
        if cfg.data.data_type != "opt":
            print(f"WARNING: Stage 0 typically uses data_type='opt', but got '{cfg.data.data_type}'")

    elif cfg.experiment.stage == 1:
        if cfg.distillation.teacher_checkpoint is None:
            raise ValueError("Stage 1 (SAR distillation) requires distillation.teacher_checkpoint")
        if not os.path.exists(cfg.distillation.teacher_checkpoint):
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {cfg.distillation.teacher_checkpoint}"
            )
        if cfg.data.data_type != "sar":
            print(f"WARNING: Stage 1 typically uses data_type='sar', but got '{cfg.data.data_type}'")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main training function."""
    # Print config
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Validate config
    validate_config(cfg)

    # Set seed for reproducibility
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Create output directory
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)

    # Initialize datamodule
    datamodule = MultimodalDataModule(cfg)

    # Initialize model
    model = DINOKDModule(cfg)

    # Setup logger with resume support
    logger = None
    if cfg.logging.wandb.enabled:
        wandb_resume_id = cfg.logging.wandb.get('resume_id', None)
        logger = WandbLogger(
            project=cfg.logging.wandb.project,
            name=cfg.experiment.name,
            save_dir=cfg.experiment.output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[f"stage{cfg.experiment.stage}", cfg.data.data_type, cfg.data.dataset],
            id=wandb_resume_id,
            resume="must" if wandb_resume_id else "allow",
        )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.experiment.output_dir,
            filename="checkpoint-{epoch:02d}-{val/APM:.4f}",
            save_top_k=cfg.checkpoint.save_top_k,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_last=True,
            every_n_epochs=cfg.checkpoint.save_every_n_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
        VisualizationCallback(
            vis_interval=cfg.logging.vis_interval,
            num_samples=4,
            save_dir=os.path.join(cfg.experiment.output_dir, "vis_results"),
        ),
    ]

    # Build trainer config
    trainer_config = {
        "accelerator": cfg.trainer.accelerator,
        "devices": cfg.trainer.devices,
        "strategy": cfg.trainer.strategy,
        "precision": cfg.trainer.precision,
        "max_epochs": cfg.trainer.max_epochs,
        "accumulate_grad_batches": cfg.trainer.accumulate_grad_batches,
        "gradient_clip_val": cfg.trainer.gradient_clip_val,
        "log_every_n_steps": cfg.trainer.log_every_n_steps,
        "val_check_interval": cfg.trainer.val_check_interval,
        "enable_checkpointing": cfg.trainer.enable_checkpointing,
        "enable_progress_bar": cfg.trainer.enable_progress_bar,
        "deterministic": cfg.trainer.deterministic,
        "logger": logger,
        "callbacks": callbacks,
        "default_root_dir": cfg.experiment.output_dir,
    }

    # Handle debug options
    if cfg.debug.fast_dev_run:
        trainer_config["fast_dev_run"] = True
    if cfg.debug.limit_train_batches < 1.0:
        trainer_config["limit_train_batches"] = cfg.debug.limit_train_batches
    if cfg.debug.limit_val_batches < 1.0:
        trainer_config["limit_val_batches"] = cfg.debug.limit_val_batches
    if cfg.debug.overfit_batches > 0:
        trainer_config["overfit_batches"] = cfg.debug.overfit_batches

    # Initialize trainer
    trainer = pl.Trainer(**trainer_config)

    # Print training info
    print(f"\nStarting training:")
    print(f"  Stage: {cfg.experiment.stage}")
    print(f"  Dataset: {cfg.data.dataset}")
    print(f"  Data type: {cfg.data.data_type}")
    print(f"  Output dir: {cfg.experiment.output_dir}")
    print(f"  Max epochs: {cfg.training.num_epochs}")
    print(f"  Batch size: {cfg.data.dataloader.batch_size}")
    print(f"  Learning rate (base): {cfg.training.optimizer.lr_base}")
    if cfg.experiment.stage == 1:
        print(f"  Teacher checkpoint: {cfg.distillation.teacher_checkpoint}")

    # Check for resume checkpoint
    ckpt_path = cfg.checkpoint.get('resume_path', None)
    if ckpt_path:
        print(f"  Resuming from: {ckpt_path}")
    print()

    # Start training (with optional checkpoint resume)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
