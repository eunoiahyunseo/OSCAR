"""
DINOv3 Knowledge Distillation LightningModule for SAR-to-Optical Training.

This module implements the 2-stage training pipeline:
- Stage 0: Train optical baseline (teacher)
- Stage 1: Train SAR encoder with knowledge distillation from teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from omegaconf import DictConfig
from typing import Optional, Dict, Any, Tuple
import warnings

from peft import LoraConfig, get_peft_model

from src.models.vit_kd_model import ViTKDDistillationModel
from src.models.adapters import StackedMlpAdapter
from src.losses.distillation_losses import (
    distillation_vicreg_loss,
    mld_loss_simple,
    attention_alignment_loss,
)


class DINOKDModule(pl.LightningModule):
    """
    PyTorch Lightning Module for DINOv3 Knowledge Distillation.

    This module handles:
    - Stage 0: Training optical baseline with LoRA fine-tuning
    - Stage 1: Training SAR encoder with knowledge distillation from frozen teacher

    Args:
        cfg: OmegaConf configuration object
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.stage = cfg.experiment.stage
        self.data_type = cfg.data.data_type
        self.use_teacher = (self.stage == 1)

        # Build models
        self._build_student_model()
        if self.use_teacher:
            self._build_teacher_model()
        else:
            self.teacher_model = None

        # Build adapters for feature transformation
        self._build_adapters()

        # Loss criterion (set in setup with pos_weight from datamodule)
        self.bce_criterion = None

        # Metrics
        self._setup_metrics()

        # Training state
        self.warmup_steps = cfg.training.scheduler.warmup_steps

    def _build_student_model(self):
        """Initialize student model with LoRA."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backbone = torch.hub.load(
                self.cfg.model.backbone.repo_path,
                self.cfg.model.backbone.name,
                source='local',
                weights=self.cfg.model.backbone.pretrained_weights
            )

        self.student_model = ViTKDDistillationModel(
            backbone=backbone,
            num_classes=self.cfg.model.classifier.num_classes,
            layers=self.cfg.model.classifier.layers_to_extract
        )

        lora_config = LoraConfig(
            r=self.cfg.model.lora.rank,
            lora_alpha=self.cfg.model.lora.alpha,
            target_modules=self.cfg.model.lora.target_modules,
            lora_dropout=self.cfg.model.lora.dropout,
            bias=self.cfg.model.lora.bias,
        )
        self.student_model = get_peft_model(self.student_model, lora_config)

        # Freeze all params except LoRA and head
        self._freeze_student_params()

    def _freeze_student_params(self):
        """Freeze all parameters except LoRA and classification head."""
        for name, param in self.student_model.named_parameters():
            param.requires_grad = False
            if 'lora' in name or 'base_model.model.head.' in name:
                param.requires_grad = True

    def _build_teacher_model(self):
        """Initialize and freeze teacher model for stage 1."""
        assert self.cfg.distillation.teacher_checkpoint is not None, \
            "Teacher checkpoint required for stage 1"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            teacher_backbone = torch.hub.load(
                self.cfg.model.backbone.repo_path,
                self.cfg.model.backbone.name,
                source='local',
                weights=self.cfg.model.backbone.pretrained_weights
            )

        self.teacher_model = ViTKDDistillationModel(
            backbone=teacher_backbone,
            num_classes=self.cfg.model.classifier.num_classes,
            layers=self.cfg.model.classifier.layers_to_extract
        )

        lora_config = LoraConfig(
            r=self.cfg.model.lora.rank,
            lora_alpha=self.cfg.model.lora.alpha,
            target_modules=self.cfg.model.lora.target_modules,
            lora_dropout=self.cfg.model.lora.dropout,
            bias=self.cfg.model.lora.bias,
        )
        self.teacher_model = get_peft_model(self.teacher_model, lora_config)

        # Load teacher checkpoint
        checkpoint = torch.load(self.cfg.distillation.teacher_checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.teacher_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            # PyTorch Lightning checkpoint format
            state_dict = {k.replace('student_model.', ''): v
                         for k, v in checkpoint['state_dict'].items()
                         if k.startswith('student_model.')}
            self.teacher_model.load_state_dict(state_dict, strict=False)
        else:
            self.teacher_model.load_state_dict(checkpoint, strict=False)

        # Freeze teacher
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

    def _build_adapters(self):
        """Build adapter modules for each distillation layer."""
        adapter_dim = self.cfg.model.backbone.embed_dim
        num_layers = len(self.cfg.model.classifier.layers_to_extract)

        self.adapters = nn.ModuleList([
            StackedMlpAdapter(
                dim=adapter_dim,
                num_layers=self.cfg.model.adapter.num_layers
            )
            for _ in range(num_layers)
        ])

    def _setup_metrics(self):
        """Initialize torchmetrics for validation."""
        num_classes = self.cfg.model.classifier.num_classes
        self.val_apm = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average='macro'
        )
        self.val_apmu = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average='micro'
        )
        self.val_fm1 = torchmetrics.F1Score(
            task="multilabel", num_labels=num_classes, average='macro'
        )
        self.val_fmu1 = torchmetrics.F1Score(
            task="multilabel", num_labels=num_classes, average='micro'
        )

    def setup(self, stage: str):
        """Called when fit/validate/test begins."""
        if stage == "fit":
            # Get pos_weight from datamodule if available
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'pos_weight'):
                pos_weight = self.trainer.datamodule.pos_weight
                if pos_weight is not None:
                    self.bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    self.bce_criterion = nn.BCEWithLogitsLoss()
            else:
                self.bce_criterion = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through student model."""
        return self.student_model(pixel_values)

    def _get_input_tensor(self, batch: Dict) -> torch.Tensor:
        """Get input tensor based on data type."""
        return batch["sar"] if self.data_type == "sar" else batch["opt"]

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Unpack batch
        if len(batch) == 3:  # BENv2 format: (img_dict, labels, label_names)
            img_dict, labels, _ = batch
        else:  # SEN12MS format: (img_dict, labels)
            img_dict, labels = batch

        student_input = self._get_input_tensor(img_dict)
        labels = labels.float()

        # Student forward pass
        student_logits, student_features = self.student_model(student_input)

        # Task loss (classification)
        loss_task = self.bce_criterion(student_logits, labels)

        if self.use_teacher:
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_logits, teacher_features = self.teacher_model(img_dict["opt"])

            # Compute distillation losses
            loss_dict = self._compute_distillation_losses(
                student_logits, teacher_logits,
                student_features, teacher_features,
                loss_task
            )

            total_loss = loss_dict["loss_classifier"] + loss_dict["loss_attn"] + loss_dict["loss_vicreg"]

            # Log metrics
            log_metrics = {
                "train/total_loss": total_loss,
                "train/task_loss": loss_task,
                "train/kd_loss": loss_dict["loss_kd"],
                "train/vicreg_loss": loss_dict["loss_vicreg"],
                "train/attn_loss": loss_dict["loss_attn"],
                "train/cls_loss": loss_dict["loss_cls"],
            }

            # Add per-layer debug statistics
            if "layer_stats" in loss_dict and loss_dict["layer_stats"]:
                log_metrics.update(loss_dict["layer_stats"])

            self.log_dict(log_metrics, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        else:
            total_loss = loss_task
            self.log("train/task_loss", loss_task, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)

        return total_loss

    def _compute_distillation_losses(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: Dict,
        teacher_features: Dict,
        loss_task: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all distillation losses."""
        cfg = self.cfg.distillation.loss_weights

        # Classifier knowledge distillation
        loss_kd = mld_loss_simple(
            student_logits, teacher_logits,
            T=self.cfg.distillation.temperature
        )

        loss_vicreg_total = 0.0
        loss_attn_total = 0.0
        loss_cls_total = 0.0

        # Per-layer debug statistics
        layer_stats = {}
        total_s_std = 0.0
        total_t_std = 0.0
        total_cos_sim = 0.0
        log_debug = self.cfg.logging.get('log_debug_stats', True)

        layers = self.cfg.model.classifier.layers_to_extract
        for layer_idx in layers:
            s_data = student_features[layer_idx]
            t_data = teacher_features[layer_idx]

            s_patch, s_cls = s_data["patch"], s_data["cls"]
            t_patch, t_cls = t_data["patch"], t_data["cls"]

            # CLS token alignment
            loss_cls_total += F.mse_loss(s_cls.unsqueeze(1), t_cls.unsqueeze(1))

            # VICReg loss
            l_vicreg, _, _, _ = distillation_vicreg_loss(
                s_patch, t_patch,
                lambda_inv=cfg.lambda_vicreg_inv,
                mu_var=cfg.lambda_vicreg_var,
                nu_cov=cfg.lambda_vicreg_cov
            )
            loss_vicreg_total += l_vicreg

            # Attention alignment
            loss_attn_total += attention_alignment_loss(s_patch, t_patch, s_cls, t_cls)

            # Compute per-layer debug statistics
            if log_debug:
                with torch.no_grad():
                    B, C, H, W = s_patch.shape
                    s_reshaped = s_patch.permute(0, 2, 3, 1).reshape(-1, C)
                    t_reshaped = t_patch.permute(0, 2, 3, 1).reshape(-1, C)

                    curr_s_std = s_reshaped.std(dim=0).mean().item()
                    curr_t_std = t_reshaped.std(dim=0).mean().item()
                    curr_cos_sim = F.cosine_similarity(s_reshaped, t_reshaped, dim=1).mean().item()

                    total_s_std += curr_s_std
                    total_t_std += curr_t_std
                    total_cos_sim += curr_cos_sim

                    layer_key = f"layer_{layer_idx:02d}"
                    layer_stats[f"debug/{layer_key}_s_std"] = curr_s_std
                    layer_stats[f"debug/{layer_key}_t_std"] = curr_t_std
                    layer_stats[f"debug/{layer_key}_cos_sim"] = curr_cos_sim
                    layer_stats[f"debug/{layer_key}_std_ratio"] = curr_s_std / (curr_t_std + 1e-6)

        num_layers = len(layers)
        loss_vicreg_total /= num_layers
        loss_attn_total /= num_layers
        loss_cls_total /= num_layers

        # Compute average debug statistics
        if log_debug and num_layers > 0:
            layer_stats["debug/avg_student_std"] = total_s_std / num_layers
            layer_stats["debug/avg_teacher_std"] = total_t_std / num_layers
            layer_stats["debug/avg_cosine_sim"] = total_cos_sim / num_layers
            layer_stats["debug/avg_std_ratio"] = (total_s_std / num_layers) / ((total_t_std / num_layers) + 1e-6)

        loss_classifier = loss_task + cfg.lambda_kd * loss_kd
        loss_attn = cfg.lambda_attn * (loss_attn_total + loss_cls_total)

        return {
            "loss_task": loss_task,
            "loss_kd": loss_kd,
            "loss_classifier": loss_classifier,
            "loss_vicreg": loss_vicreg_total,
            "loss_attn": loss_attn,
            "loss_cls": loss_cls_total,
            "layer_stats": layer_stats,
        }

    def validation_step(self, batch: Tuple, batch_idx: int):
        """Validation step."""
        # Unpack batch
        if len(batch) == 3:
            img_dict, labels, _ = batch
        else:
            img_dict, labels = batch

        inputs = self._get_input_tensor(img_dict)
        labels = labels.int()

        # Forward pass
        logits, _ = self.student_model(inputs)
        preds = torch.sigmoid(logits)

        # Update metrics
        self.val_apm.update(preds, labels)
        self.val_apmu.update(preds, labels)
        self.val_fm1.update(preds, labels)
        self.val_fmu1.update(preds, labels)

    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        metrics = {
            "val/APM": self.val_apm.compute(),
            "val/APu": self.val_apmu.compute(),
            "val/FM1": self.val_fm1.compute(),
            "val/Fu1": self.val_fmu1.compute(),
        }
        self.log_dict(metrics, sync_dist=True)

        # Reset metrics
        self.val_apm.reset()
        self.val_apmu.reset()
        self.val_fm1.reset()
        self.val_fmu1.reset()

    def configure_optimizers(self):
        """Configure optimizer with 3 parameter groups."""
        lora_params = []
        head_params = []

        for name, param in self.student_model.named_parameters():
            if param.requires_grad:
                if 'lora' in name:
                    lora_params.append(param)
                elif 'head' in name:
                    head_params.append(param)

        adapter_params = list(self.adapters.parameters())

        param_groups = [
            {"params": lora_params, "lr": self.cfg.training.optimizer.lr_base, "initial_lr": self.cfg.training.optimizer.lr_base},
            {"params": head_params, "lr": self.cfg.training.optimizer.lr_base, "initial_lr": self.cfg.training.optimizer.lr_base},
            {"params": adapter_params, "lr": self.cfg.training.optimizer.lr_adapter, "initial_lr": self.cfg.training.optimizer.lr_adapter},
        ]

        optimizer = Adam(param_groups, lr=self.cfg.training.optimizer.lr_base)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.num_epochs,
            eta_min=self.cfg.training.scheduler.eta_min
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Manual warmup implementation."""
        # Apply warmup
        if self.global_step < self.warmup_steps:
            lr_scale = (self.global_step + 1) / self.warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = pg.get('initial_lr', pg['lr']) * lr_scale

        # Call the actual optimizer step
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        # Log learning rate
        if self.global_step % 10 == 0:
            self.log("train/lr_base", optimizer.param_groups[0]['lr'], sync_dist=True)
            if len(optimizer.param_groups) > 2:
                self.log("train/lr_adapter", optimizer.param_groups[2]['lr'], sync_dist=True)

    def on_train_epoch_start(self):
        """Ensure teacher is in eval mode at start of each epoch."""
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Periodic memory cleanup to prevent OOM on long training runs."""
        # Clear cache every 100 batches
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
