"""
Visualization callback for attention maps and PCA features.

This callback generates and logs visualizations during training to
monitor the knowledge distillation process.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import io
import os
from typing import Optional, Dict, List


def normalize_for_display(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor to (H, W, C) numpy array normalized to 0-1."""
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    img_normalized = np.zeros_like(img_np, dtype=float)
    for i in range(img_np.shape[2]):
        band = img_np[..., i]
        band_min, band_max = band.min(), band.max()
        if band_max > band_min:
            img_normalized[..., i] = (band - band_min) / (band_max - band_min)
        else:
            img_normalized[..., i] = band

    return np.clip(img_normalized, 0, 1)


class VisualizationCallback(Callback):
    """
    Callback for generating attention map and PCA visualizations.

    Args:
        vis_interval: Number of steps between visualizations
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualization images
    """

    def __init__(
        self,
        vis_interval: int = 1000,
        num_samples: int = 4,
        save_dir: str = "./vis_results"
    ):
        super().__init__()
        self.vis_interval = vis_interval
        self.num_samples = num_samples
        self.save_dir = save_dir

        self.fixed_val_batch = None
        self.mean = torch.tensor((0.430, 0.411, 0.296)).view(3, 1, 1)
        self.std = torch.tensor((0.213, 0.156, 0.143)).view(3, 1, 1)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Fetch fixed validation batch for consistent visualization."""
        if trainer.global_rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)

            val_dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))

            if len(batch) == 3:
                img_dict, labels, _ = batch
            else:
                img_dict, labels = batch

            self.fixed_val_batch = {
                "sar": img_dict["sar"][:self.num_samples].to(pl_module.device),
                "opt": img_dict["opt"][:self.num_samples].to(pl_module.device),
            }

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int
    ):
        """Generate visualizations at specified intervals."""
        if trainer.global_step % self.vis_interval != 0:
            return

        if trainer.global_rank != 0:
            return

        if self.fixed_val_batch is None:
            return

        try:
            # Validation batch visualization
            self._generate_visualizations(trainer, pl_module)

            # Training batch visualization
            self._generate_train_visualizations(trainer, pl_module, batch)
        except Exception as e:
            print(f"Visualization failed: {e}")

    def _generate_train_visualizations(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch
    ):
        """Generate and log visualizations for the current training batch."""
        pl_module.eval()

        with torch.no_grad():
            # Unpack batch
            if len(batch) == 3:
                img_dict, labels, _ = batch
            else:
                img_dict, labels = batch

            train_sar = img_dict["sar"][:self.num_samples].to(pl_module.device)
            train_opt = img_dict["opt"][:self.num_samples].to(pl_module.device)

            # Get input based on data type
            train_input = train_sar if pl_module.data_type == "sar" else train_opt

            # Student forward pass
            _, s_features = pl_module.student_model(train_input)

            # Teacher forward pass (if available)
            if pl_module.teacher_model is not None:
                _, t_features = pl_module.teacher_model(train_opt)
            else:
                t_features = None

            # Get last layer features
            layers = pl_module.cfg.model.classifier.layers_to_extract
            last_layer = layers[-1]

            s_data = s_features[last_layer]
            t_data = t_features[last_layer] if t_features else None

            # Generate attention visualization
            attn_img = self._visualize_attention(
                train_input, train_opt,
                s_data, t_data,
                trainer.current_epoch, trainer.global_step,
                pl_module.use_teacher,
                pl_module.data_type
            )

            # Generate PCA visualization
            pca_img = self._visualize_pca(
                train_input, train_opt,
                s_data["patch"], t_data["patch"] if t_data else None,
                trainer.current_epoch, trainer.global_step,
                pl_module.use_teacher,
                pl_module.data_type
            )

            # Log to wandb if available
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                import wandb
                trainer.logger.experiment.log({
                    "train/relation_map_batch": wandb.Image(attn_img),
                    "train/patch_pca_batch": wandb.Image(pca_img),
                }, step=trainer.global_step)

        pl_module.train()

    def _generate_visualizations(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """Generate and log attention and PCA visualizations."""
        pl_module.eval()

        with torch.no_grad():
            vis_sar = self.fixed_val_batch["sar"]
            vis_opt = self.fixed_val_batch["opt"]

            # Get input based on data type
            vis_input = vis_sar if pl_module.data_type == "sar" else vis_opt

            # Student forward pass
            _, s_features = pl_module.student_model(vis_input)

            # Teacher forward pass (if available)
            if pl_module.teacher_model is not None:
                _, t_features = pl_module.teacher_model(vis_opt)
            else:
                t_features = None

            # Get last layer features
            layers = pl_module.cfg.model.classifier.layers_to_extract
            last_layer = layers[-1]

            s_data = s_features[last_layer]
            t_data = t_features[last_layer] if t_features else None

            # Generate attention visualization
            attn_img = self._visualize_attention(
                vis_input, vis_opt,
                s_data, t_data,
                trainer.current_epoch, trainer.global_step,
                pl_module.use_teacher,
                pl_module.data_type
            )

            # Generate PCA visualization
            pca_img = self._visualize_pca(
                vis_input, vis_opt,
                s_data["patch"], t_data["patch"] if t_data else None,
                trainer.current_epoch, trainer.global_step,
                pl_module.use_teacher,
                pl_module.data_type
            )

            # Log to wandb if available
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                import wandb
                trainer.logger.experiment.log({
                    "inference/attention_map": wandb.Image(attn_img),
                    "inference/pca_features": wandb.Image(pca_img),
                }, step=trainer.global_step)

        pl_module.train()

    def _visualize_attention(
        self,
        student_tensor: torch.Tensor,
        teacher_tensor: torch.Tensor,
        s_data: Dict,
        t_data: Optional[Dict],
        epoch: int,
        step: int,
        use_teacher: bool,
        data_type: str
    ) -> Image.Image:
        """Generate attention map visualization."""
        B = min(student_tensor.shape[0], self.num_samples)

        s_cls, s_patch = s_data["cls"][:B], s_data["patch"][:B]

        if use_teacher and t_data is not None:
            t_cls, t_patch = t_data["cls"][:B], t_data["patch"][:B]

        cols = 4 if use_teacher else 2
        fig, axs = plt.subplots(B, cols, figsize=(5 * cols, 4 * B))
        fig.suptitle(f"Attention Maps (Epoch {epoch} Step {step})", fontsize=16)

        if B == 1:
            axs = axs[np.newaxis, :]

        mean = self.mean.to(student_tensor.device)
        std = self.std.to(student_tensor.device)

        for idx in range(B):
            # Compute student attention map
            curr_s_cls = s_cls[idx:idx+1]
            curr_s_patch = s_patch[idx:idx+1]
            flat_s_patch = curr_s_patch.flatten(2).transpose(1, 2)

            attn_map_s = torch.bmm(
                F.normalize(curr_s_cls.unsqueeze(1), dim=2),
                F.normalize(flat_s_patch, dim=2).transpose(1, 2)
            ).view(1, curr_s_patch.shape[2], curr_s_patch.shape[3])

            if use_teacher and t_data is not None:
                curr_t_cls = t_cls[idx:idx+1]
                curr_t_patch = t_patch[idx:idx+1]
                flat_t_patch = curr_t_patch.flatten(2).transpose(1, 2)
                attn_map_t = torch.bmm(
                    F.normalize(curr_t_cls.unsqueeze(1), dim=2),
                    F.normalize(flat_t_patch, dim=2).transpose(1, 2)
                ).view(1, curr_t_patch.shape[2], curr_t_patch.shape[3])

            img_h, img_w = student_tensor.shape[-2:]

            def min_max_norm(arr):
                return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

            s_map_resized = F.interpolate(
                attn_map_s.unsqueeze(0), size=(img_h, img_w), mode='bicubic'
            ).squeeze().detach().cpu().numpy()
            s_map_vis = min_max_norm(s_map_resized)

            if use_teacher and t_data is not None:
                t_map_resized = F.interpolate(
                    attn_map_t.unsqueeze(0), size=(img_h, img_w), mode='bicubic'
                ).squeeze().detach().cpu().numpy()
                t_map_vis = min_max_norm(t_map_resized)

            # Prepare display images
            if data_type == "sar":
                s_img = student_tensor[idx].cpu().float() * std.cpu() + mean.cpu()
                s_disp = s_img[0:1] if s_img.shape[0] >= 2 else s_img
                cmap_student = 'gray'
            else:
                opt_raw = student_tensor[idx].cpu() * std.cpu() + mean.cpu()
                s_disp = torch.clamp(opt_raw, 0, 1)
                cmap_student = None

            student_disp = normalize_for_display(s_disp)

            teacher_disp = None
            if use_teacher:
                t_raw = teacher_tensor[idx].cpu() * std.cpu() + mean.cpu()
                teacher_disp = torch.clamp(t_raw, 0, 1)
                teacher_disp = normalize_for_display(teacher_disp)

            # Plot
            axs[idx, 0].imshow(student_disp, cmap=cmap_student)
            if idx == 0:
                axs[idx, 0].set_title(f"Student Input ({data_type.upper()})")
            axs[idx, 0].axis('off')

            axs[idx, 1].imshow(student_disp, cmap=cmap_student)
            axs[idx, 1].imshow(s_map_vis, cmap='jet', alpha=0.5)
            if idx == 0:
                axs[idx, 1].set_title("Student Attention")
            axs[idx, 1].axis('off')

            if use_teacher and teacher_disp is not None:
                axs[idx, 2].imshow(teacher_disp)
                if idx == 0:
                    axs[idx, 2].set_title("Teacher Input (Opt)")
                axs[idx, 2].axis('off')

                axs[idx, 3].imshow(teacher_disp)
                axs[idx, 3].imshow(t_map_vis, cmap='jet', alpha=0.5)
                if idx == 0:
                    axs[idx, 3].set_title("Teacher Attention")
                axs[idx, 3].axis('off')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()

        return result_image

    def _visualize_pca(
        self,
        student_tensor: torch.Tensor,
        teacher_tensor: torch.Tensor,
        s_patch: torch.Tensor,
        t_patch: Optional[torch.Tensor],
        epoch: int,
        step: int,
        use_teacher: bool,
        data_type: str
    ) -> Image.Image:
        """Generate PCA visualization of patch features."""
        B = min(student_tensor.shape[0], self.num_samples)

        cols = 4 if use_teacher else 2
        fig, axs = plt.subplots(B, cols, figsize=(5 * cols, 4 * B))
        fig.suptitle(f"PCA Features (Epoch {epoch} Step {step})", fontsize=16)

        if B == 1:
            axs = axs[np.newaxis, :]

        mean = self.mean.to(student_tensor.device)
        std = self.std.to(student_tensor.device)

        for idx in range(B):
            s_feat = s_patch[idx].detach().cpu()
            C, H, W = s_feat.shape
            N = H * W
            s_flat = s_feat.view(C, -1).permute(1, 0).numpy()
            s_mean = s_flat.mean(axis=0, keepdims=True)
            s_centered = s_flat - s_mean

            if use_teacher and t_patch is not None:
                t_feat = t_patch[idx].detach().cpu()
                t_flat = t_feat.view(C, -1).permute(1, 0).numpy()
                t_mean = t_flat.mean(axis=0, keepdims=True)
                t_centered = t_flat - t_mean
                combined = np.concatenate([s_centered, t_centered], axis=0)
            else:
                combined = s_centered

            pca = PCA(n_components=3)
            pca.fit(combined)
            pca_feats = pca.transform(combined)

            pca_min = pca_feats.min(axis=0)
            pca_max = pca_feats.max(axis=0)
            denom = pca_max - pca_min
            denom[denom == 0] = 1.0
            pca_feats = (pca_feats - pca_min) / denom

            s_pca = pca_feats[:N]
            s_pca_img = torch.from_numpy(s_pca).view(H, W, 3).permute(2, 0, 1).unsqueeze(0).float()
            target_h, target_w = student_tensor.shape[-2:]
            s_pca_big = F.interpolate(s_pca_img, size=(target_h, target_w), mode='nearest').squeeze(0)
            s_pca_vis = s_pca_big.permute(1, 2, 0).numpy()

            t_pca_vis = None
            if use_teacher and t_patch is not None:
                t_pca = pca_feats[N:]
                t_pca_img = torch.from_numpy(t_pca).view(H, W, 3).permute(2, 0, 1).unsqueeze(0).float()
                t_pca_big = F.interpolate(t_pca_img, size=(target_h, target_w), mode='nearest').squeeze(0)
                t_pca_vis = t_pca_big.permute(1, 2, 0).numpy()

            # Prepare display images
            if data_type == "sar":
                s_img = student_tensor[idx].cpu().float() * std.cpu() + mean.cpu()
                s_disp = s_img[0:1] if s_img.shape[0] >= 2 else s_img
                cmap_student = 'gray'
            else:
                opt_raw = student_tensor[idx].cpu() * std.cpu() + mean.cpu()
                s_disp = torch.clamp(opt_raw, 0, 1)
                cmap_student = None

            student_disp = normalize_for_display(s_disp)

            teacher_disp = None
            if use_teacher:
                t_raw = teacher_tensor[idx].cpu() * std.cpu() + mean.cpu()
                teacher_disp = torch.clamp(t_raw, 0, 1)
                teacher_disp = normalize_for_display(teacher_disp)

            # Plot
            axs[idx, 0].imshow(student_disp, cmap=cmap_student)
            if idx == 0:
                axs[idx, 0].set_title(f"Student Input ({data_type.upper()})")
            axs[idx, 0].axis('off')

            axs[idx, 1].imshow(s_pca_vis)
            if idx == 0:
                axs[idx, 1].set_title("Student PCA")
            axs[idx, 1].axis('off')

            if use_teacher and teacher_disp is not None:
                axs[idx, 2].imshow(teacher_disp)
                if idx == 0:
                    axs[idx, 2].set_title("Teacher Input")
                axs[idx, 2].axis('off')

                axs[idx, 3].imshow(t_pca_vis)
                if idx == 0:
                    axs[idx, 3].set_title("Teacher PCA")
                axs[idx, 3].axis('off')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()

        return result_image
