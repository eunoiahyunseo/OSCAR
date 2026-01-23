"""
Distillation loss functions for knowledge distillation training.

This module contains various loss functions used in the SAR-to-Optical
knowledge distillation process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def distillation_vicreg_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_inv: float = 25.0,
    mu_var: float = 25.0,
    nu_cov: float = 1.0,
    eps: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VICReg-style distillation loss for feature alignment.

    This loss encourages:
    - Invariance: Student features match teacher features (MSE)
    - Variance: Student maintains similar variance to teacher
    - Covariance: Student covariance matrix matches teacher

    Args:
        x: Student patch features of shape (B, C, H, W)
        y: Teacher patch features of shape (B, C, H, W)
        lambda_inv: Weight for invariance loss
        mu_var: Weight for variance loss
        nu_cov: Weight for covariance loss
        eps: Small constant for numerical stability

    Returns:
        total_loss: Weighted sum of all losses
        loss_inv: Invariance loss (MSE)
        loss_var: Variance loss
        loss_cov: Covariance loss
    """
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).reshape(-1, C)
    y = y.permute(0, 2, 3, 1).reshape(-1, C)

    loss_inv = F.mse_loss(x, y)

    with torch.cuda.amp.autocast(enabled=False):
        x = x.float()
        y = y.float()

        std_x = torch.sqrt(x.var(dim=0) + eps)
        with torch.no_grad():
            target_std_y = torch.sqrt(y.var(dim=0) + eps).detach()

        loss_var = torch.mean(F.relu(target_std_y - std_x))

        x_centered = x - x.mean(dim=0)
        y_centered = y - y.mean(dim=0)

        N = x.shape[0]

        cov_x = (x_centered.T @ x_centered) / (N - 1)
        with torch.no_grad():
            cov_y = ((y_centered.T @ y_centered) / (N - 1)).detach()

        loss_cov = F.mse_loss(cov_x, cov_y)

    total_loss = (lambda_inv * loss_inv +
                  mu_var * loss_var +
                  nu_cov * loss_cov)

    return total_loss, loss_inv, loss_var, loss_cov


def mld_loss_simple(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float = 4.0
) -> torch.Tensor:
    """
    Modified Logit Distillation loss for classifier alignment.

    Uses temperature-scaled sigmoid for soft labels from teacher.

    Args:
        student_logits: Student classification logits
        teacher_logits: Teacher classification logits
        T: Temperature for softening probabilities

    Returns:
        Binary cross-entropy loss between student and soft teacher labels
    """
    with torch.no_grad():
        teacher_probs = torch.sigmoid(teacher_logits / T)

    student_logits_scaled = student_logits / T

    loss = F.binary_cross_entropy_with_logits(
        student_logits_scaled,
        teacher_probs,
        reduction='mean'
    )

    return loss


def gram_loss_from_maps(x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    """
    Gram matrix distillation loss.

    Computes MSE between Gram matrices of student and teacher features.

    Args:
        x_s: Student patch map (B, C, Hs, Ws)
        x_t: Teacher patch map (B, C, Ht, Wt)

    Returns:
        MSE loss between Gram matrices
    """
    B, C, Hs, Ws = x_s.shape

    # L2 normalize along channel dimension
    x_s = F.normalize(x_s, dim=1)

    # Resize teacher to student spatial size if needed
    if x_t.shape[-2:] != (Hs, Ws):
        x_t = F.interpolate(x_t, size=(Hs, Ws), mode='bicubic', align_corners=False)
    x_t = F.normalize(x_t, dim=1)

    # Reshape to (B, P, C) where P = H * W
    Xs = x_s.flatten(2).transpose(1, 2).contiguous()
    Xt = x_t.flatten(2).transpose(1, 2).contiguous()

    # Compute Gram matrices G = XX^T
    Gs = torch.bmm(Xs, Xs.transpose(1, 2))
    Gt = torch.bmm(Xt, Xt.transpose(1, 2))

    return F.mse_loss(Gs, Gt)


def attention_alignment_loss(
    s_patch: torch.Tensor,
    t_patch: torch.Tensor,
    s_cls: torch.Tensor,
    t_cls: torch.Tensor,
    scale_factor: float = 20.0
) -> torch.Tensor:
    """
    Attention map alignment loss using KL divergence.

    Computes attention maps as CLS-to-patch similarity and aligns them
    using KL divergence.

    Args:
        s_patch: Student patch features (B, C, H, W)
        t_patch: Teacher patch features (B, C, H, W)
        s_cls: Student CLS token (B, C) or (B, 1, C)
        t_cls: Teacher CLS token (B, C) or (B, 1, C)
        scale_factor: Temperature scaling for attention logits

    Returns:
        KL divergence loss between attention distributions
    """
    B, C, H, W = s_patch.shape

    # Flatten patch tokens: (B, C, H, W) -> (B, N, C) where N = H * W
    s_patch_tokens = s_patch.flatten(2).transpose(1, 2)
    t_patch_tokens = t_patch.flatten(2).transpose(1, 2)

    # Ensure CLS tokens are (B, 1, C)
    if s_cls.dim() == 2:
        s_cls = s_cls.unsqueeze(1)
    if t_cls.dim() == 2:
        t_cls = t_cls.unsqueeze(1)

    # Normalize
    s_cls_norm = F.normalize(s_cls, dim=2)
    s_patch_norm = F.normalize(s_patch_tokens, dim=2)
    t_cls_norm = F.normalize(t_cls, dim=2)
    t_patch_norm = F.normalize(t_patch_tokens, dim=2)

    # Compute attention maps: (B, 1, N)
    s_attn_map = torch.bmm(s_cls_norm, s_patch_norm.transpose(1, 2))
    t_attn_map = torch.bmm(t_cls_norm, t_patch_norm.transpose(1, 2))

    # Scale and convert to probabilities
    s_log_probs = F.log_softmax(s_attn_map * scale_factor, dim=-1)
    t_probs = F.softmax(t_attn_map * scale_factor, dim=-1)

    return F.kl_div(s_log_probs, t_probs, reduction='batchmean')


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Compute Gaussian kernel for MMD."""
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = torch.cdist(x, y, p=2).pow(2)
    s = torch.matmul(beta, dist.view(1, -1))
    return torch.sum(torch.exp(-s), 0)


class MK_MMDLoss(nn.Module):
    """
    Multikernel Maximum Mean Discrepancy (MK-MMD) Loss.

    Uses multiple Gaussian kernels with different bandwidths to compute
    the MMD distance between two distributions.

    Args:
        sigmas: List of kernel bandwidths
    """

    def __init__(self, sigmas: List[float] = [0.01, 0.1, 1., 10., 100.]):
        super().__init__()
        self.register_buffer("sigmas", torch.tensor(sigmas))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute MK-MMD loss between two feature distributions.

        Args:
            x: First distribution features (B, C, H, W)
            y: Second distribution features (B, C, H, W)

        Returns:
            MK-MMD loss value
        """
        # Reshape: (B, C, H, W) -> (B*H*W, C)
        x = x.flatten(2).transpose(1, 2).reshape(-1, x.size(1))
        y = y.flatten(2).transpose(1, 2).reshape(-1, y.size(1))

        # Compute MMD
        xx = _gaussian_kernel(x, x, self.sigmas).mean()
        yy = _gaussian_kernel(y, y, self.sigmas).mean()
        xy = _gaussian_kernel(x, y, self.sigmas).mean()

        return xx + yy - 2 * xy
