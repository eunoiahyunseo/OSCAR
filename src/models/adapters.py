"""
Adapter modules for knowledge distillation.

These adapters are used to transform student features to match teacher features
during the knowledge distillation process.
"""

import torch
import torch.nn as nn


class ResidualAdapter(nn.Module):
    """
    Residual adapter with zero-initialized output for identity mapping at initialization.

    This adapter uses a residual connection where the transform block is initialized
    to output zeros, ensuring the initial output is approximately the input (identity).

    Args:
        dim: Input/output feature dimension
        hidden_dim_ratio: Ratio for hidden dimension (hidden_dim = dim * ratio)
    """

    def __init__(self, dim: int, hidden_dim_ratio: float = 0.25):
        super().__init__()
        hidden_dim = int(dim * hidden_dim_ratio)

        self.transform_block = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )

        # Zero-initialize the last layer for identity mapping at start
        nn.init.constant_(self.transform_block[-1].weight, 0)
        nn.init.constant_(self.transform_block[-1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        return x + self.transform_block(x)


class StackedMlpAdapter(nn.Module):
    """
    Stacked adapter module consisting of multiple ResidualAdapter layers.

    Args:
        dim: Input/output feature dimension
        num_layers: Number of ResidualAdapter layers to stack
    """

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()

        self.adapters = nn.ModuleList([
            ResidualAdapter(dim, hidden_dim_ratio=2.0)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        for adapter in self.adapters:
            x = adapter(x)
        return x
