"""
ViT Knowledge Distillation Model for SAR-to-Optical Translation.

This module contains the ViTKDDistillationModel class that wraps a DINOv3 backbone
to extract intermediate features at specified layers for knowledge distillation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class ViTKDDistillationModel(nn.Module):
    """
    Wrapper for DINOv3 backbone to extract intermediate features at specified layers.

    This model is used for both teacher and student in the knowledge distillation process.
    It extracts patch tokens and CLS tokens from specified transformer layers.

    Args:
        backbone: DINOv3 backbone model (loaded via torch.hub)
        num_classes: Number of output classes for classification head
        layers: List of layer indices to extract features from
    """

    def __init__(self, backbone, num_classes: int, layers: List[int]):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim

        self.head = nn.Linear(self.embed_dim, num_classes)

        self.layers_to_extract = sorted(list(set(layers)))
        self.output_map = {layer_idx: i for i, layer_idx in enumerate(self.layers_to_extract)}

        if not self.layers_to_extract:
            raise ValueError("layers list cannot be empty.")

        self.final_cls_layer_idx = max(self.layers_to_extract)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
        """
        Forward pass extracting logits and intermediate features.

        Args:
            pixel_values: Input images of shape (B, C, H, W)

        Returns:
            logits: Classification logits of shape (B, num_classes)
            intermediate_features: Dict mapping layer indices to feature dicts containing:
                - "patch": Patch feature maps of shape (B, C, H', W')
                - "cls": CLS token of shape (B, C)
        """
        intermediate_outputs = self.backbone.get_intermediate_layers(
            pixel_values,
            n=self.layers_to_extract,
            reshape=False,
            norm=True,
            return_class_token=True
        )

        input_h, input_w = pixel_values.shape[-2:]
        patch_size = self.backbone.patch_size
        H = input_h // patch_size
        W = input_w // patch_size

        intermediate_features = {}
        for layer_idx in self.layers_to_extract:
            output_idx = self.output_map[layer_idx]
            patch_tokens, cls_token = intermediate_outputs[output_idx]

            B, N, C = patch_tokens.shape
            patch_map = patch_tokens.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

            intermediate_features[layer_idx] = {
                "patch": patch_map,
                "cls": cls_token
            }

        # Get final CLS token for classification
        _, cls_tokens_final = intermediate_outputs[-1]
        logits = self.head(cls_tokens_final)

        return logits, intermediate_features
