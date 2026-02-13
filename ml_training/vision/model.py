"""Vision models for tomato ripeness classification.

Provides MobileNetV2 (configurable width) with a modified classifier head
for N-class tomato ripeness classification. Supports loading ImageNet
pretrained weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def create_model(config: dict) -> nn.Module:
    """Create a vision model from config.

    Args:
        config: Full config dict (containing 'vision' key).

    Returns:
        PyTorch model ready for training.
    """
    vision_cfg = config["vision"]
    model_name = vision_cfg.get("model_name", "mobilenetv2")
    num_classes = vision_cfg.get("num_classes", 4)
    pretrained = vision_cfg.get("pretrained", True)
    freeze_backbone = vision_cfg.get("freeze_backbone", False)

    if model_name == "mobilenetv2":
        model = _create_mobilenetv2(
            num_classes=num_classes,
            width_mult=vision_cfg.get("width_mult", 0.35),
            pretrained=pretrained,
        )
    elif model_name == "mobilenetv3_small":
        model = _create_mobilenetv3_small(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'mobilenetv2' or 'mobilenetv3_small'.")

    if freeze_backbone:
        _freeze_backbone(model, model_name)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Output classes:   {num_classes}")

    return model


def _create_mobilenetv2(
    num_classes: int,
    width_mult: float = 0.35,
    pretrained: bool = True,
) -> nn.Module:
    """Create MobileNetV2 with custom width and classifier.

    Note: torchvision only provides pretrained weights for width_mult=1.0.
    For other widths, we use random init for the backbone but this still
    benefits from the architecture design.
    """
    if pretrained and width_mult == 1.0:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        weights = None

    # For non-1.0 widths with pretrained=True, we load 1.0 weights and
    # create a fresh model — the architecture differs so we can't transfer
    if pretrained and width_mult != 1.0:
        print(f"  Note: Pretrained weights only available for width_mult=1.0.")
        print(f"  Using width_mult={width_mult} with random initialization.")

    model = models.mobilenet_v2(weights=weights, width_mult=width_mult)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    return model


def _create_mobilenetv3_small(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Create MobileNetV3-Small with custom classifier."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace classifier head
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )

    return model


def _freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze all parameters except the classifier head."""
    if model_name in ("mobilenetv2", "mobilenetv3_small"):
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    print("  Backbone frozen — only classifier head is trainable.")


def get_model_summary(model: nn.Module, input_size: int = 224) -> dict:
    """Get model size information for documentation.

    Args:
        model: PyTorch model.
        input_size: Input image size.

    Returns:
        Dict with param counts and estimated sizes.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate sizes
    fp32_size_mb = total_params * 4 / (1024 * 1024)
    int8_size_kb = total_params * 1 / 1024

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "fp32_size_mb": round(fp32_size_mb, 2),
        "int8_size_kb": round(int8_size_kb, 1),
        "input_shape": f"1x3x{input_size}x{input_size}",
    }
