"""Export trained PyTorch model to ONNX and ESP-DL format.

Pipeline:
    1. Load best PyTorch checkpoint
    2. Export to ONNX format
    3. Quantize via esp-ppq → .espdl INT8 model
    4. Save all artifacts for firmware embedding

Usage:
    python -m ml_training.vision.export_espdl --config ml_training/config.yaml --checkpoint outputs/vision_*/best_model.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from ml_training.vision.dataset import TomatoDataset, build_transforms
from ml_training.vision.model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to ONNX and ESP-DL format")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: same as checkpoint)")
    parser.add_argument("--skip-espdl", action="store_true", help="Only export ONNX, skip ESP-DL quantization")
    return parser.parse_args()


def export_onnx(
    model: torch.nn.Module,
    input_size: int,
    onnx_path: Path,
) -> None:
    """Export PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model.
        input_size: Input image dimension (e.g., 224).
        onnx_path: Output path for .onnx file.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"✓ ONNX model exported to {onnx_path}")
    print(f"  File size: {onnx_path.stat().st_size / 1024:.1f} KB")


def create_calibration_dataloader(
    config: dict,
    num_samples: int = 256,
) -> torch.utils.data.DataLoader:
    """Create a calibration dataloader for quantization.

    Uses validation transforms (no augmentation) on a subset of training data.

    Args:
        config: Full config dict.
        num_samples: Number of calibration samples.

    Returns:
        DataLoader yielding (images, labels) tuples.
    """
    vision_cfg = config["vision"]
    dataset_cfg = vision_cfg["dataset"]

    transform = build_transforms(vision_cfg, split="val")
    dataset = TomatoDataset(
        root=dataset_cfg["root"],
        class_names=dataset_cfg["class_names"],
        transform=transform,
    )

    # Subsample
    num_samples = min(num_samples, len(dataset))
    indices = np.random.RandomState(42).choice(len(dataset), num_samples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        subset, batch_size=32, shuffle=False, num_workers=2,
    )


def quantize_espdl(
    onnx_path: Path,
    espdl_path: Path,
    config: dict,
) -> None:
    """Quantize ONNX model to ESP-DL INT8 format using esp-ppq.

    Args:
        onnx_path: Path to ONNX model.
        espdl_path: Output path for .espdl model.
        config: Full config dict.
    """
    try:
        from esppq import espdl_quantize_onnx
    except ImportError:
        try:
            from ppq import espdl_quantize_onnx
        except ImportError:
            print("⚠ esp-ppq not installed. Install with: pip install esp-ppq")
            print("  Skipping ESP-DL quantization. ONNX model is still available.")
            return

    vision_cfg = config["vision"]
    export_cfg = vision_cfg.get("export", {})

    target_chip = export_cfg.get("target_chip", "esp32s3")
    quant_bits = export_cfg.get("quant_bits", 8)
    num_calibration = export_cfg.get("calibration_samples", 256)

    print(f"\nQuantizing for {target_chip} with {quant_bits}-bit precision...")

    # Create calibration dataloader
    calib_loader = create_calibration_dataloader(config, num_calibration)

    # Calibration data as list of numpy arrays
    calib_data = []
    for images, _ in calib_loader:
        for img in images:
            calib_data.append(img.numpy())

    espdl_path.parent.mkdir(parents=True, exist_ok=True)

    # Quantize
    espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_data,
        calib_steps=len(calib_data) // 32,
        input_shape=[1, 3, vision_cfg.get("input_size", 224), vision_cfg.get("input_size", 224)],
        target=target_chip,
        num_of_bits=quant_bits,
        export_test_values=True,
    )
    print(f"✓ ESP-DL model exported to {espdl_path}")
    if espdl_path.exists():
        print(f"  File size: {espdl_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    vision_cfg = config["vision"]
    export_cfg = vision_cfg.get("export", {})

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("--- Loading Model ---")
    model = create_model(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']+1} (val_acc={ckpt['val_acc']:.4f})")

    input_size = vision_cfg.get("input_size", 224)

    # Step 1: Export to ONNX
    print("\n--- Exporting to ONNX ---")
    onnx_path = output_dir / export_cfg.get("onnx_path", "tomato_classifier.onnx").split("/")[-1]
    export_onnx(model, input_size, onnx_path)

    # Step 2: Quantize to ESP-DL
    if not args.skip_espdl:
        print("\n--- Quantizing to ESP-DL ---")
        espdl_path = output_dir / export_cfg.get("espdl_path", "tomato_classifier.espdl").split("/")[-1]
        quantize_espdl(onnx_path, espdl_path, config)
    else:
        print("\nSkipping ESP-DL quantization (--skip-espdl flag)")

    print(f"\n✅ Export complete. Artifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()
