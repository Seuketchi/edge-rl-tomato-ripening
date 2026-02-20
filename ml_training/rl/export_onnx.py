"""Export distilled student policy (PyTorch) to ONNX format.

Pipeline:
    1. Load trained StudentPolicy checkpoint (.pth)
    2. Export to ONNX (opset 13)
    3. Print size and input/output shape info

Usage:
    python -m ml_training.rl.export_onnx \\
        --checkpoint outputs/distill_*/student_policy.pth \\
        --output-dir outputs/distill_<timestamp>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from ml_training.rl.distill import StudentPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export student DQN policy to ONNX")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to student_policy.pth")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    rl_cfg = config["rl"]
    policy_cfg = rl_cfg["policy"]["student"]
    env_cfg = rl_cfg["environment"]

    # State dim (9D): ripeness, temperature, humidity, days_elapsed, target_day,
    #                  ripeness_rate, temp_deviation, days_remaining, is_near_target
    state_dim = 9
    n_actions = 4  # maintain, heat, cool, harvest

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load student policy
    print("--- Loading Student Policy ---")
    model = StudentPolicy(
        state_dim=state_dim,
        action_dim=n_actions,
        hidden_sizes=policy_cfg["hidden_sizes"],
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  FP32 size:  {total_params * 4 / 1024:.1f} KB")
    print(f"  INT8 est.:  {total_params / 1024:.1f} KB")

    # Export ONNX
    onnx_path = output_dir / "rl_policy.onnx"
    dummy_input = torch.zeros(1, state_dim)

    print(f"\n--- Exporting to ONNX ---")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=18,
        input_names=["state"],
        output_names=["action_logits"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
    )

    size_kb = onnx_path.stat().st_size / 1024
    print(f"✓ ONNX policy exported to {onnx_path}")
    print(f"  File size: {size_kb:.1f} KB")
    print(f"  Input:     [1, {state_dim}]  (normalised state vector)")
    print(f"  Output:    [1, {n_actions}]  (action logits: maintain/heat/cool/harvest)")

    print(f"\n✅ Export complete. Artifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()
