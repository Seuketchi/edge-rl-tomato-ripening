#!/usr/bin/env python3
"""Export distilled student MLP weights to C header files for ESP32.

Generates:
  1. policy_weights.h  — FP32 or INT8 weight/bias arrays for each layer
  2. golden_vectors.h  — 20 (state, expected_action) test pairs

Usage:
    python ml_training/rl/export_policy_c.py
    python ml_training/rl/export_policy_c.py --student outputs/.../student_policy.pth
    python ml_training/rl/export_policy_c.py --verify   # cross-check NumPy vs PyTorch
    python ml_training/rl/export_policy_c.py --int8     # export as quantized INT8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from ml_training.rl.distill import StudentPolicy
from ml_training.rl.environment import TomatoRipeningEnv


def load_student(pth_path: Path) -> tuple[StudentPolicy, dict]:
    """Load student checkpoint and return model + metadata."""
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
    model = StudentPolicy(
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        hidden_sizes=ckpt["hidden_sizes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def extract_weights(model: StudentPolicy) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract (weight, bias) pairs from Sequential network."""
    layers = []
    for module in model.network:
        if hasattr(module, "weight"):
            w = module.weight.detach().numpy()
            b = module.bias.detach().numpy()
            layers.append((w, b))
    return layers


def quantize_mlp(layers: list[tuple[np.ndarray, np.ndarray]], states: np.ndarray):
    """Calibrate and quantize MLP to symmetric INT8 (per-channel weights)."""
    q_layers = []
    
    max_in = np.percentile(np.abs(states), 99.9)
    s_x = max_in / 127.0 if max_in > 0 else 1.0
    
    current_x = states
    
    for i, (w, b) in enumerate(layers):
        # Per-channel weight quantization
        max_w = np.max(np.abs(w), axis=1, keepdims=True)
        s_w = np.where(max_w > 0, max_w / 127.0, 1.0)
        w_q = np.clip(np.round(w / s_w), -127, 127).astype(np.int8)
        
        current_y = current_x @ w.T + b
        if i < len(layers) - 1:
            current_y = np.maximum(current_y, 0.0)
            
        max_y = np.percentile(np.abs(current_y), 99.9)
        s_y = max_y / 127.0 if max_y > 0 else 1.0
        
        s_acc = s_w.flatten() * s_x
        b_q = np.round(b / s_acc).astype(np.int32)
        
        M = s_acc / s_y
        
        # M = M_0 * 2^-shift
        M_0 = np.zeros_like(M, dtype=np.int32)
        final_shift = np.zeros_like(M, dtype=np.int32)
        
        for j in range(len(M)):
            shift = 0
            tmp_M = M[j]
            while tmp_M < 0.5 and tmp_M > 0:
                tmp_M *= 2
                shift += 1
            while tmp_M >= 1.0:
                tmp_M /= 2
                shift -= 1
            M_0[j] = int(round(tmp_M * (1 << 15)))
            final_shift[j] = shift + 15
        
        q_layers.append({
            'w_q': w_q,
            'b_q': b_q,
            'mult': M_0,
            'shift': final_shift,
            's_x': s_x,
            's_y': s_y,
        })
        
        current_x = current_y
        s_x = s_y
        
    return q_layers


def numpy_forward_fp32(state: np.ndarray, layers: list) -> int:
    """Pure NumPy MLP FP32 forward pass."""
    x = state.copy()
    for i, (w, b) in enumerate(layers):
        x = w @ x + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0.0)
    return int(np.argmax(x))


def numpy_forward_int8(state: np.ndarray, q_layers: list) -> int:
    """Pure NumPy MLP INT8 forward pass."""
    s_in = q_layers[0]['s_x']
    x_q = np.clip(np.round(state / s_in), -127, 127).astype(np.int8)
    
    for i, l in enumerate(q_layers):
        w_q = l['w_q']
        b_q = l['b_q']
        mult = l['mult']
        shift = l['shift']
        
        # Accumulate
        acc = x_q.astype(np.int32) @ w_q.astype(np.int32).T + b_q
        
        # Rescale
        scaled = np.floor((acc.astype(np.int64) * mult + (1 << (shift - 1))) / (1 << shift))
        
        if i < len(q_layers) - 1:
            x_q = np.clip(scaled, 0, 127).astype(np.int8)
        else:
            x_q = scaled
            
    return int(np.argmax(x_q))


def _fmt_array(arr: np.ndarray, name: str, dtype_str: str) -> str:
    """Format a flat array as a C initializer."""
    flat = arr.flatten()
    lines = [f"static const {dtype_str} {name}[{len(flat)}] = {{"]
    # 8 values per line
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        if dtype_str == "float":
            vals = ", ".join(f"{v: .8e}" for v in chunk)
        else:
            vals = ", ".join(f"{v}" for v in chunk)
        lines.append(f"    {vals},")
    lines.append("};")
    return "\n".join(lines)


def generate_policy_weights_h_fp32(layers: list, meta: dict, out_path: Path):
    """Write policy_weights.h with FP32 weight arrays."""
    header = [
        "/* Auto-generated by ml_training/rl/export_policy_c.py — DO NOT EDIT */",
        f"/* Student MLP (FP32): {meta['state_dim']}D input, {meta['action_dim']} actions */",
        f"/* Total params: {meta['total_params']:,} */",
        "#pragma once",
        "",
        f"#define STUDENT_STATE_DIM  {meta['state_dim']}",
        f"#define STUDENT_ACTION_DIM {meta['action_dim']}",
        f"#define STUDENT_NUM_LAYERS {len(layers)}",
        "#define STUDENT_USE_INT8   0",
        "",
    ]
    for i, (w, b) in enumerate(layers):
        header.append(f"/* Layer {i}: [{w.shape[1]}] -> [{w.shape[0]}] */")
        header.append(f"#define LAYER{i}_IN  {w.shape[1]}")
        header.append(f"#define LAYER{i}_OUT {w.shape[0]}")
        header.append(_fmt_array(w, f"w{i}", "float"))
        header.append(_fmt_array(b, f"b{i}", "float"))
        header.append("")

    out_path.write_text("\n".join(header))
    print(f"✓ {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def generate_policy_weights_h_int8(q_layers: list, meta: dict, out_path: Path):
    """Write policy_weights.h with INT8 weight arrays and scaling factors."""
    header = [
        "/* Auto-generated by ml_training/rl/export_policy_c.py — DO NOT EDIT */",
        f"/* Student MLP (INT8 Symmetric): {meta['state_dim']}D input, {meta['action_dim']} actions */",
        f"/* Total params: {meta['total_params']:,} */",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define STUDENT_STATE_DIM  {meta['state_dim']}",
        f"#define STUDENT_ACTION_DIM {meta['action_dim']}",
        f"#define STUDENT_NUM_LAYERS {len(q_layers)}",
        "#define STUDENT_USE_INT8   1",
        "",
        f"#define STUDENT_INPUT_SCALE {q_layers[0]['s_x']:.8e}f",
        "",
    ]
    for i, l in enumerate(q_layers):
        w_q, b_q = l['w_q'], l['b_q']
        header.append(f"/* Layer {i}: [{w_q.shape[1]}] -> [{w_q.shape[0]}] */")
        header.append(f"#define LAYER{i}_IN  {w_q.shape[1]}")
        header.append(f"#define LAYER{i}_OUT {w_q.shape[0]}")
        header.append(_fmt_array(w_q, f"w{i}", "int8_t"))
        header.append(_fmt_array(b_q, f"b{i}", "int32_t"))
        header.append(_fmt_array(l['mult'], f"mult{i}", "int32_t"))
        header.append(_fmt_array(l['shift'], f"shift{i}", "int32_t"))
        header.append("")

    out_path.write_text("\n".join(header))
    print(f"✓ {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def generate_golden_vectors(model: StudentPolicy, config: dict,
                            n_vectors: int = 20, out_path: Path = None):
    """Generate golden test vectors from actual environment rollouts."""
    env = TomatoRipeningEnv(config=config, seed=12345)
    all_states = []
    
    for ep in range(15):
        obs, _ = env.reset()
        done = False
        while not done:
            all_states.append(obs.copy())
            action = model.predict(obs)
            obs, _, term, trunc, _ = env.step(int(action) if not isinstance(action, int) else action)
            done = term or trunc
        if len(all_states) >= n_vectors * 5:
            break

    indices = np.linspace(0, len(all_states) - 1, n_vectors, dtype=int)
    sampled = [all_states[i] for i in indices]

    lines = [
        "/* Auto-generated by ml_training/rl/export_policy_c.py — DO NOT EDIT */",
        "#pragma once",
        "",
        f"#define NUM_GOLDEN_VECTORS {n_vectors}",
        f"#define GOLDEN_STATE_DIM   {len(sampled[0])}",
        "",
        f"static const float golden_states[{n_vectors}][{len(sampled[0])}] = {{",
    ]

    expected_actions = []
    for i, state in enumerate(sampled):
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            logits = model(x)
            action = int(logits.argmax(dim=-1).item())
        expected_actions.append(action)
        vals = ", ".join(f"{v: .6e}" for v in state)
        lines.append(f"    {{ {vals} }},  /* -> action {action} */")

    lines.append("};")
    lines.append("")
    lines.append(f"static const uint8_t golden_actions[{n_vectors}] = {{")
    lines.append("    " + ", ".join(str(a) for a in expected_actions))
    lines.append("};")

    if out_path:
        out_path.write_text("\n".join(lines))
        print(f"✓ {out_path} ({n_vectors} vectors)")

    return np.array(all_states), expected_actions, sampled


def verify(model, config, use_int8: bool, layers: list, q_layers: list = None):
    """Cross-check PyTorch, NumPy (FP32 or INT8), and expected actions."""
    print(f"\\n--- Verification ({'INT8' if use_int8 else 'FP32'}) ---")
    _, _, sampled_states = generate_golden_vectors(model, config, n_vectors=100)
    match = 0
    
    for state in sampled_states:
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            pt_action = int(model(x).argmax(dim=-1).item())
            
        if use_int8:
            np_action = numpy_forward_int8(state, q_layers)
        else:
            np_action = numpy_forward_fp32(state, layers)
            
        if np_action == pt_action:
            match += 1

    pct = match / len(sampled_states) * 100
    print(f"{'✓' if match == len(sampled_states) else '⚠'} NumPy vs PyTorch Fidelity: {match}/{len(sampled_states)} ({pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=str, default=None,
                        help="Path to student_policy.pth")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml")
    parser.add_argument("--out-dir", type=str,
                        default="edge_firmware/main",
                        help="Output directory for C headers")
    parser.add_argument("--verify", action="store_true",
                        help="Cross-check NumPy vs PyTorch")
    parser.add_argument("--int8", action="store_true",
                        help="Export symmetric INT8 quantized weights instead of FP32")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.student:
        pth_path = Path(args.student)
    else:
        candidates = sorted(Path("outputs").rglob("student_policy.pth"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("✗ No student_policy.pth found. Run distillation first.")
            return
        pth_path = candidates[0]

    print(f"Loading student from {pth_path}")
    model, meta = load_student(pth_path)
    layers = extract_weights(model)

    print(f"  Architecture: {meta['state_dim']}D → {meta['hidden_sizes']} → {meta['action_dim']}")
    print(f"  Parameters: {meta['total_params']:,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.int8:
        print("\\n--- Quantizing to INT8 ---")
        all_states, _, _ = generate_golden_vectors(model, config, n_vectors=20) 
        q_layers = quantize_mlp(layers, all_states)
        generate_policy_weights_h_int8(q_layers, meta, out_dir / "policy_weights.h")
    else:
        generate_policy_weights_h_fp32(layers, meta, out_dir / "policy_weights.h")
        q_layers = None

    generate_golden_vectors(model, config, n_vectors=20,
                            out_path=out_dir / "golden_vectors.h")

    if args.verify:
        verify(model, config, args.int8, layers, q_layers)


if __name__ == "__main__":
    main()
