"""
Generate supplementary thesis figures.

Produces (saved to docs/thesis/images/):
  1. policy_heatmap.png        — DQN action as function of (X, days_remaining)
  2. algo_comparison.png       — DQN vs PPO vs A2C vs baselines (reward, timing, quality)
  3. reward_histogram.png      — 100-episode reward distribution for DQN
  4. royg_stages.png           — ROYG colour band with stage labels
  5. reward_decomposition.png  — Quality + timing penalty as function of harvest day

Usage:
    python generate_new_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN, PPO, A2C
from ml_training.rl.environment import TomatoRipeningEnv

OUT = ROOT / "docs" / "thesis" / "images"
OUT.mkdir(parents=True, exist_ok=True)

with open(ROOT / "ml_training" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# ── Paths ────────────────────────────────────────────────────────────
_O = ROOT / "outputs"
PATHS = {
    "dqn": _O / "rl_20260303_174001" / "best_model" / "best_model.zip",
    "ppo": _O / "algo_comparison_20260226_210137" / "ppo" / "best_model" / "best_model.zip",
    "a2c": _O / "algo_comparison_20260226_210137" / "a2c" / "best_model" / "best_model.zip",
}

# ROYG palette (X=1 green → X=0 red)
ROYG_X   = [1.0, 0.85, 0.65, 0.40, 0.15, 0.0]
ROYG_RGB = [
    (58/255,  125/255, 68/255),
    (125/255, 165/255, 58/255),
    (201/255, 168/255, 37/255),
    (232/255, 140/255, 58/255),
    (217/255,  79/255, 58/255),
    (192/255,  48/255, 42/255),
]
STAGES = ["Green", "Breaker", "Turning", "Pink", "Light Red", "Red"]

def royg_color(x: float) -> tuple[float, float, float]:
    x = float(np.clip(x, 0, 1))
    for i in range(len(ROYG_X) - 1):
        if x >= ROYG_X[i + 1]:
            t = (x - ROYG_X[i + 1]) / (ROYG_X[i] - ROYG_X[i + 1])
            a, b = np.array(ROYG_RGB[i]), np.array(ROYG_RGB[i + 1])
            return tuple((a + t * (b - a)).tolist())
    return ROYG_RGB[-1]

# ── Helper: run N episodes ────────────────────────────────────────────
def run_episodes(policy_fn, n=100, seed_offset=0, variant="B"):
    rewards, timing_errors, qualities = [], [], []
    for i in range(n):
        env = TomatoRipeningEnv(config=CONFIG, state_variant=variant, seed=42 + seed_offset + i)
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action = policy_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
        rewards.append(total)
        if "timing_error"    in info: timing_errors.append(info["timing_error"])
        if "harvest_quality" in info: qualities.append(info["harvest_quality"])
    return np.array(rewards), np.array(timing_errors), np.array(qualities)


# ════════════════════════════════════════════════════════════════════
# Figure 1 — Policy action heatmap
# ════════════════════════════════════════════════════════════════════
def fig_policy_heatmap():
    print("Generating Figure 1: policy heatmap…")
    model = DQN.load(str(PATHS["dqn"]).replace(".zip", ""))

    # Grid: X (ripeness) × days_remaining
    x_vals  = np.linspace(0.05, 1.0, 80)
    rem_vals = np.linspace(0.0, 6.0, 80)
    action_grid = np.zeros((len(x_vals), len(rem_vals)), dtype=int)

    for i, x in enumerate(x_vals):
        for j, t_rem in enumerate(rem_vals):
            t_e   = max(0.0, 5.0 - t_rem)   # assume 5d target
            dx_dt = -0.02 * (27 - 15) * x    # typical at 27°C
            x_ref = x * np.exp(-0.02 * (27 - 15) * t_e)
            # RGB from ROYG colour
            rgb = np.array(royg_color(x), dtype=np.float32)
            obs = np.array([
                x, dx_dt, x_ref,
                rgb[0], rgb[1], rgb[2],      # C_μ
                0.05, 0.05, 0.03,            # C_σ (typical)
                rgb[0], rgb[1], rgb[2],      # C_mode ≈ mean
                27.0, 65.0,                  # T, H
                t_e, t_rem,                  # elapsed, remaining
            ], dtype=np.float32)
            with torch.no_grad():
                action_grid[i, j] = int(model.predict(obs, deterministic=True)[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#999999", "#c84b38", "#3a6abf"]
    labels = ["Maintain", "Heat", "Cool"]
    cmap   = matplotlib.colors.ListedColormap(colors)

    im = ax.pcolormesh(rem_vals, x_vals, action_grid,
                       cmap=cmap, vmin=0, vmax=2, shading="auto")

    # Harvest threshold
    ax.axhline(0.15, color="white", linewidth=1.5, linestyle="--", alpha=0.9, label="Ripe threshold (X=0.15)")

    ax.set_xlabel("Days Remaining Until Target", fontsize=11)
    ax.set_ylabel("Chromatic Index X  (1=Green → 0=Red)", fontsize=11)
    ax.set_title("DQN Policy: Chosen Action by State", fontsize=12, fontweight="bold")

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
    patches.append(mpatches.Patch(color="white", label="── Ripe threshold (X=0.15)"))
    ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    path = OUT / "policy_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════
# Figure 2 — Algorithm comparison (DQN / PPO / A2C / Fixed-Day / Random)
# ════════════════════════════════════════════════════════════════════
def fig_algo_comparison():
    print("Generating Figure 2: algorithm comparison…")

    dqn_model = DQN.load(str(PATHS["dqn"]).replace(".zip", ""))
    ppo_model = PPO.load(str(PATHS["ppo"]).replace(".zip", ""))
    a2c_model = A2C.load(str(PATHS["a2c"]).replace(".zip", ""))

    rng = np.random.default_rng(0)

    policies = {
        "DQN":         lambda obs: int(dqn_model.predict(obs, deterministic=True)[0]),
        "PPO":         lambda obs: int(ppo_model.predict(obs, deterministic=True)[0]),
        "A2C":         lambda obs: int(a2c_model.predict(obs, deterministic=True)[0]),
        "Fixed-Day":   lambda obs: 0,
        "Fixed-Stage5":lambda obs: 1 if float(obs[0]) > 0.3 else 0,
        "Random":      lambda obs: int(rng.integers(0, 3)),
    }
    colors = ["#2a7a3b", "#c84b38", "#3a6abf", "#888888", "#e8883a", "#aaaaaa"]

    results = {}
    for name, fn in policies.items():
        print(f"    evaluating {name}…")
        rews, te, q = run_episodes(fn, n=100, seed_offset=5000)
        results[name] = {"reward": rews, "timing": te, "quality": q}

    names  = list(results.keys())
    means  = [results[n]["reward"].mean()  for n in names]
    stds   = [results[n]["reward"].std()   for n in names]
    timing = [results[n]["timing"].mean()  for n in names]
    qual   = [results[n]["quality"].mean() for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(names))
    w = 0.6

    def bar(ax, vals, errs, ylabel, title, fmt=".2f"):
        bars = ax.bar(x, vals, w, yerr=errs, color=colors, alpha=0.85,
                      capsize=4, ecolor="#555", error_kw={"linewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="#aaa", linewidth=0.8)
        for bar_, val in zip(bars, vals):
            ax.text(bar_.get_x() + bar_.get_width()/2, bar_.get_height() + (max(vals)-min(vals))*0.02,
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=8)

    bar(axes[0], means,  stds,             "Mean Episode Reward",     "Episode Reward",       ".1f")
    bar(axes[1], timing, [0]*len(names),   "Mean Timing Error (days)", "Harvest Timing Error", ".2f")
    bar(axes[2], qual,   [0]*len(names),   "Mean Quality Score",       "Harvest Quality",      ".3f")

    fig.suptitle("Policy Comparison: 100-Episode Evaluation", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUT / "algo_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════
# Figure 3 — DQN reward distribution histogram
# ════════════════════════════════════════════════════════════════════
def fig_reward_histogram():
    print("Generating Figure 3: reward distribution histogram…")
    model = DQN.load(str(PATHS["dqn"]).replace(".zip", ""))
    policy_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])
    rewards, timing, quality = run_episodes(policy_fn, n=100, seed_offset=7000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Reward histogram
    ax = axes[0]
    ax.hist(rewards, bins=25, color="#2a7a3b", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(rewards.mean(), color="#c84b38", linewidth=2,
               linestyle="--", label=f"Mean = {rewards.mean():.2f}")
    ax.axvline(0, color="#888", linewidth=1, linestyle=":", label="Zero reward")
    ax.set_xlabel("Total Episode Reward", fontsize=11)
    ax.set_ylabel("Episode Count", fontsize=11)
    ax.set_title("DQN Episode Reward Distribution\n(100 episodes)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Timing error histogram
    ax2 = axes[1]
    ax2.hist(timing, bins=20, color="#3a6abf", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax2.axvline(timing.mean(), color="#c84b38", linewidth=2,
                linestyle="--", label=f"Mean = {timing.mean():.2f}d")
    ax2.set_xlabel("Timing Error (days)", fontsize=11)
    ax2.set_ylabel("Episode Count", fontsize=11)
    ax2.set_title("DQN Harvest Timing Error Distribution\n(100 episodes)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUT / "reward_histogram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════
# Figure 4 — ROYG ripening stages visual
# ════════════════════════════════════════════════════════════════════
def fig_royg_stages():
    print("Generating Figure 4: ROYG stages visual…")

    fig, ax = plt.subplots(figsize=(10, 2.4))
    n = 300
    xs = np.linspace(1.0, 0.0, n)
    for i, x in enumerate(xs):
        ax.axvspan(i/n, (i+1)/n, color=royg_color(x), alpha=1.0)

    # Stage boundary lines and labels
    thresholds = list(zip(ROYG_X[:-1], ROYG_X[1:], STAGES))
    for x_hi, x_lo, label in thresholds:
        x_mid = (x_hi + x_lo) / 2
        norm  = 1.0 - x_mid          # position in [0,1] plot (left=green, right=red)
        ax.text(norm, 0.75, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="none"))
        # boundary tick
        bnd = 1.0 - x_lo
        if 0 < bnd < 1:
            ax.axvline(bnd, color="white", linewidth=1, alpha=0.5)

    # X-value axis
    x_ticks = [0, 0.15, 0.40, 0.65, 0.85, 1.0]
    ax.set_xticks([1.0 - v for v in x_ticks])
    ax.set_xticklabels([f"X={v}" for v in x_ticks], fontsize=8.5)
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Chromatic Index X = G / (R + G)    ←  ripe    unripe  →", fontsize=10)
    ax.set_title("Tomato Ripening Stages (ROYG Chromatic Index Scale)", fontsize=11, fontweight="bold")

    # Arrow annotations
    ax.annotate("Harvest\nzone", xy=(1.0-0.08, 0.25), fontsize=8, ha="center",
                color="white", style="italic")

    plt.tight_layout()
    path = OUT / "royg_stages.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════
# Figure 5 — Reward function decomposition
# ════════════════════════════════════════════════════════════════════
def fig_reward_decomposition():
    print("Generating Figure 5: reward decomposition…")

    target_day = 5.0
    harvest_days = np.linspace(1.0, 8.0, 200)

    # Quality score: based on ripeness at harvest
    # X at harvest day d: X(d) = X0 * exp(-k1*(T-Tbase)*d)
    # Use typical X0=0.9, k1=0.02, T=27, Tbase=15
    k1, T, Tbase, X0 = 0.02, 27.0, 15.0, 0.9
    X_at_harvest = X0 * np.exp(-k1 * (T - Tbase) * harvest_days)

    # Quality = 1 - X (riper = better), clipped
    quality = np.clip(1.0 - X_at_harvest, 0, 1)

    # Timing penalty = |harvest_day - target_day|²  (approx)
    timing_error = np.abs(harvest_days - target_day)
    timing_penalty = -(timing_error ** 2) * 2.0   # scale to match actual rewards

    # Quality reward component (positive when X < 0.15)
    quality_reward = np.where(X_at_harvest < 0.15, quality * 5.0, -quality * 3.0)

    # Total approximate reward
    total = quality_reward + timing_penalty

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(harvest_days, quality_reward,  color="#2a7a3b", linewidth=2, label="Quality component")
    ax.plot(harvest_days, timing_penalty,  color="#c84b38", linewidth=2, linestyle="--", label="Timing penalty")
    ax.plot(harvest_days, total,           color="#1a1a2e", linewidth=2.5, label="Total reward (approx.)")
    ax.axvline(target_day, color="#c8a838", linewidth=1.5, linestyle=":", label=f"Target day ({target_day}d)")
    ax.axvline(harvest_days[np.argmax(total)], color="#3a6abf", linewidth=1.5,
               linestyle=":", label=f"Optimal harvest ≈ {harvest_days[np.argmax(total)]:.1f}d")
    ax.axhline(0, color="#aaa", linewidth=0.8)

    # Mark ripe threshold crossing
    ripe_day = harvest_days[np.where(X_at_harvest < 0.15)[0][0]] if np.any(X_at_harvest < 0.15) else None
    if ripe_day:
        ax.axvline(ripe_day, color="#e8883a", linewidth=1, linestyle="-.",
                   label=f"X reaches 0.15 at ≈{ripe_day:.1f}d")

    ax.set_xlabel("Harvest Day", fontsize=11)
    ax.set_ylabel("Reward Component Value", fontsize=11)
    ax.set_title("Reward Function Decomposition\n(target=5d, X₀=0.9, k₁=0.02, T=27°C)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(1, 8)

    plt.tight_layout()
    path = OUT / "reward_decomposition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Saving all figures to {OUT}\n")
    fig_royg_stages()         # fast, no model
    fig_reward_decomposition() # fast, no model
    fig_policy_heatmap()       # DQN model only
    fig_reward_histogram()     # 100 DQN episodes
    fig_algo_comparison()      # 600 episodes total (slowest)
    print(f"\nDone. All figures saved to {OUT}/")
