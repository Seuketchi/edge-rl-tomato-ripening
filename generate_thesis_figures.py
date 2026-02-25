#!/usr/bin/env python3
"""Generate publication-quality thesis figures for IEEE two-column format.

Usage:
    python generate_thesis_figures.py
    python generate_thesis_figures.py --model-dir outputs/rl_20260217_095300
    python generate_thesis_figures.py --figures episode envelope tracking comparison distillation training

Outputs all figures to docs/thesis/images/ as 300 DPI PNGs sized for IEEE
single-column width (3.5 in ≈ 88 mm).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml
from matplotlib.patches import Patch
from stable_baselines3 import DQN

from ml_training.rl.environment import TomatoRipeningEnv

# ─────────────────────────────────────────────
# IEEE Style Configuration
# ─────────────────────────────────────────────
IEEE_COL_WIDTH = 3.5          # inches  (single column)
IEEE_TEXT_WIDTH = 7.16        # inches  (full page width, for figure* )
DPI = 300
FONT_SIZE = 8                 # IEEE standard body text ≈ 8–9 pt
FONT_FAMILY = "serif"         # Times-like, matches IEEEtran

# Color palette — adapted from Tol's muted qualitative scheme
C = {
    "blue":    "#332288",
    "cyan":    "#88CCEE",
    "green":   "#44AA99",
    "yellow":  "#DDCC77",
    "red":     "#CC6677",
    "purple":  "#AA4499",
    "grey":    "#999999",
    "orange":  "#EE7733",
    "rose":    "#EE3377",
}

# Action colour map (consistent across all figures)
ACT_COLORS = [C["grey"], C["red"], C["cyan"]]
ACT_LABELS = ["Maintain", r"Heat (+1$\degree$C)", r"Cool ($-$1$\degree$C)"]


def _apply_ieee_style():
    """Configure matplotlib RC params for IEEE publication quality."""
    plt.rcParams.update({
        # Font
        "font.family":       FONT_FAMILY,
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":         FONT_SIZE,
        "axes.titlesize":    FONT_SIZE,
        "axes.labelsize":    FONT_SIZE,
        "xtick.labelsize":   FONT_SIZE - 1,
        "ytick.labelsize":   FONT_SIZE - 1,
        "legend.fontsize":   FONT_SIZE - 1,
        # Lines
        "lines.linewidth":   1.0,
        "lines.markersize":  3,
        # Axes
        "axes.linewidth":    0.6,
        "axes.grid":         True,
        "grid.linewidth":    0.4,
        "grid.alpha":        0.30,
        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        # Legend
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "0.8",
        "legend.fancybox":    False,
        "legend.handlelength": 1.5,
        # Figure
        "figure.dpi":        DPI,
        "savefig.dpi":       DPI,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.02,
    })


# ─────────────────────────────────────────────
# Helper: run one deterministic episode
# ─────────────────────────────────────────────
def _run_episode(env, model):
    obs, _ = env.reset()
    done = False
    hours, ripe, temp, acts = [], [], [], []
    total_r = 0.0
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        state = env.simulator.get_state()
        hours.append(step)
        ripe.append(state["_true_ripeness"])
        temp.append(state["_true_temperature"])
        acts.append(action)
        obs, reward, term, trunc, _ = env.step(action)
        total_r += reward
        done = term or trunc
        step += 1
    final_state = env.simulator.get_state()
    timing_err = abs(final_state["days_elapsed"] - env.target_day)
    return {
        "hours": np.array(hours),
        "ripe": np.array(ripe),
        "temp": np.array(temp),
        "acts": np.array(acts),
        "reward": total_r,
        "steps": step,
        "quality": 1.0 - final_state["_true_ripeness"],
        "timing_err": timing_err,
    }


# ─────────────────────────────────────────────
# Figure generators
# ─────────────────────────────────────────────

def fig_episode(model, config, out_dir: Path, n_episodes: int = 3):
    """Three-panel episode figure: Chromatic Index, Temperature, Action bar."""
    env = TomatoRipeningEnv(config=config, seed=42)
    for ep in range(1, n_episodes + 1):
        data = _run_episode(env, model)
        days = data["hours"] / 24.0
        n_steps = len(days)

        fig, axes = plt.subplots(
            3, 1, figsize=(IEEE_COL_WIDTH, 3.0), sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 0.6], "hspace": 0.08},
        )
        ax1, ax2, ax3 = axes

        # ── Panel 1: Chromatic index ──
        ax1.plot(days, data["ripe"], color=C["green"], linewidth=1.2,
                 label=r"$X(t)$", zorder=3)
        ax1.axhline(0.15, color=C["red"], ls="--", lw=0.8,
                     label=r"Harvest ($X{=}0.15$)")
        ax1.axhspan(0, 0.15, color=C["red"], alpha=0.06, zorder=0)
        ax1.set_ylabel("Chromatic Index $X$")
        ax1.set_ylim(-0.02, 1.05)
        ax1.legend(loc="upper right", ncol=2)
        ax1.set_title(
            f"Episode {ep}:  R = {data['reward']:+.2f},  "
            f"Q = {data['quality']:.3f},  "
            f"$\\Delta t$ = {data['timing_err']:.2f} d",
            fontsize=FONT_SIZE, fontweight="bold",
        )

        # ── Panel 2: Temperature ──
        ax2.plot(days, data["temp"], color=C["red"], linewidth=0.9)
        ax2.axhline(35.0, color=C["rose"], ls=":", lw=0.7,
                     label="Safety (35 °C)")
        ax2.axhline(12.5, color=C["blue"], ls=":", lw=0.7,
                     label=r"$T_{\rm base}$")
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_ylim(8, 40)
        ax2.legend(loc="upper right", ncol=2)

        # ── Panel 3: Action timeline ──
        for i, a in enumerate(data["acts"]):
            ax3.barh(0, 1 / 24.0, left=i / 24.0, height=0.8,
                     color=ACT_COLORS[a], edgecolor="none", linewidth=0)
        ax3.set_xlim(0, n_steps / 24.0)
        ax3.set_yticks([])
        ax3.set_ylabel("Act.", rotation=0, labelpad=12)
        ax3.set_xlabel("Time (days)")
        patches = [Patch(fc=c, label=l) for c, l in zip(ACT_COLORS, ACT_LABELS)]
        ax3.legend(handles=patches, loc="upper right", ncol=3,
                   fontsize=FONT_SIZE - 2, handlelength=1.0)

        fname = out_dir / f"episode_{ep}.png"
        fig.savefig(fname)
        plt.close(fig)
        print(f"  ✓ {fname.name}  (R={data['reward']:+.2f})")


def fig_envelope(model, config, out_dir: Path, n_episodes: int = 50):
    """Domain-randomization envelope (mean ± 1σ)."""
    env = TomatoRipeningEnv(config=config)
    trajs_x, trajs_t = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, xs, ts = False, [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            state = env.simulator.get_state()
            xs.append(state["_true_ripeness"])
            ts.append(state["_true_temperature"])
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        trajs_x.append(xs)
        trajs_t.append(ts)

    max_len = max(len(t) for t in trajs_x)
    pad_x = np.array([t + [np.nan] * (max_len - len(t)) for t in trajs_x])
    pad_t = np.array([t + [np.nan] * (max_len - len(t)) for t in trajs_t])
    days = np.arange(max_len) / 24.0
    mean_x, std_x = np.nanmean(pad_x, 0), np.nanstd(pad_x, 0)
    mean_t, std_t = np.nanmean(pad_t, 0), np.nanstd(pad_t, 0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(IEEE_COL_WIDTH, 2.8), sharex=True,
        gridspec_kw={"hspace": 0.10},
    )

    # Chromatic index
    ax1.fill_between(days, mean_x - std_x, mean_x + std_x,
                      color=C["green"], alpha=0.20, linewidth=0)
    ax1.plot(days, mean_x, color=C["green"], lw=1.2, label=r"Mean $X$")
    ax1.axhline(0.15, color=C["red"], ls="--", lw=0.8,
                 label="Harvest threshold")
    ax1.set_ylabel("Chromatic Index $X$")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="upper right")
    ax1.set_title(
        f"Domain Randomization Envelope ({n_episodes} episodes)",
        fontsize=FONT_SIZE, fontweight="bold",
    )

    # Temperature
    ax2.fill_between(days, mean_t - std_t, mean_t + std_t,
                      color=C["red"], alpha=0.15, linewidth=0)
    ax2.plot(days, mean_t, color=C["red"], lw=1.0, label="Mean $T$")
    ax2.axhline(35.0, color=C["rose"], ls=":", lw=0.7, label="Safety")
    ax2.axhline(12.5, color=C["blue"], ls=":", lw=0.7, label=r"$T_{\rm base}$")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylim(8, 40)
    ax2.legend(loc="lower right", ncol=3)

    fname = out_dir / "domain_randomization_envelope.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"  ✓ {fname.name}")


def fig_tracking(model, config, out_dir: Path):
    """Actual vs reference tracking plot."""
    env = TomatoRipeningEnv(config=config, seed=42)
    data = _run_episode(env, model)
    days = data["hours"] / 24.0

    # Reference curve: exponential decay matching the ODE physics
    # X_ref(t) = X0 * exp(-k1 * (T_mean - T_base) * t)
    x0 = data["ripe"][0]
    k1 = config.get("rl", {}).get("simulator", {}).get("k1", 0.02)
    t_base = config.get("rl", {}).get("simulator", {}).get("t_base", 12.5)
    t_mean = np.mean(data["temp"])   # use actual mean temperature
    x_ref = x0 * np.exp(-k1 * (t_mean - t_base) * days)
    x_ref = np.clip(x_ref, 0.0, 1.0)

    tracking_error = np.abs(data["ripe"] - x_ref)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(IEEE_COL_WIDTH, 2.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # Top: trajectories
    ax1.plot(days, x_ref, color=C["grey"], ls="--", lw=1.0,
             label=r"$X_{\rm ref}$", zorder=2)
    ax1.plot(days, data["ripe"], color=C["blue"], lw=1.2,
             label=r"$X_{\rm actual}$", zorder=3)
    ax1.axhline(0.15, color=C["red"], ls=":", lw=0.6, alpha=0.6)
    ax1.fill_between(days, x_ref, data["ripe"],
                      color=C["cyan"], alpha=0.15, linewidth=0)
    ax1.set_ylabel("Chromatic Index $X$")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="upper right")
    ax1.set_title("Tracking Performance", fontsize=FONT_SIZE, fontweight="bold")

    # Bottom: error
    ax2.fill_between(days, 0, tracking_error, color=C["orange"], alpha=0.4,
                      linewidth=0)
    ax2.plot(days, tracking_error, color=C["orange"], lw=0.8)
    ax2.set_ylabel(r"$|e|$")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylim(0, max(tracking_error) * 1.3 + 0.01)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(3))

    fname = out_dir / "tracking_performance.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"  ✓ {fname.name}")


def fig_comparison(model, config, out_dir: Path):
    """Side-by-side comparison of RL agent vs Fixed-Stage5 heuristic."""
    # ── RL agent ──
    env_rl = TomatoRipeningEnv(config=config, seed=42)
    rl = _run_episode(env_rl, model)

    # ── Heuristic (fixed_stage5: heat when X > 0.3) ──
    env_h = TomatoRipeningEnv(config=config, seed=42)
    obs, _ = env_h.reset()
    done = False
    h_hours, h_ripe, h_temp, h_acts = [], [], [], []
    step = 0
    while not done:
        state = env_h.simulator.get_state()
        x = state["_true_ripeness"]
        action = 1 if x > 0.3 else 0          # heuristic logic
        h_hours.append(step)
        h_ripe.append(x)
        h_temp.append(state["_true_temperature"])
        h_acts.append(action)
        obs, _, term, trunc, _ = env_h.step(action)
        done = term or trunc
        step += 1
    h_days = np.array(h_hours) / 24.0

    rl_days = rl["hours"] / 24.0

    # Use shared x-axis to avoid xlabel/title overlap
    max_days = max(rl_days[-1], h_days[-1])
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(IEEE_COL_WIDTH, 2.8), sharex=True,
        gridspec_kw={"hspace": 0.10},
    )

    # Top: Chromatic index comparison
    ax1.plot(rl_days, rl["ripe"], color=C["blue"], lw=1.2,
             label="Edge-RL", zorder=3)
    ax1.plot(h_days, h_ripe, color=C["orange"], lw=1.0, ls="--",
             label="Heuristic", zorder=2)
    ax1.axhline(0.15, color=C["red"], ls=":", lw=0.6, alpha=0.6)
    ax1.set_ylabel("Chromatic Index $X$")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="upper right")
    ax1.set_title("Policy Comparison: Edge-RL vs. Heuristic",
                   fontsize=FONT_SIZE, fontweight="bold")

    # Bottom: Temperature comparison
    ax2.plot(rl_days, rl["temp"], color=C["blue"], lw=1.0, label="Edge-RL")
    ax2.plot(h_days, h_temp, color=C["orange"], lw=0.9, ls="--",
             label="Heuristic")
    ax2.axhline(35.0, color=C["rose"], ls=":", lw=0.6, alpha=0.6,
                 label="Safety")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylim(8, 40)
    ax2.legend(loc="upper right", ncol=3)

    fname = out_dir / "comparison_trajectory.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"  ✓ {fname.name}")


def fig_distillation(model_dir: Path, out_dir: Path):
    """Distillation convergence curves (loss + accuracy)."""
    # Find a distillation directory that contains real training history
    distill_dir = None
    search_paths = list(model_dir.rglob("distill_*")) + list(Path("outputs").glob("distill_*"))
    for d in sorted(search_paths, reverse=True):  # prefer newest
        if d.is_dir() and (d / "training_history.json").exists():
            distill_dir = d
            break
    if distill_dir is None:
        print("  ✗ No distillation directory with training_history.json found")
        print("    Re-run distillation to generate real data.")
        return

    # Require real training history — never fabricate data
    history_path = distill_dir / "training_history.json"

    with open(history_path) as f:
        hist = json.load(f)
    epochs = np.arange(1, len(hist["loss"]) + 1)
    loss = np.array(hist["loss"])
    acc = np.array(hist.get("accuracy", hist.get("acc", [])))

    fig, ax1 = plt.subplots(figsize=(IEEE_COL_WIDTH, 1.8))
    ax2 = ax1.twinx()

    l1, = ax1.plot(epochs, loss, color=C["red"], lw=1.0, label="CE Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss", color=C["red"])
    ax1.tick_params(axis="y", colors=C["red"])
    ax1.set_ylim(bottom=0)

    l2, = ax2.plot(epochs, acc * 100, color=C["blue"], lw=1.0,
                    label="Action Fidelity")
    ax2.set_ylabel("Action Fidelity (%)", color=C["blue"])
    ax2.tick_params(axis="y", colors=C["blue"])
    ax2.set_ylim(50, 102)
    ax2.axhline(97.8, color=C["blue"], ls=":", lw=0.5, alpha=0.5)

    ax1.set_title("Policy Distillation Convergence",
                   fontsize=FONT_SIZE, fontweight="bold")
    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="center right", fontsize=FONT_SIZE - 2)

    fname = out_dir / "distillation_curves.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"  ✓ {fname.name}")


def fig_training(model_dir: Path, out_dir: Path):
    """Training reward curve (from SB3 eval logs)."""
    eval_log = model_dir / "eval_logs" / "evaluations.npz"
    if not eval_log.exists():
        print(f"  ⚠ {eval_log} not found — skipping training curve")
        return

    data = np.load(eval_log)
    timesteps = data["timesteps"]
    results = data["results"]          # (n_evals, n_episodes)
    mean_r = np.mean(results, axis=1)
    std_r = np.std(results, axis=1)

    # Clip outliers for cleaner y-range
    p5, p95 = np.percentile(mean_r, 5), np.percentile(mean_r, 95)
    y_pad = (p95 - p5) * 0.3
    y_lo = max(p5 - y_pad, np.min(mean_r - std_r))
    y_hi = p95 + y_pad

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 1.8))
    ax.fill_between(timesteps / 1e3, mean_r - std_r, mean_r + std_r,
                     color=C["green"], alpha=0.18, linewidth=0)
    ax.plot(timesteps / 1e3, mean_r, color=C["green"], lw=1.0)
    ax.set_xlabel("Training Timesteps ($\\times 10^3$)")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("DQN Training Curve", fontsize=FONT_SIZE, fontweight="bold")
    ax.axhline(0, color="k", ls=":", lw=0.4, alpha=0.4)
    ax.set_ylim(y_lo, y_hi)

    fname = out_dir / "policy_improvement.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"  ✓ {fname.name}")


def fig_action_dist(model, config, out_dir: Path, n_episodes: int = 100):
    """Action distribution bar chart."""
    env = TomatoRipeningEnv(config=config, seed=0)
    all_actions = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(int(action))
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

    counts = [all_actions.count(i) for i in range(3)]
    pcts = [c / len(all_actions) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 1.6))
    bars = ax.bar(ACT_LABELS, pcts, color=ACT_COLORS,
                  edgecolor="black", linewidth=0.4, width=0.55)
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{p:.1f}%", ha="center", fontsize=FONT_SIZE - 1,
                fontweight="bold")
    ax.set_ylabel("Frequency (%)")
    ax.set_ylim(0, max(pcts) + 8)
    ax.set_title("Learned Action Distribution",
                  fontsize=FONT_SIZE, fontweight="bold")

    fname = out_dir / "action_distribution.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"  ✓ {fname.name}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

FIGURE_MAP = {
    "episode":      lambda m, c, d, md: fig_episode(m, c, d),
    "envelope":     lambda m, c, d, md: fig_envelope(m, c, d),
    "tracking":     lambda m, c, d, md: fig_tracking(m, c, d),
    "comparison":   lambda m, c, d, md: fig_comparison(m, c, d),
    "distillation": lambda m, c, d, md: fig_distillation(md, d),
    "training":     lambda m, c, d, md: fig_training(md, d),
    "action_dist":  lambda m, c, d, md: fig_action_dist(m, c, d),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=str, default="outputs/rl_20260217_095300",
                        help="Path to RL training output directory")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml")
    parser.add_argument("--out-dir", type=str, default="docs/thesis/images",
                        help="Output directory for figures")
    parser.add_argument("--figures", nargs="*", default=list(FIGURE_MAP.keys()),
                        choices=list(FIGURE_MAP.keys()),
                        help="Which figures to generate (default: all)")
    args = parser.parse_args()

    _apply_ieee_style()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_path = model_dir / "best_model" / "best_model.zip"
    if not model_path.exists():
        model_path = model_dir / "final_model.zip"
    print(f"Loading model from {model_path}")
    model = DQN.load(model_path)

    print(f"Generating {len(args.figures)} figure(s) → {out_dir}/\n")
    for name in args.figures:
        print(f"[{name}]")
        FIGURE_MAP[name](model, config, out_dir, model_dir)

    print(f"\n✅ All figures saved to {out_dir}/ at {DPI} DPI")


if __name__ == "__main__":
    main()
