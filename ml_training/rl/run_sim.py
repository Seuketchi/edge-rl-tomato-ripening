"""Unified simulation runner for the Edge-RL tomato ripening system.

Replaces run_simulation.py, run_evaluation.py, verify_env.py, run_sim_demo.py,
and run_box2d_viz.py with a single entry point.

Usage:
    python -m ml_training.rl.run_sim --mode verify
    python -m ml_training.rl.run_sim --mode demo   [--model PATH] [--seed N]
    python -m ml_training.rl.run_sim --mode eval   [--model PATH] [--episodes N] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import DQN

from ml_training.rl.environment import TomatoRipeningEnv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_NAMES = ["Maintain", "Heat (+1°C)", "Cool (−1°C)"]
ACTION_COLORS = ["#888888", "#e74c3c", "#3498db"]
ENERGY_COST = {0: 0.1, 1: 1.0, 2: 0.3}

# 6-stop ROYG gradient: X=1.0 (unripe green) → X=0.0 (ripe red)
# Breakpoints: 1.0, 0.85, 0.65, 0.40, 0.15, 0.0
_ROYG_X = np.array([1.0, 0.85, 0.65, 0.40, 0.15, 0.0])
_ROYG_STOPS = np.array([
    [0.20, 0.78, 0.22],  # X=1.0  deep green
    [0.55, 0.75, 0.15],  # X=0.85 yellow-green
    [0.90, 0.80, 0.08],  # X=0.65 yellow
    [0.95, 0.55, 0.08],  # X=0.40 orange
    [0.90, 0.22, 0.10],  # X=0.15 light-red
    [0.75, 0.08, 0.08],  # X=0.0  deep red
], dtype=float)

DEFAULT_CONFIG_PATH = Path("ml_training/config.yaml")


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------

def royg_color(x: float) -> tuple[float, float, float]:
    """Map X∈[0,1] → ROYG RGB via 6-stop linear interpolation.

    ROYG convention: X=1.0 is unripe green, X=0.0 is ripe red.
    """
    x = float(np.clip(x, 0.0, 1.0))
    # np.interp requires xp monotonically increasing; _ROYG_X is decreasing,
    # so reverse both arrays.
    r = float(np.interp(x, _ROYG_X[::-1], _ROYG_STOPS[::-1, 0]))
    g = float(np.interp(x, _ROYG_X[::-1], _ROYG_STOPS[::-1, 1]))
    b = float(np.interp(x, _ROYG_X[::-1], _ROYG_STOPS[::-1, 2]))
    return (r, g, b)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def discover_model(model_arg: str | None) -> Path:
    """Return model path from argument or auto-discover the latest trained model.

    Search order: outputs/rl_*/best_model/best_model.zip, then
    outputs/rl_*/final_model.zip, sorted lexicographically descending
    (timestamp in dir name → newest first).
    """
    if model_arg is not None:
        p = Path(model_arg)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p

    candidates = sorted(Path(".").glob("outputs/rl_*/best_model/best_model.zip"), reverse=True)
    if not candidates:
        candidates = sorted(Path(".").glob("outputs/rl_*/final_model.zip"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No trained model found. Train one first:\n"
            "  python -m ml_training.rl.train_dqn --config ml_training/config.yaml"
        )
    model_path = candidates[0]
    print(f"Auto-discovered model: {model_path}")
    return model_path


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML config; returns empty dict if file not found."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def make_output_dir(base: str | None = None) -> Path:
    """Create and return outputs/sim_<timestamp>/ (or custom base) directory."""
    if base is not None:
        p = Path(base)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = Path("outputs") / f"sim_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env: TomatoRipeningEnv, model: DQN, seed: int | None = None) -> dict:
    """Execute one episode and collect full trajectory data.

    Calls env.simulator.get_state() BEFORE env.step() to capture the
    decision-time state (true values, not post-step noisy readings).

    Returns:
        Dict with keys: hours, X, temperature, humidity, actions, rewards,
        energy, x_ref, total_reward, total_energy, final_X, auto_harvest,
        timing_error, quality.
    """
    obs, _ = env.reset(seed=seed)

    traj: dict = {
        "hours": [],
        "X": [],
        "temperature": [],
        "humidity": [],
        "actions": [],
        "rewards": [],
        "energy": [],
        "x_ref": [],
    }

    done = False
    step = 0
    total_reward = 0.0
    info: dict = {}

    while not done:
        state = env.simulator.get_state()

        traj["hours"].append(step)
        traj["X"].append(state["_true_ripeness"])
        traj["temperature"].append(state["_true_temperature"])
        traj["humidity"].append(state["_true_humidity"])
        traj["x_ref"].append(env.simulator.compute_x_ref(state["days_elapsed"]))

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        traj["actions"].append(action)
        traj["energy"].append(ENERGY_COST[action])

        obs, reward, terminated, truncated, info = env.step(action)
        traj["rewards"].append(float(reward))
        total_reward += float(reward)
        done = terminated or truncated
        step += 1

    # Append one final state after the loop
    state = env.simulator.get_state()
    traj["hours"].append(step)
    traj["X"].append(state["_true_ripeness"])
    traj["temperature"].append(state["_true_temperature"])
    traj["humidity"].append(state["_true_humidity"])
    traj["x_ref"].append(env.simulator.compute_x_ref(state["days_elapsed"]))

    traj["total_reward"] = total_reward
    traj["total_energy"] = sum(traj["energy"])
    traj["final_X"] = state["_true_ripeness"]
    traj["auto_harvest"] = info.get("auto_harvest", False)
    traj["timing_error"] = info.get("timing_error", float("nan"))
    traj["quality"] = info.get("harvest_quality", float("nan"))

    return traj


# ---------------------------------------------------------------------------
# Plotting helpers (private)
# ---------------------------------------------------------------------------

def _plot_demo(traj: dict, episode_idx: int, out_path: Path) -> None:
    """3-panel figure: ripeness with ROYG background, temperature, action bars."""
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1]},
    )

    hours = np.array(traj["hours"])
    X = np.array(traj["X"])
    temp = np.array(traj["temperature"])
    actions = np.array(traj["actions"])

    # Panel 1: Chromatic Index with ROYG background spans
    ax1 = axes[0]
    for i in range(len(hours) - 1):
        ax1.axvspan(hours[i], hours[i + 1], alpha=0.18,
                    color=royg_color(X[i]), linewidth=0)
    ax1.plot(hours, X, color="white", linewidth=2.0, label="Chromatic Index $X$", zorder=3)
    ax1.axhline(y=0.15, color="red", linestyle="--", alpha=0.8,
                label="Harvest threshold ($X=0.15$)", zorder=4)
    if traj["auto_harvest"]:
        ax1.axvline(x=len(actions), color="gold", linestyle="-",
                    linewidth=1.5, alpha=0.9, label="Harvest", zorder=4)
    ax1.set_ylabel("Chromatic Index $X$ (ROYG)", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.2)
    harvest_type = "Auto-harvest" if traj["auto_harvest"] else "Deadline"
    ax1.set_title(
        f"Episode {episode_idx + 1}: {harvest_type} at step {len(actions)} | "
        f"Reward: {traj['total_reward']:+.2f} | Quality: {traj['quality']:.3f} | "
        f"Timing err: {traj['timing_error']:.2f}d",
        fontsize=11,
    )

    # Panel 2: Temperature with safety bounds shaded
    ax2 = axes[1]
    ax2.plot(hours[:-1], temp[:-1], color="#e74c3c", linewidth=1.5, label="Temperature")
    ax2.axhline(y=35.0, color="red", linestyle=":", alpha=0.6, label="Safety limit (35°C)")
    ax2.axhline(y=12.5, color="steelblue", linestyle=":", alpha=0.6, label="$T_{base}$ (12.5°C)")
    ax2.fill_between(hours[:-1], 35.0, 40.0, alpha=0.07, color="red")
    ax2.fill_between(hours[:-1], 0.0, 12.5, alpha=0.07, color="steelblue")
    ax2.set_ylabel("Temperature (°C)", fontsize=11)
    ax2.set_ylim(10, 38)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.2)

    # Panel 3: Action bars coloured by action type
    ax3 = axes[2]
    for i, a in enumerate(actions):
        ax3.bar(hours[i], 1, width=0.9, color=ACTION_COLORS[a], alpha=0.85, linewidth=0)
    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(ACTION_COLORS, ACTION_NAMES)]
    ax3.legend(handles=patches, loc="upper right", fontsize=9, ncol=3)
    ax3.set_ylabel("Action", fontsize=11)
    ax3.set_yticks([])
    ax3.set_xlabel("Time (hours)", fontsize=11)
    ax3.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_envelope(trajectories: list[dict], out_path: Path) -> None:
    """Mean ± std envelope across all episodes."""
    max_len = max(len(t["X"]) for t in trajectories)

    # Pad shorter episodes with NaN so we can nanmean across them
    X_mat = np.full((len(trajectories), max_len), np.nan)
    for i, t in enumerate(trajectories):
        n = len(t["X"])
        X_mat[i, :n] = t["X"]

    mean_X = np.nanmean(X_mat, axis=0)
    std_X = np.nanstd(X_mat, axis=0)
    hours = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(12, 5))
    for t in trajectories:
        ax.plot(t["hours"], t["X"], alpha=0.12, linewidth=0.8, color="gray")
    ax.fill_between(hours, mean_X - std_X, mean_X + std_X,
                    alpha=0.28, color="#3498db", label="±1 std")
    ax.plot(hours, mean_X, color="#2980b9", linewidth=2.0, label="Mean $X$")
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.7, label="Harvest threshold")

    rewards = [t["total_reward"] for t in trajectories]
    ax.set_title(
        f"DQN Agent: {len(trajectories)} Episodes (Domain Randomization Envelope)\n"
        f"Mean reward: {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}",
        fontsize=12,
    )
    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_ylabel("Chromatic Index $X$ (ROYG)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_action_dist(trajectories: list[dict], out_path: Path) -> None:
    """Aggregate action frequency bar chart across all episodes."""
    all_actions = np.concatenate([t["actions"] for t in trajectories])
    counts = np.bincount(all_actions, minlength=3)
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(ACTION_NAMES, pcts, color=ACTION_COLORS, edgecolor="white", linewidth=1.5)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Action Frequency (%)", fontsize=11)
    ax.set_title("DQN Learned Action Distribution", fontsize=12)
    ax.set_ylim(0, max(pcts) + 12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_demo(
    model: DQN,
    config: dict,
    output_dir: Path,
    seed: int = 42,
    n_episodes: int = 3,
) -> None:
    """Run n_episodes, plot the best-reward episode as a 3-panel figure.

    Output: demo_trajectory.png
    """
    print(f"Running {n_episodes} demo episodes...")
    trajectories = []
    for i in range(n_episodes):
        env = TomatoRipeningEnv(config=config, seed=seed + i)
        traj = run_episode(env, model, seed=seed + i)
        trajectories.append(traj)
        harvest_type = "auto-harvest" if traj["auto_harvest"] else "deadline"
        print(
            f"  Episode {i + 1}: reward={traj['total_reward']:+.2f}, "
            f"steps={len(traj['actions'])}, final_X={traj['final_X']:.3f}, "
            f"quality={traj['quality']:.3f}, {harvest_type}"
        )

    best = max(trajectories, key=lambda t: t["total_reward"])
    best_idx = trajectories.index(best)
    _plot_demo(best, best_idx, output_dir / "demo_trajectory.png")
    print(f"Demo complete. Best episode: {best_idx + 1} (reward={best['total_reward']:+.2f})")


def run_eval(
    model: DQN,
    config: dict,
    output_dir: Path,
    n_episodes: int = 30,
    seed: int = 42,
) -> None:
    """Run N episodes, produce envelope + action distribution + summary JSON.

    Outputs: eval_envelope.png, eval_action_dist.png, eval_summary.json
    """
    print(f"Running {n_episodes} evaluation episodes...")
    trajectories = []
    for i in range(n_episodes):
        env = TomatoRipeningEnv(config=config, seed=seed + i)
        traj = run_episode(env, model, seed=seed + i)
        trajectories.append(traj)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_episodes} episodes done...")

    _plot_envelope(trajectories, output_dir / "eval_envelope.png")
    _plot_action_dist(trajectories, output_dir / "eval_action_dist.png")

    rewards = [t["total_reward"] for t in trajectories]
    timing = [t["timing_error"] for t in trajectories if np.isfinite(t["timing_error"])]
    quality = [t["quality"] for t in trajectories if np.isfinite(t["quality"])]
    energy = [t["total_energy"] for t in trajectories]
    auto_harvest_rate = sum(1 for t in trajectories if t["auto_harvest"]) / n_episodes

    summary = {
        "n_episodes": n_episodes,
        "reward": {"mean": float(np.mean(rewards)), "std": float(np.std(rewards))},
        "timing_error_days": {
            "mean": float(np.mean(timing)) if timing else float("nan"),
            "std": float(np.std(timing)) if timing else float("nan"),
        },
        "quality": {
            "mean": float(np.mean(quality)) if quality else float("nan"),
            "std": float(np.std(quality)) if quality else float("nan"),
        },
        "energy": {"mean": float(np.mean(energy)), "std": float(np.std(energy))},
        "auto_harvest_rate": auto_harvest_rate,
    }

    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete:")
    print(f"  Mean reward:       {summary['reward']['mean']:+.2f} ± {summary['reward']['std']:.2f}")
    if timing:
        print(
            f"  Mean timing error: {summary['timing_error_days']['mean']:.2f} ± "
            f"{summary['timing_error_days']['std']:.2f} days"
        )
    print(f"  Mean quality:      {summary['quality']['mean']:.3f} ± {summary['quality']['std']:.3f}")
    print(f"  Auto-harvest rate: {auto_harvest_rate * 100:.0f}%")
    print(f"  Outputs saved to:  {output_dir}/")


def run_verify(config: dict) -> bool:
    """Verify environment correctness without a model.

    Checks:
        1. obs_shape — reset returns (16,) float32 observation
        2. all_actions — actions 0, 1, 2 each produce valid obs + finite reward
        3. reward_finite — full episode with action=0 yields finite rewards throughout

    Returns:
        True if all 3 checks pass, False otherwise.
    """
    env = TomatoRipeningEnv(config=config, seed=0, state_variant="B")
    results: list[bool] = []

    # Check 1: observation shape and dtype
    obs, _ = env.reset()
    passed = obs.shape == (16,) and obs.dtype == np.float32
    print(f"{'[PASS]' if passed else '[FAIL]'} obs_shape: {obs.shape} — expected (16,)")
    results.append(passed)

    # Check 2: all 3 actions return valid obs + finite reward
    action_ok = True
    for a in range(3):
        obs, _ = env.reset()
        step_obs, reward, _, _, _ = env.step(a)
        ok = step_obs.shape == (16,) and np.isfinite(reward)
        print(f"{'[PASS]' if ok else '[FAIL]'} action_{a}: step returned finite reward")
        if not ok:
            action_ok = False
    results.append(action_ok)

    # Check 3: reward finite over a complete episode (maintain action)
    obs, _ = env.reset()
    n_steps = 0
    all_finite = True
    done = False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(0)
        n_steps += 1
        if not np.isfinite(reward):
            all_finite = False
            print(f"[FAIL] reward_finite: non-finite reward at step {n_steps}: {reward}")
            break
        done = terminated or truncated
    if all_finite:
        print(f"[PASS] reward_finite: {n_steps} steps all finite")
    results.append(all_finite)

    n_pass = sum(results)
    print(f"Verify complete: {n_pass}/{len(results)} checks passed")
    return n_pass == len(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge-RL simulation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  verify  Check environment correctness (no model required)
  demo    Run 3 episodes, save best-trajectory plot
  eval    Run N episodes, save envelope + action-dist + summary JSON
        """,
    )
    parser.add_argument(
        "--mode", choices=["demo", "eval", "verify"], required=True,
        help="Simulation mode",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model .zip (auto-discovered if omitted)")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of episodes for eval mode (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/sim_<timestamp>/)")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.mode == "verify":
        ok = run_verify(config)
        sys.exit(0 if ok else 1)

    model_path = discover_model(args.model)
    print(f"Loading model: {model_path}")
    model = DQN.load(str(model_path))
    output_dir = make_output_dir(args.output_dir)

    if args.mode == "demo":
        run_demo(model, config, output_dir, seed=args.seed, n_episodes=3)
    elif args.mode == "eval":
        run_eval(model, config, output_dir, n_episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
