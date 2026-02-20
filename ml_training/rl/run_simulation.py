"""Run a visual simulation of the trained DQN agent.

Loads the trained DQN model and runs episodes in the TomatoRipeningEnv,
producing trajectory plots suitable for thesis figures.

Usage:
    python -m ml_training.rl.run_simulation --model outputs/rl_20260217_095300/final_model.zip
    python -m ml_training.rl.run_simulation --model outputs/rl_20260217_095300/final_model.zip --episodes 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
from stable_baselines3 import DQN

from ml_training.rl.environment import TomatoRipeningEnv


ACTION_NAMES = ["Maintain", "Heat (+1°C)", "Cool (−1°C)"]
ACTION_COLORS = ["#888888", "#e74c3c", "#3498db"]


def create_env(config: dict, seed: int = 42) -> TomatoRipeningEnv:
    """Create an environment from config."""
    env = TomatoRipeningEnv(config=config, seed=seed)
    return env


def get_tomato_color_royg(x: float) -> tuple[float, float, float]:
    """Map chromatic index X (ROYG: 1=Green, 0=Red) to RGB color."""
    x = np.clip(x, 0.0, 1.0)
    if x > 0.66:
        # Green zone (X near 1.0)
        frac = (x - 0.66) / 0.34
        return (0.2 + 0.3 * (1 - frac), 0.6 + 0.3 * frac, 0.1)
    elif x > 0.33:
        # Yellow-Orange zone
        frac = (x - 0.33) / 0.33
        return (0.9 + 0.1 * frac, 0.5 + 0.4 * frac, 0.05)
    else:
        # Red zone (X near 0.0)
        frac = x / 0.33
        return (0.8 + 0.2 * (1 - frac), 0.1 + 0.3 * frac, 0.05)


def run_episode(env: TomatoRipeningEnv, model: DQN, seed: int | None = None) -> dict:
    """Run one episode and collect trajectory data."""
    obs, info = env.reset(seed=seed)

    trajectory = {
        "hours": [],
        "X": [],
        "temperature": [],
        "humidity": [],
        "actions": [],
        "rewards": [],
        "x_ref": [],
    }

    done = False
    step = 0
    total_reward = 0.0

    while not done:
        sim_state = env.simulator.get_state()

        # Record state
        trajectory["hours"].append(step)
        trajectory["X"].append(sim_state["ripeness"])
        trajectory["temperature"].append(sim_state["temperature"])
        trajectory["humidity"].append(sim_state["humidity"])
        trajectory["x_ref"].append(sim_state.get("x_ref", np.nan))

        # Agent decides
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        trajectory["actions"].append(action)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory["rewards"].append(reward)
        total_reward += reward
        done = terminated or truncated
        step += 1

    # Record final state
    sim_state = env.simulator.get_state()
    trajectory["hours"].append(step)
    trajectory["X"].append(sim_state["ripeness"])
    trajectory["temperature"].append(sim_state["temperature"])
    trajectory["humidity"].append(sim_state["humidity"])
    trajectory["x_ref"].append(sim_state.get("x_ref", np.nan))

    trajectory["total_reward"] = total_reward
    trajectory["final_X"] = sim_state["ripeness"]
    trajectory["auto_harvest"] = info.get("auto_harvest", False)
    trajectory["deadline_harvest"] = info.get("deadline_harvest", False)
    trajectory["timing_error"] = info.get("timing_error", np.nan)
    trajectory["quality"] = info.get("harvest_quality", np.nan)

    return trajectory


def plot_single_episode(traj: dict, output_path: str, episode_idx: int = 0) -> None:
    """Create a detailed 3-panel trajectory plot for one episode."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1]})

    hours = np.array(traj["hours"])
    X = np.array(traj["X"])
    temp = np.array(traj["temperature"])
    actions = np.array(traj["actions"])
    rewards = np.array(traj["rewards"])

    # --- Panel 1: Chromatic Index ---
    ax1 = axes[0]
    ax1.plot(hours, X, color="#2ecc71", linewidth=2.5, label="Chromatic Index $X$")
    ax1.axhline(y=0.15, color="red", linestyle="--", alpha=0.7, label="Harvest threshold ($X=0.15$)")
    ax1.fill_between(hours, 0, 0.15, alpha=0.08, color="red")

    # Color the background based on X value
    for i in range(len(hours) - 1):
        color = get_tomato_color_royg(X[i])
        ax1.axvspan(hours[i], hours[i + 1], alpha=0.15, color=color, linewidth=0)

    ax1.set_ylabel("Chromatic Index $X$ (ROYG)", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    harvest_type = "Auto-harvest" if traj["auto_harvest"] else "Deadline"
    ax1.set_title(
        f"Episode {episode_idx + 1}: {harvest_type} at step {len(actions)} | "
        f"Reward: {traj['total_reward']:+.2f} | "
        f"Quality: {traj['quality']:.3f} | "
        f"Timing err: {traj['timing_error']:.2f}d",
        fontsize=11,
    )

    # --- Panel 2: Temperature ---
    ax2 = axes[1]
    ax2.plot(hours[:-1], temp[:-1], color="#e74c3c", linewidth=1.5, label="Temperature")
    ax2.axhline(y=35, color="red", linestyle=":", alpha=0.5, label="Safety limit (35°C)")
    ax2.axhline(y=12.5, color="blue", linestyle=":", alpha=0.5, label="$T_{base}$ (12.5°C)")
    ax2.fill_between(hours[:-1], 35, 40, alpha=0.06, color="red")
    ax2.fill_between(hours[:-1], 0, 12.5, alpha=0.06, color="blue")
    ax2.set_ylabel("Temperature (°C)", fontsize=11)
    ax2.set_ylim(10, 38)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Actions ---
    ax3 = axes[2]
    for i, a in enumerate(actions):
        ax3.bar(hours[i], 1, width=0.9, color=ACTION_COLORS[a], alpha=0.8)
    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(ACTION_COLORS, ACTION_NAMES)]
    ax3.legend(handles=patches, loc="upper right", fontsize=9, ncol=3)
    ax3.set_ylabel("Action", fontsize=11)
    ax3.set_yticks([])
    ax3.set_xlabel("Time (hours)", fontsize=11)
    ax3.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_multi_episode_overlay(trajectories: list[dict], output_path: str) -> None:
    """Plot multiple episode trajectories overlaid (for domain randomization envelope)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax1, ax2 = axes

    for i, traj in enumerate(trajectories):
        hours = np.array(traj["hours"])
        X = np.array(traj["X"])
        temp = np.array(traj["temperature"])
        alpha = 0.4 if len(trajectories) > 3 else 0.8
        ax1.plot(hours, X, alpha=alpha, linewidth=1.2)
        ax2.plot(hours[:-1], temp[:-1], alpha=alpha, linewidth=1.0)

    ax1.axhline(y=0.15, color="red", linestyle="--", alpha=0.7, label="Harvest threshold")
    ax1.fill_between([0, max(len(t["hours"]) for t in trajectories)], 0, 0.15, alpha=0.06, color="red")
    ax1.set_ylabel("Chromatic Index $X$", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        f"DQN Agent: {len(trajectories)} Episodes with Domain Randomization\n"
        f"Mean reward: {np.mean([t['total_reward'] for t in trajectories]):+.2f} ± "
        f"{np.std([t['total_reward'] for t in trajectories]):.2f}",
        fontsize=11,
    )

    ax2.axhline(y=35, color="red", linestyle=":", alpha=0.5, label="Safety limit")
    ax2.axhline(y=12.5, color="blue", linestyle=":", alpha=0.5, label="$T_{base}$")
    ax2.set_ylabel("Temperature (°C)", fontsize=11)
    ax2.set_xlabel("Time (hours)", fontsize=11)
    ax2.set_ylim(10, 38)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_action_distribution(trajectories: list[dict], output_path: str) -> None:
    """Plot aggregate action distribution across episodes."""
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
    ax.set_ylim(0, max(pcts) + 10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DQN simulation with trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .zip")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to simulate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Output dir
    if args.output_dir is None:
        model_dir = Path(args.model).parent
        output_dir = model_dir / "simulation"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = DQN.load(args.model)

    # Run episodes
    print(f"\nRunning {args.episodes} simulation episodes...")
    trajectories = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        env = create_env(config, seed=seed)
        traj = run_episode(env, model, seed=seed)
        trajectories.append(traj)

        harvest = "auto-harvest" if traj["auto_harvest"] else "deadline"
        print(f"  Episode {ep + 1}: reward={traj['total_reward']:+.2f}, "
              f"steps={len(traj['actions'])}, "
              f"final_X={traj['final_X']:.3f}, "
              f"quality={traj['quality']:.3f}, "
              f"timing_err={traj['timing_error']:.2f}d, "
              f"{harvest}")

    # Generate plots
    print(f"\nGenerating plots in {output_dir}/...")

    # Individual episode plots (first 3)
    for i, traj in enumerate(trajectories[:3]):
        plot_single_episode(traj, str(output_dir / f"episode_{i + 1}.png"), episode_idx=i)

    # Multi-episode overlay
    plot_multi_episode_overlay(trajectories, str(output_dir / "trajectory_envelope.png"))

    # Action distribution
    plot_action_distribution(trajectories, str(output_dir / "action_distribution.png"))

    # Summary stats
    rewards = [t["total_reward"] for t in trajectories]
    timing = [t["timing_error"] for t in trajectories]
    quality = [t["quality"] for t in trajectories]
    auto_harvest_rate = sum(1 for t in trajectories if t["auto_harvest"]) / len(trajectories)

    print(f"\n{'=' * 55}")
    print(f"  SIMULATION SUMMARY ({args.episodes} episodes)")
    print(f"{'=' * 55}")
    print(f"  Mean reward:       {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean timing error: {np.mean(timing):.2f} ± {np.std(timing):.2f} days")
    print(f"  Mean quality:      {np.mean(quality):.3f} ± {np.std(quality):.3f}")
    print(f"  Auto-harvest rate: {auto_harvest_rate * 100:.0f}%")
    print(f"  All plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
