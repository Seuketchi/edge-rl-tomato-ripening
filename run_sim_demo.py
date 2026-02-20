
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import yaml
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from ml_training.rl.environment import TomatoRipeningEnv

MODEL_PATH = "outputs/rl_20260217_150831/best_model/best_model.zip"


def run_visual_demo():
    print("=== Edge-RL Trained DQN — Simulation Demo ===\n")

    # Load config & model
    with open("ml_training/config.yaml") as f:
        config = yaml.safe_load(f)

    model = DQN.load(MODEL_PATH)
    env = TomatoRipeningEnv(config=config, seed=42)

    # ---------- run 3 episodes ----------
    n_episodes = 3
    all_episodes = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        history = {"hours": [], "ripeness": [], "temperature": [],
                   "humidity": [], "actions": [], "reward": []}
        total_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            state = env.simulator.get_state()

            history["hours"].append(step)
            history["ripeness"].append(state["_true_ripeness"])
            history["temperature"].append(state["_true_temperature"])
            history["humidity"].append(state["_true_humidity"])
            history["actions"].append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            history["reward"].append(total_reward)
            done = terminated or truncated
            step += 1

        all_episodes.append(history)
        action_names = ["Maintain", "Heat", "Cool"]
        act_counts = [history["actions"].count(a) for a in range(3)]
        act_pcts = [c / len(history["actions"]) * 100 for c in act_counts]

        print(f"Episode {ep+1}: {step} steps, reward {total_reward:.2f}")
        print(f"  Final ripeness: {history['ripeness'][-1]:.3f}")
        print(f"  Actions: " + ", ".join(
            f"{action_names[i]} {act_pcts[i]:.0f}%" for i in range(3)))
        print()

    # ---------- plot best episode ----------
    best_idx = max(range(n_episodes),
                   key=lambda i: all_episodes[i]["reward"][-1])
    h = all_episodes[best_idx]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1]})

    # --- Panel 1: Ripeness + Temperature ---
    ax1 = axes[0]
    c1 = "tab:green"
    ax1.plot(h["hours"], h["ripeness"], color=c1, linewidth=2,
             label="Chromatic Index X")
    ax1.set_ylabel("Chromatic Index X", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(-0.05, 1.1)
    ax1.axhspan(0, 0.15, color="green", alpha=0.08, label="Harvest zone (X ≤ 0.15)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Edge-RL DQN — Trained Agent Ripening Control", fontsize=13)

    ax1b = ax1.twinx()
    c2 = "tab:red"
    ax1b.plot(h["hours"], h["temperature"], color=c2, linestyle="--",
              alpha=0.7, linewidth=1.5, label="Chamber Temp")
    ax1b.set_ylabel("Temperature (°C)", color=c2)
    ax1b.tick_params(axis="y", labelcolor=c2)
    ax1b.legend(loc="upper right", fontsize=9)

    # --- Panel 2: Cumulative reward ---
    ax2 = axes[1]
    ax2.plot(h["hours"], h["reward"], color="tab:purple", linewidth=1.5)
    ax2.set_ylabel("Cumulative Reward")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Actions ---
    ax3 = axes[2]
    action_colors = {0: "gray", 1: "red", 2: "blue"}
    colors = [action_colors[a] for a in h["actions"]]
    ax3.scatter(h["hours"], h["actions"], c=colors, s=12, alpha=0.8)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["Maintain", "Heat", "Cool (OFF)"])
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Action")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = "outputs/ripening_trajectory_demo.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")
    print("==================================================")


if __name__ == "__main__":
    run_visual_demo()
