"""DQN training script for tomato harvest timing policy.

Trains a Deep Q-Network agent using Stable Baselines3 on the
TomatoRipeningEnv (Discrete action space). Includes evaluation against
fixed-rule baselines.

Usage:
    python -m ml_training.rl.train_dqn --config ml_training/config.yaml
    python -m ml_training.rl.train_dqn --config ml_training/config.yaml --total-timesteps 1000 --smoke-test
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from ml_training.rl.environment import TomatoRipeningEnv


class MetricsCallback(BaseCallback):
    """Callback to log training metrics for plotting."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        # Collect episode stats from info buffer
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


def create_env(config: dict, seed: int = 42) -> TomatoRipeningEnv:
    """Create a single environment from config."""
    return TomatoRipeningEnv(config=config, seed=seed)


def evaluate_baseline(
    config: dict,
    policy_type: str = "fixed_stage5",
    n_episodes: int = 100,
    seed: int = 42,
) -> dict:
    """Evaluate a fixed-rule baseline policy.

    Args:
        config: Full config dict.
        policy_type: Baseline type:
            - "fixed_stage5": Harvest when ripeness >= 4.5
            - "fixed_day": Harvest on target day regardless of ripeness
            - "random": Random actions
        n_episodes: Number of evaluation episodes.
        seed: Random seed.

    Returns:
        Dict with evaluation metrics.
    """
    rng = np.random.default_rng(seed)
    rewards = []
    timing_errors = []
    harvest_qualities = []

    for i in range(n_episodes):
        env = TomatoRipeningEnv(config=config, seed=seed + i)
        obs, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            if policy_type == "fixed_stage5":
                # In ROYG, X=0 is ripe. Heat to accelerate when X > 0.15
                x = obs[0]
                action = 1 if x > 0.3 else 0  # Heat or maintain
            elif policy_type == "fixed_day":
                # Maintain temperature — let natural ripening happen
                action = 0
            elif policy_type == "random":
                action = rng.integers(0, 3)  # 3 actions: maintain/heat/cool
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        if "timing_error" in info:
            timing_errors.append(info["timing_error"])
        if "harvest_quality" in info:
            harvest_qualities.append(info["harvest_quality"])

    return {
        "policy": policy_type,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_timing_error": float(np.mean(timing_errors)) if timing_errors else None,
        "mean_quality": float(np.mean(harvest_qualities)) if harvest_qualities else None,
        "harvest_rate": len(timing_errors) / n_episodes,
    }


def evaluate_trained_policy(
    model: DQN,
    config: dict,
    n_episodes: int = 100,
    seed: int = 42,
) -> dict:
    """Evaluate the trained DQN policy.

    Returns:
        Dict with evaluation metrics.
    """
    rewards = []
    timing_errors = []
    harvest_qualities = []
    action_counts = {0: 0, 1: 0, 2: 0}

    for i in range(n_episodes):
        env = TomatoRipeningEnv(config=config, seed=seed + i)
        obs, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        if "timing_error" in info:
            timing_errors.append(info["timing_error"])
        if "harvest_quality" in info:
            harvest_qualities.append(info["harvest_quality"])

    total_actions = sum(action_counts.values())
    action_names = ["maintain", "heat", "cool"]

    return {
        "policy": "DQN (trained)",
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_timing_error": float(np.mean(timing_errors)) if timing_errors else None,
        "mean_quality": float(np.mean(harvest_qualities)) if harvest_qualities else None,
        "harvest_rate": len(timing_errors) / n_episodes,
        "action_distribution": {
            name: action_counts[i] / total_actions
            for i, name in enumerate(action_names)
        },
    }


def plot_training_rewards(
    episode_rewards: list[float],
    output_dir: Path,
    window: int = 50,
) -> None:
    """Plot episode rewards over training."""
    if len(episode_rewards) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episode_rewards, alpha=0.3, color="blue", label="Raw")

    # Smoothed
    if len(episode_rewards) >= window:
        smoothed = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode="valid",
        )
        ax.plot(
            range(window - 1, len(episode_rewards)),
            smoothed,
            color="red",
            linewidth=2,
            label=f"Smoothed (window={window})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Training — Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "training_rewards.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved reward plot to {path}")


def plot_comparison(results: list[dict], output_dir: Path) -> None:
    """Plot comparison between trained policy and baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [r["policy"] for r in results]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"][:len(results)]

    # Reward comparison
    rewards = [r["mean_reward"] for r in results]
    reward_stds = [r["std_reward"] for r in results]
    axes[0].bar(names, rewards, yerr=reward_stds, color=colors, alpha=0.8, capsize=5)
    axes[0].set_ylabel("Mean Episode Reward")
    axes[0].set_title("Reward Comparison")
    axes[0].tick_params(axis="x", rotation=20)

    # Timing error
    timing = [r.get("mean_timing_error") or 0 for r in results]
    axes[1].bar(names, timing, color=colors, alpha=0.8)
    axes[1].set_ylabel("Mean Timing Error (days)")
    axes[1].set_title("Harvest Timing Accuracy")
    axes[1].tick_params(axis="x", rotation=20)

    # Harvest quality
    quality = [r.get("mean_quality") or 0 for r in results]
    axes[2].bar(names, quality, color=colors, alpha=0.8)
    axes[2].set_ylabel("Mean Quality Score")
    axes[2].set_title("Harvest Quality")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    path = output_dir / "policy_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN policy for tomato harvest timing")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    rl_cfg = config.get("rl", {})
    train_cfg = rl_cfg.get("training", {})
    seed = config.get("project", {}).get("seed", 42)

    total_timesteps = args.total_timesteps or train_cfg.get("total_timesteps", 500_000)
    if args.smoke_test:
        total_timesteps = 1000
        print("=== SMOKE TEST MODE ===")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("project", {}).get("output_dir", "outputs")) / f"rl_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = output_dir / "tensorboard"

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create environments
    print("--- Creating Environments ---")
    n_envs = train_cfg.get("n_envs", 4)
    if args.smoke_test:
        n_envs = 1

    def make_env(rank: int):
        def _init():
            return TomatoRipeningEnv(config=config, seed=seed + rank)
        return _init

    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Eval environment
    eval_env = TomatoRipeningEnv(config=config, seed=seed + 1000)

    # Create DQN model
    print("\n--- Creating DQN Model ---")
    policy_cfg = rl_cfg.get("policy", {}).get("teacher", {})
    hidden_sizes = policy_cfg.get("hidden_sizes", [256, 256])

    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        buffer_size=train_cfg.get("buffer_size", 100_000),
        batch_size=train_cfg.get("batch_size", 256),
        gamma=train_cfg.get("gamma", 0.99),
        tau=train_cfg.get("tau", 0.005),
        exploration_fraction=0.7,
        exploration_final_eps=0.05,
        target_update_interval=5000,
        learning_starts=10000,
        policy_kwargs={"net_arch": hidden_sizes},
        verbose=1,
        seed=seed,
        device="auto",
        tensorboard_log=str(tb_log_dir),
    )

    print(f"  Policy arch: {hidden_sizes}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  TensorBoard log: {tb_log_dir}")

    # Callbacks
    metrics_cb = MetricsCallback()
    eval_freq = train_cfg.get("eval_freq", 10_000) // n_envs
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=train_cfg.get("eval_episodes", 20),
        eval_freq=eval_freq,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        deterministic=True,
    )

    # Train
    print(f"\n--- Training DQN for {total_timesteps:,} steps ---")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_cb, eval_cb],
        progress_bar=True,
    )

    # Save final model
    model.save(str(output_dir / "final_model"))
    print(f"\nSaved final model to {output_dir}/final_model")

    # Plot training rewards
    plot_training_rewards(metrics_cb.episode_rewards, output_dir)

    # Evaluate trained policy
    print("\n--- Evaluating Policies ---")
    n_eval = 100 if not args.smoke_test else 10
    results = []

    sac_results = evaluate_trained_policy(model, config, n_episodes=n_eval, seed=seed + 2000)
    results.append(sac_results)
    print(f"\nDQN Policy:")
    print(f"  Mean reward:      {sac_results['mean_reward']:.2f} ± {sac_results['std_reward']:.2f}")
    print(f"  Mean timing err:  {sac_results.get('mean_timing_error', 'N/A')}")
    print(f"  Mean quality:     {sac_results.get('mean_quality', 'N/A')}")
    print(f"  Harvest rate:     {sac_results['harvest_rate']:.2%}")
    if "action_distribution" in sac_results:
        print(f"  Action dist:      {sac_results['action_distribution']}")

    # Baseline comparisons
    for baseline in ["fixed_stage5", "fixed_day", "random"]:
        baseline_results = evaluate_baseline(config, policy_type=baseline, n_episodes=n_eval, seed=seed + 3000)
        results.append(baseline_results)
        print(f"\n{baseline} baseline:")
        print(f"  Mean reward:      {baseline_results['mean_reward']:.2f} ± {baseline_results['std_reward']:.2f}")
        print(f"  Mean timing err:  {baseline_results.get('mean_timing_error', 'N/A')}")
        print(f"  Mean quality:     {baseline_results.get('mean_quality', 'N/A')}")
        print(f"  Harvest rate:     {baseline_results['harvest_rate']:.2%}")

    # Save results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot comparison
    plot_comparison(results, output_dir)

    print(f"\n✅ RL training complete. All artifacts saved to {output_dir}/")
    print(f"   View TensorBoard: tensorboard --logdir {tb_log_dir}")


if __name__ == "__main__":
    main()
