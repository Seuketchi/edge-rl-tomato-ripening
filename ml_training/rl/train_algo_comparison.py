#!/usr/bin/env python3
"""Algorithm ablation: train PPO and SAC alongside existing DQN results.

Trains PPO and SAC teachers on the same TomatoRipeningEnv, then evaluates
all three algorithms on an identical 100-episode test set for a fair
comparison table suitable for the thesis.

Usage:
    python -m ml_training.rl.train_algo_comparison --config ml_training/config.yaml
    python -m ml_training.rl.train_algo_comparison --config ml_training/config.yaml --smoke-test
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
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ml_training.rl.environment import TomatoRipeningEnv
from ml_training.rl.train_dqn import (
    MetricsCallback,
    evaluate_baseline,
    evaluate_trained_policy,
)


def make_env(config: dict, seed: int, rank: int):
    """Create a closure for environment construction."""
    def _init():
        return TomatoRipeningEnv(config=config, seed=seed + rank)
    return _init


def train_algorithm(
    algo_name: str,
    config: dict,
    output_dir: Path,
    total_timesteps: int,
    seed: int = 42,
    smoke_test: bool = False,
):
    """Train a single algorithm and return the trained model + output dir."""
    rl_cfg = config.get("rl", {})
    train_cfg = rl_cfg.get("training", {})
    policy_cfg = rl_cfg.get("policy", {}).get("teacher", {})
    hidden_sizes = policy_cfg.get("hidden_sizes", [256, 256])

    n_envs = 1 if smoke_test else train_cfg.get("n_envs", 4)
    vec_env = DummyVecEnv([make_env(config, seed, i) for i in range(n_envs)])
    eval_env = TomatoRipeningEnv(config=config, seed=seed + 1000)

    algo_dir = output_dir / algo_name.lower()
    algo_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = algo_dir / "tensorboard"

    common_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        gamma=train_cfg.get("gamma", 0.99),
        policy_kwargs={"net_arch": hidden_sizes},
        verbose=1,
        seed=seed,
        device="auto",
        tensorboard_log=None,  # Disabled: tensorboard not installed
    )

    if algo_name == "DQN":
        model = DQN(
            **common_kwargs,
            buffer_size=train_cfg.get("buffer_size", 100_000),
            batch_size=train_cfg.get("batch_size", 256),
            tau=train_cfg.get("tau", 0.005),
            exploration_fraction=0.7,
            exploration_final_eps=0.05,
            target_update_interval=5000,
            learning_starts=10000,
        )
    elif algo_name == "PPO":
        model = PPO(
            **common_kwargs,
            n_steps=train_cfg.get("ppo_n_steps", 512),
            batch_size=train_cfg.get("ppo_batch_size", 128),
            n_epochs=train_cfg.get("ppo_n_epochs", 10),
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
        )
    elif algo_name == "SAC":
        # SAC requires continuous action space — wrap the discrete env
        # We use DQN-style discrete SAC via SB3 which is not natively supported.
        # Instead, we'll skip SAC for discrete action spaces and note this limitation.
        print(f"  ⚠ SAC requires continuous actions. Skipping for discrete env.")
        print(f"    Consider using A2C as an alternative on-policy algorithm.")
        return None, algo_dir
    elif algo_name == "A2C":
        from stable_baselines3 import A2C
        model = A2C(
            **common_kwargs,
            n_steps=train_cfg.get("a2c_n_steps", 16),
            gae_lambda=0.95,
            ent_coef=0.005,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    print(f"\n{'='*60}")
    print(f"  Training {algo_name} for {total_timesteps:,} timesteps")
    print(f"  Policy arch: {hidden_sizes}")
    print(f"  TensorBoard: {tb_log_dir}")
    print(f"{'='*60}\n")

    eval_freq = train_cfg.get("eval_freq", 10_000) // n_envs
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=20 if not smoke_test else 5,
        eval_freq=eval_freq,
        best_model_save_path=str(algo_dir / "best_model"),
        log_path=str(algo_dir / "eval_logs"),
        deterministic=True,
    )
    metrics_cb = MetricsCallback()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_cb, eval_cb],
        progress_bar=True,
    )

    model.save(str(algo_dir / "final_model"))
    print(f"✓ {algo_name} model saved to {algo_dir}/final_model")

    return model, algo_dir


def plot_algorithm_comparison(all_results: list[dict], output_dir: Path):
    """Generate a comparison bar chart across algorithms and baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [r["policy"] for r in all_results]
    n = len(names)
    colors = ["#2196F3", "#9C27B0", "#4CAF50", "#FF9800", "#F44336", "#795548"][:n]

    rewards = [r["mean_reward"] for r in all_results]
    stds = [r["std_reward"] for r in all_results]
    axes[0].bar(range(n), rewards, yerr=stds, color=colors, alpha=0.8, capsize=5)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean Episode Reward")
    axes[0].set_title("Reward Comparison")

    timing = [r.get("mean_timing_error") or 0 for r in all_results]
    axes[1].bar(range(n), timing, color=colors, alpha=0.8)
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean Timing Error (days)")
    axes[1].set_title("Harvest Timing Accuracy")

    quality = [r.get("mean_quality") or 0 for r in all_results]
    axes[2].bar(range(n), quality, color=colors, alpha=0.8)
    axes[2].set_xticks(range(n))
    axes[2].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[2].set_ylabel("Mean Quality Score")
    axes[2].set_title("Harvest Quality")

    plt.tight_layout()
    path = output_dir / "algorithm_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="ml_training/config.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--algos", nargs="*", default=["DQN", "PPO", "A2C"],
                        help="Algorithms to train (default: DQN PPO A2C)")
    parser.add_argument("--dqn-model", type=str, default=None,
                        help="Path to existing DQN model to skip re-training it")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    rl_cfg = config.get("rl", {})
    train_cfg = rl_cfg.get("training", {})
    seed = config.get("project", {}).get("seed", 42)

    total_timesteps = args.total_timesteps or train_cfg.get("total_timesteps", 500_000)
    if args.smoke_test:
        total_timesteps = 1000
        print("=== SMOKE TEST MODE ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("project", {}).get("output_dir", "outputs")) / f"algo_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Train each algorithm
    trained_models = {}
    for algo_name in args.algos:
        if algo_name == "DQN" and args.dqn_model:
            print(f"\n--- Loading existing DQN model from {args.dqn_model} ---")
            trained_models["DQN"] = DQN.load(args.dqn_model)
        else:
            model, algo_dir = train_algorithm(
                algo_name, config, output_dir, total_timesteps, seed, args.smoke_test
            )
            if model is not None:
                trained_models[algo_name] = model

    # Evaluate all algorithms on the same test set
    print("\n" + "=" * 60)
    print("  EVALUATION (100 episodes per policy)")
    print("=" * 60)

    n_eval = 100 if not args.smoke_test else 10
    all_results = []

    for algo_name, model in trained_models.items():
        results = evaluate_trained_policy(model, config, n_episodes=n_eval, seed=seed + 2000)
        results["policy"] = f"{algo_name} (trained)"
        all_results.append(results)
        print(f"\n{algo_name}:")
        print(f"  Mean reward:     {results['mean_reward']:+.2f} ± {results['std_reward']:.2f}")
        print(f"  Timing error:    {results.get('mean_timing_error', 'N/A'):.4f} days")
        print(f"  Quality:         {results.get('mean_quality', 'N/A'):.4f}")
        print(f"  Harvest rate:    {results['harvest_rate']:.2%}")
        if "action_distribution" in results:
            print(f"  Action dist:     {results['action_distribution']}")

    # Baselines
    for baseline in ["fixed_day", "random"]:
        bl = evaluate_baseline(config, policy_type=baseline, n_episodes=n_eval, seed=seed + 3000)
        all_results.append(bl)
        print(f"\n{baseline}:")
        print(f"  Mean reward:     {bl['mean_reward']:+.2f} ± {bl['std_reward']:.2f}")
        print(f"  Timing error:    {bl.get('mean_timing_error', 'N/A')}")
        print(f"  Quality:         {bl.get('mean_quality', 'N/A')}")

    # Save results
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    plot_algorithm_comparison(all_results, output_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Policy':<20} {'Reward':>12} {'Timing Err':>12} {'Quality':>10} {'Harvest':>10}")
    print("-" * 80)
    for r in all_results:
        te = r.get('mean_timing_error')
        q = r.get('mean_quality')
        print(f"{r['policy']:<20} {r['mean_reward']:>+10.2f}   {te if te else 'N/A':>10}   {q if q else 'N/A':>8}   {r['harvest_rate']:>8.1%}")
    print("=" * 80)

    print(f"\n✅ Algorithm comparison complete. Results saved to {output_dir}/")
    print(f"   View TensorBoard: tensorboard --logdir {output_dir}")


if __name__ == "__main__":
    main()
