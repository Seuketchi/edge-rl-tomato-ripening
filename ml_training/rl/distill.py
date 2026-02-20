"""Policy distillation and export for edge deployment.

Distills the trained DQN teacher policy into a compact student MLP,
then exports to ONNX and optionally to ESP-DL format.

Pipeline:
    1. Load trained DQN model (teacher)
    2. Generate rollout dataset using teacher policy
    3. Train student MLP via knowledge distillation
    4. Export student → ONNX → ESP-DL INT8

Usage:
    python -m ml_training.rl.distill --config ml_training/config.yaml --teacher outputs/rl_*/final_model.zip
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from stable_baselines3 import DQN

from ml_training.rl.environment import TomatoRipeningEnv


class StudentPolicy(nn.Module):
    """Compact MLP policy for edge deployment.

    Simple feedforward network: state → action logits.

    Args:
        state_dim: Input state dimension.
        action_dim: Number of discrete actions.
        hidden_sizes: List of hidden layer sizes.
    """

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 4,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        layers = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State tensor of shape (batch, state_dim).

        Returns:
            Action logits of shape (batch, action_dim).
        """
        return self.network(x)

    def predict(self, state: np.ndarray) -> int:
        """Predict action from numpy state (for evaluation).

        Args:
            state: State array of shape (state_dim,).

        Returns:
            Argmax action index.
        """
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            logits = self.forward(x)
            return int(logits.argmax(dim=-1).item())


def generate_teacher_rollouts(
    teacher: DQN,
    config: dict,
    num_samples: int = 100_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (state, action) pairs from teacher policy.

    Args:
        teacher: Trained DQN model.
        config: Full config dict.
        num_samples: Number of (state, action) pairs to collect.
        seed: Random seed.

    Returns:
        Tuple of (states, actions) numpy arrays.
    """
    states = []
    actions = []
    episode = 0

    while len(states) < num_samples:
        env = TomatoRipeningEnv(config=config, seed=seed + episode)
        obs, _ = env.reset()
        done = False

        while not done and len(states) < num_samples:
            action, _ = teacher.predict(obs, deterministic=True)
            action = int(action)

            states.append(obs.copy())
            actions.append(action)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        episode += 1

    print(f"Collected {len(states)} samples from {episode} episodes")
    return np.array(states), np.array(actions)


def distill(
    teacher: DQN,
    config: dict,
    output_dir: Path,
) -> StudentPolicy:
    """Run knowledge distillation from teacher to student.

    Args:
        teacher: Trained DQN model.
        config: Full config dict.
        output_dir: Directory to save artifacts.

    Returns:
        Trained student policy.
    """
    rl_cfg = config.get("rl", {})
    distill_cfg = rl_cfg.get("distillation", {})
    student_cfg = rl_cfg.get("policy", {}).get("student", {})
    seed = config.get("project", {}).get("seed", 42)

    # Generate rollout data
    print("\n--- Generating Teacher Rollouts ---")
    num_samples = distill_cfg.get("num_samples", 100_000)
    states, actions = generate_teacher_rollouts(
        teacher, config, num_samples=num_samples, seed=seed,
    )

    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)

    # Create dataset
    dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor)
    batch_size = distill_cfg.get("batch_size", 512)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )

    # Create student
    state_dim = states.shape[1]
    action_dim = 3  # discrete actions: maintain, heat, cool
    hidden_sizes = student_cfg.get("hidden_sizes", [64, 64])

    student = StudentPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )
    total_params = sum(p.numel() for p in student.parameters())
    print(f"\nStudent model:")
    print(f"  Architecture: {[state_dim]} → {hidden_sizes} → [{action_dim}]")
    print(f"  Parameters: {total_params:,}")
    print(f"  FP32 size: {total_params * 4 / 1024:.1f} KB")
    print(f"  INT8 size: {total_params / 1024:.1f} KB (estimated)")

    # Training
    lr = distill_cfg.get("learning_rate", 0.001)
    epochs = distill_cfg.get("epochs", 100)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Distilling for {epochs} epochs ---")
    losses = []
    accuracies = []

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        student.train()
        for batch_states, batch_actions in loader:
            optimizer.zero_grad()
            logits = student(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_states.size(0)
            _, predicted = logits.max(1)
            total += batch_actions.size(0)
            correct += predicted.eq(batch_actions).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

    # Save training history for reproducible figures
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({"loss": losses, "accuracy": accuracies}, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Save student
    student_path = output_dir / "student_policy.pth"
    torch.save({
        "model_state_dict": student.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_sizes": hidden_sizes,
        "final_accuracy": accuracies[-1],
        "total_params": total_params,
    }, student_path)
    print(f"\nSaved student model to {student_path}")

    # Plot distillation curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(losses, linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Distillation Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(accuracies, linewidth=2, color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Action Prediction Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "distillation_curves.png", dpi=150)
    plt.close(fig)

    return student


def export_student_onnx(
    student: StudentPolicy,
    state_dim: int,
    onnx_path: Path,
) -> None:
    """Export student policy to ONNX."""
    student.eval()
    dummy_input = torch.randn(1, state_dim)

    torch.onnx.export(
        student,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["state"],
        output_names=["action_logits"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
    )
    print(f"✓ Student ONNX exported to {onnx_path}")
    print(f"  File size: {onnx_path.stat().st_size / 1024:.1f} KB")


def evaluate_student(
    student: StudentPolicy,
    config: dict,
    n_episodes: int = 100,
    seed: int = 42,
) -> dict:
    """Evaluate the distilled student policy in the environment."""
    rewards = []
    timing_errors = []
    harvest_qualities = []

    for i in range(n_episodes):
        env = TomatoRipeningEnv(config=config, seed=seed + i)
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = student.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        if "timing_error" in info:
            timing_errors.append(info["timing_error"])
        if "harvest_quality" in info:
            harvest_qualities.append(info["harvest_quality"])

    return {
        "policy": "Student (distilled)",
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_timing_error": float(np.mean(timing_errors)) if timing_errors else None,
        "mean_quality": float(np.mean(harvest_qualities)) if harvest_qualities else None,
        "harvest_rate": len(timing_errors) / n_episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill DQN policy to compact student")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml")
    parser.add_argument("--teacher", type=str, required=True, help="Path to trained DQN model .zip")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-espdl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or config.get("project", {}).get("output_dir", "outputs")) / f"distill_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load teacher
    print("--- Loading Teacher ---")
    teacher = DQN.load(args.teacher)
    print(f"Loaded DQN model from {args.teacher}")

    # Distill
    print("\n--- Knowledge Distillation ---")
    student = distill(teacher, config, output_dir)

    # Export to ONNX
    print("\n--- Exporting to ONNX ---")
    state_dim = 9  # from environment
    onnx_path = output_dir / "rl_policy.onnx"
    try:
        export_student_onnx(student, state_dim, onnx_path)
    except Exception as e:
        print(f"⚠️ ONNX export failed (skipping): {e}")

    # Evaluate student
    print("\n--- Evaluating Student vs Teacher ---")
    seed = config.get("project", {}).get("seed", 42)

    student_results = evaluate_student(student, config, n_episodes=100, seed=seed + 5000)
    print(f"\nStudent policy:")
    print(f"  Mean reward:      {student_results['mean_reward']:.2f} ± {student_results['std_reward']:.2f}")
    print(f"  Mean timing err:  {student_results.get('mean_timing_error', 'N/A')}")
    print(f"  Mean quality:     {student_results.get('mean_quality', 'N/A')}")
    print(f"  Harvest rate:     {student_results['harvest_rate']:.2%}")

    # Save results
    with open(output_dir / "distillation_results.json", "w") as f:
        json.dump(student_results, f, indent=2)

    print(f"\n✅ Distillation complete. Artifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()
