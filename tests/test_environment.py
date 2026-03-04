"""Pytest tests for TomatoRipeningEnv (Variant B, 16D state).

Covers concerns not already in test_continuous_refactor.py:
  - Observation shape contract
  - Action space validity (all 3 actions)
  - Reward finiteness over a complete episode
"""

import numpy as np
import pytest

from ml_training.rl.environment import TomatoRipeningEnv


@pytest.fixture
def env():
    return TomatoRipeningEnv(seed=0, state_variant="B")


def test_obs_shape_is_16d(env):
    """Reset must return a (16,) float32 observation for Variant B."""
    obs, _ = env.reset()
    assert obs.shape == (16,), f"Expected (16,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"


def test_all_three_actions_valid(env):
    """Each action (0=maintain, 1=heat, 2=cool) must return a valid step."""
    for action in range(3):
        obs, _ = env.reset()
        step_obs, reward, terminated, truncated, info = env.step(action)
        assert step_obs.shape == (16,), (
            f"Action {action}: expected obs shape (16,), got {step_obs.shape}"
        )
        assert np.isfinite(reward), (
            f"Action {action}: reward is not finite: {reward}"
        )


def test_reward_finite_over_full_episode(env):
    """Running to termination with action=0 (maintain) must never produce non-finite rewards."""
    env.reset()
    done = False
    step = 0
    while not done:
        _, reward, terminated, truncated, _ = env.step(0)
        assert np.isfinite(reward), f"Non-finite reward at step {step}: {reward}"
        done = terminated or truncated
        step += 1
