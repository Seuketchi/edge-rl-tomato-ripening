"""Gymnasium environment for tomato ripening RL training.

Wraps the TomatoRipeningSimulator with a standard Gymnasium interface
compatible with Stable Baselines3.

State space (9 dimensions):
    [ripeness, temperature, humidity, days_elapsed, target_day,
     ripeness_rate, temp_deviation, days_remaining, is_near_target]

Action space (Discrete, 4 actions):
    0 = maintain temperature
    1 = heat (+1°C/hour)
    2 = cool (-1°C/hour)
    3 = harvest (terminal)

Reward function:
    - Quality reward at harvest (based on ripeness vs target)
    - Timing reward (harvesting near target day)
    - Energy penalty (for heating/cooling actions)
    - Spoilage penalty (overripening)
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from ml_training.rl.simulator import SimulatorConfig, TomatoRipeningSimulator


class TomatoRipeningEnv(gym.Env):
    """Gymnasium environment for tomato harvest timing optimization.

    The agent observes the ripening state and must decide when to harvest
    tomatoes to maximize quality at a target date.

    Args:
        config: Dict with 'rl.environment' and 'rl.simulator' keys.
        seed: Random seed.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: dict | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        # Parse config
        if config is None:
            config = {}
        rl_cfg = config.get("rl", {})
        env_cfg = rl_cfg.get("environment", {})
        sim_cfg = rl_cfg.get("simulator", {})

        # Environment parameters
        self.steps_per_day = env_cfg.get("steps_per_day", 24)
        self.max_episode_days = env_cfg.get("episode_days", 7)
        self.max_steps = self.steps_per_day * self.max_episode_days
        self.target_day_range = tuple(env_cfg.get("target_day_range", [3, 7]))

        # Create simulator
        self.rng = np.random.default_rng(seed)
        sim_config = SimulatorConfig(
            k1=sim_cfg.get("k1", 0.08),
            t_base=sim_cfg.get("t_base", 12.5),
            r_max=sim_cfg.get("r_max", 5.0),
            temp_noise_std=sim_cfg.get("temp_noise_std", 0.5),
            humidity_noise_std=sim_cfg.get("humidity_noise_std", 2.0),
        )
        self.simulator = TomatoRipeningSimulator(config=sim_config, rng=self.rng)

        # Spaces
        self.action_space = gym.spaces.Discrete(4)

        # Observation: 9-dimensional state vector
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 10, 40, 0, 0, -1, -15, -10, 0], dtype=np.float32),
            high=np.array([6, 30, 100, 15, 15, 1, 15, 15, 1], dtype=np.float32),
        )

        # Episode state
        self.target_day: float = 5.0
        self.current_step: int = 0
        self.prev_ripeness: float = 0.0
        self.harvested: bool = False

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment for a new episode.

        Returns:
            Tuple of (observation, info dict).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.simulator.rng = self.rng

        state = self.simulator.reset()
        self.target_day = self.rng.uniform(*self.target_day_range)
        self.current_step = 0
        self.prev_ripeness = state["ripeness"]
        self.harvested = False

        obs = self._make_observation(state)
        info = {"target_day": self.target_day}

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step.

        Args:
            action: 0=maintain, 1=heat, 2=cool, 3=harvest

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        self.current_step += 1
        dt_hours = 24.0 / self.steps_per_day

        # Step simulator
        state = self.simulator.step(action, dt_hours)

        # Compute reward
        reward = self._compute_reward(action, state)

        # Check termination
        terminated = False
        truncated = False
        info = {
            "target_day": self.target_day,
            "ripeness": state["ripeness"],
            "temperature": state["_true_temperature"],
            "action": action,
        }

        if action == 3:  # harvest
            terminated = True
            self.harvested = True
            info["harvest_day"] = state["days_elapsed"]
            info["harvest_quality"] = self.simulator.compute_quality_score()
            info["timing_error"] = abs(state["days_elapsed"] - self.target_day)
        elif self.simulator.is_overripe():
            terminated = True
            info["spoiled"] = True
        elif self.current_step >= self.max_steps:
            truncated = True
            info["timeout"] = True

        self.prev_ripeness = state["ripeness"]
        obs = self._make_observation(state)

        return obs, reward, terminated, truncated, info

    def _make_observation(self, state: dict) -> np.ndarray:
        """Convert simulator state to observation vector.

        Observation features (9D):
            0: ripeness (0-5)
            1: temperature (°C, noisy)
            2: humidity (%, noisy)
            3: days_elapsed
            4: target_day
            5: ripeness_rate (change since last step, normalized)
            6: temperature deviation from 20°C midpoint
            7: days_remaining until target
            8: is_near_target (1 if within 0.5 days of target)
        """
        ripeness_rate = (state["ripeness"] - self.prev_ripeness)
        days_remaining = self.target_day - state["days_elapsed"]
        is_near_target = 1.0 if abs(days_remaining) < 0.5 else 0.0

        obs = np.array([
            state["ripeness"],
            state["temperature"],
            state["humidity"],
            state["days_elapsed"],
            self.target_day,
            np.clip(ripeness_rate, -1, 1),
            state["temperature"] - 20.0,
            days_remaining,
            is_near_target,
        ], dtype=np.float32)

        return obs

    def _compute_reward(self, action: int, state: dict) -> float:
        """Compute step reward.

        Reward components:
            1. Harvest quality reward (at harvest only)
            2. Timing reward (at harvest only)
            3. Energy penalty (for heating/cooling)
            4. Spoilage penalty (if overripe)
            5. Progress shaping (small reward for controlled ripening)
        """
        reward = 0.0

        if action == 3:  # harvest
            # Quality: how close to target ripeness (stage 4 = light red)
            quality = self.simulator.compute_quality_score(target_ripeness=4.0)
            reward += quality * 15.0

            # Timing: how close to target day
            timing_error = abs(state["days_elapsed"] - self.target_day)
            
            # Penalize harvesting too early (> 1 day before target)
            if (self.target_day - state["days_elapsed"]) > 1.0:
                 reward -= 10.0
            
            if timing_error < 0.5:
                reward += 10.0      # Perfect timing bonus (boosted)
            elif timing_error < 1.0:
                reward += 3.0       # Good timing
            elif timing_error < 2.0:
                reward += 1.0       # Acceptable
            else:
                reward -= timing_error * 1.0  # Penalty for bad timing

            # Bonus for harvesting at good ripeness level
            ripeness = state["ripeness"]
            if 3.5 <= ripeness <= 4.5:
                reward += 5.0       # Optimal harvest window (stronger)
            elif 3.0 <= ripeness <= 5.0:
                reward += 1.0       # Acceptable

        else:
            # Energy penalty for active temperature control (reduced)
            if action in (1, 2):  # heat or cool
                reward -= 0.02

            # Small ripening progress reward (scaled down to not dilute harvest signal)
            ripeness_change = state["ripeness"] - self.prev_ripeness
            if ripeness_change > 0 and state["ripeness"] < 5.0:
                reward += ripeness_change * 0.02

            # Spoilage penalty
            if self.simulator.is_overripe():
                reward -= 10.0

        # Timeout penalty: agent must learn to harvest
        if self.current_step >= self.max_steps and not self.harvested:
            reward -= 5.0

        return reward

    def render(self) -> None:
        """Print current state to console."""
        state = self.simulator.get_state()
        stage = self.simulator.get_ripeness_stage()
        stage_names = ["Green", "Breaker", "Turning", "Pink", "Light Red", "Red"]
        stage_name = stage_names[min(stage, 5)]

        print(
            f"Day {state['days_elapsed']:.1f}/{self.target_day:.1f} | "
            f"Ripeness: {state['ripeness']:.2f} ({stage_name}) | "
            f"Temp: {state['temperature']:.1f}°C | "
            f"Humidity: {state['humidity']:.1f}%"
        )
