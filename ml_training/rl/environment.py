"""Gymnasium environment for tomato ripening RL training.

Wraps the TomatoRipeningSimulator with a standard Gymnasium interface
compatible with Stable Baselines3.

Key design decisions (per advisor feedback):
    - Chromatic index X follows ROYG spectral mapping (1=Green, 0=Red)
    - ODE: dX/dt = -k₁(T - T_base) X   (X decays toward 0 as fruit ripens)
    - Temperature actions are INCREMENTAL (±1°C setpoint changes)
    - Harvest action REMOVED — harvest is automatic post-processing
    - Reward uses auxiliary function f(dX/dt, t_rem) instead of r_eff
    - Safety penalty is PROGRESSIVE (not cliff)
    - Three ablation state-space variants: A (scalar), B (+RGB stats), C (+max-pool)

State space variants:
    Option A (7D):  [X, Ẋ, X_ref, T, H, t_e, t_rem]
    Option B (16D): Option A + [C_μ(3), C_σ(3), C_mode(3)]
    Option C (20D): Option B + [max_pool(4)]

Action space (Discrete, 3 actions):
    0 = Maintain (no setpoint change)
    1 = Heat (+ΔT °C incremental)
    2 = Cool (-ΔT °C incremental)

Reward function:
    r_t = r_track + r_progress + c_safety  (+ terminal harvest bonus)
    - r_track:    Rate-tracking: -λ|dX/dt_daily - desired_rate|
    - r_progress: Progress:      β × (prev_X - X)
    - c_safety:   Progressive penalty for consecutive thermal violations (capped)
    - bonus:      +harvest_bonus × timing_frac on auto-harvest termination
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ml_training.rl.simulator import SimulatorConfig, TomatoRipeningSimulator

# State-space variant dimensions
STATE_DIMS = {"A": 7, "B": 16, "C": 20}


class TomatoRipeningEnv(gym.Env):
    """Gymnasium environment for tomato harvest timing optimization.

    The agent controls chamber temperature via incremental setpoint
    adjustments to optimise the ripening trajectory toward a target
    harvest date.  Harvest is triggered automatically when X reaches
    the ripe threshold or when remaining time expires (post-processing).

    Args:
        config: Dict with 'rl.environment' and 'rl.simulator' keys.
        seed: Random seed.
        state_variant: State-space ablation variant ("A", "B", or "C").
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: dict | None = None,
        seed: int = 42,
        state_variant: str = "B",
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

        # Reward parameters
        reward_cfg = env_cfg.get("reward", {})
        self.lambda_rate = reward_cfg.get("lambda_rate", 0.5)
        self.safety_alpha = reward_cfg.get("safety_alpha", 2.0)
        self.safety_cap = reward_cfg.get("safety_cap", 5)
        self.ripe_threshold = reward_cfg.get("ripe_threshold", 0.15)
        self.t_rem_epsilon = reward_cfg.get("t_rem_epsilon", 0.1)
        self.harvest_bonus = reward_cfg.get("harvest_bonus", 10.0)
        self.progress_weight = reward_cfg.get("progress_weight", 2.0)

        # State variant
        self.state_variant = state_variant.upper()
        if self.state_variant not in STATE_DIMS:
            raise ValueError(f"state_variant must be one of {list(STATE_DIMS)}")
        self.state_dim = STATE_DIMS[self.state_variant]

        # Max-pool output dimension (only used in variant C)
        self.max_pool_dim = env_cfg.get("max_pool_dim", 4)

        # Create simulator
        self.rng = np.random.default_rng(seed)
        sim_config = SimulatorConfig(
            k1=sim_cfg.get("k1", 0.02),
            t_base=sim_cfg.get("t_base", 12.5),
            x_min=sim_cfg.get("x_min", 0.0),
            temp_noise_std=sim_cfg.get("temp_noise_std", 0.5),
            humidity_noise_std=sim_cfg.get("humidity_noise_std", 2.0),
            delta_t_step=sim_cfg.get("delta_t_step", 1.0),
        )
        self.simulator = TomatoRipeningSimulator(config=sim_config, rng=self.rng)

        # Action space: 3 discrete actions (maintain, heat, cool)
        self.action_space = gym.spaces.Discrete(3)

        # Observation space (depends on variant)
        self.observation_space = self._build_obs_space()

        # Episode state
        self.target_day: float = 5.0
        self.current_step: int = 0
        self.prev_ripeness: float = 1.0
        self.consecutive_safety_violations: int = 0

    def _build_obs_space(self) -> gym.spaces.Box:
        """Construct observation space bounds for the chosen variant."""
        if self.state_variant == "A":
            # [X, dX/dt, X_ref, T, H, t_e, t_rem]  — 7D
            low = np.array([0, -1, 0, 10, 0, 0, -30], dtype=np.float32)
            high = np.array([1, 1, 1, 40, 100, 30, 30], dtype=np.float32)
        elif self.state_variant == "B":
            # [X, dX/dt, X_ref, C_mu(3), C_sig(3), C_mode(3), T, H, t_e, t_rem] — 16D
            low = np.array(
                [0, -1, 0] + [0]*3 + [0]*3 + [0]*3 + [10, 0, 0, -30],
                dtype=np.float32,
            )
            high = np.array(
                [1, 1, 1] + [1]*3 + [1]*3 + [1]*3 + [40, 100, 30, 30],
                dtype=np.float32,
            )
        else:  # "C"
            # Option B (16D) + max_pool(4) = 20D
            low = np.array(
                [0, -1, 0] + [0]*3 + [0]*3 + [0]*3 + [10, 0, 0, -30] + [-1]*self.max_pool_dim,
                dtype=np.float32,
            )
            high = np.array(
                [1, 1, 1] + [1]*3 + [1]*3 + [1]*3 + [40, 100, 30, 30] + [1]*self.max_pool_dim,
                dtype=np.float32,
            )
        return gym.spaces.Box(low=low, high=high)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.current_step = 0
        self.consecutive_safety_violations = 0

        # Reset simulator
        raw_state = self.simulator.reset()
        self.target_day = self.rng.uniform(*self.target_day_range)
        self.prev_ripeness = raw_state["ripeness"]

        return self._make_observation(raw_state), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step: apply action, compute reward, check termination.

        Harvest is NOT an action — it is automatic post-processing:
            - Episode terminates when X ≤ ripe_threshold (auto-harvest)
            - Episode terminates when t_rem ≤ 0 (deadline reached)
            - Episode truncates at max_steps

        Args:
            action: 0=maintain, 1=heat, 2=cool.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Simulator step
        raw_state = self.simulator.step(action, dt_hours=1.0)
        self.current_step += 1

        obs = self._make_observation(raw_state)
        reward = self._compute_reward(raw_state, action, obs)

        # Check termination
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        x = raw_state["ripeness"]
        t_e = raw_state["days_elapsed"]
        t_rem = self.target_day - t_e

        # Auto-harvest: X reached ripe threshold
        if x <= self.ripe_threshold:
            terminated = True
            info["auto_harvest"] = True
            info["harvest_quality"] = self.simulator.compute_quality_score()
            info["timing_error"] = abs(t_rem)
            # Terminal bonus: scaled by timing accuracy (on-time → full bonus)
            timing_frac = 1.0 - min(abs(t_rem) / self.max_episode_days, 1.0)
            reward += self.harvest_bonus * timing_frac

        # Deadline reached (t_rem ≤ 0): harvest whatever we have
        elif t_rem <= 0:
            terminated = True
            info["deadline_harvest"] = True
            info["harvest_quality"] = self.simulator.compute_quality_score()
            info["timing_error"] = 0.0
            # Partial bonus based on how ripe it got (X close to 0 = good)
            ripeness_frac = 1.0 - min(x / 0.5, 1.0)
            reward += self.harvest_bonus * 0.5 * ripeness_frac

        # Truncation (safety net)
        if self.current_step >= self.max_steps:
            truncated = True
            if "harvest_quality" not in info:
                info["harvest_quality"] = self.simulator.compute_quality_score()
                info["timing_error"] = abs(t_rem)

        return obs, reward, terminated, truncated, info

    def _make_observation(self, state: dict) -> np.ndarray:
        """Build observation vector for the active state variant.

        Variant A (7D):  [X, Ẋ, X_ref, T, H, t_e, t_rem]
        Variant B (16D): A + [C_μ(3), C_σ(3), C_mode(3)]
        Variant C (20D): B + [max_pool(4)]
        """
        # Continuous Chromatic Index (ROYG: 1=Green, 0=Red)
        x = state["ripeness"]

        # dX/dt (velocity) — finite difference
        dx_dt = x - self.prev_ripeness
        self.prev_ripeness = x

        # X_ref from analytical ODE at ideal temp
        x_ref = self.simulator.compute_x_ref(state["days_elapsed"])

        # Remaining time until target harvest
        t_e = state["days_elapsed"]
        t_rem = self.target_day - t_e

        # Scalars shared by all variants
        base = [
            x,                     # 0: X
            dx_dt,                 # 1: dX/dt
            x_ref,                 # 2: X_ref (time-varying reference signal)
        ]

        if self.state_variant == "A":
            obs_list = base + [
                state["temperature"],   # 3: T
                state["humidity"],      # 4: H
                t_e,                    # 5: t_e
                t_rem,                  # 6: t_rem
            ]
        else:
            # RGB stats for B and C
            rgb_mean = state["rgb"]
            rgb_std = np.array([0.1, 0.1, 0.05], dtype=np.float32)  # Mock
            rgb_mode = state.get("rgb_mode", rgb_mean)

            obs_list = base + [
                rgb_mean[0], rgb_mean[1], rgb_mean[2],       # 3-5:  C_μ
                rgb_std[0], rgb_std[1], rgb_std[2],          # 6-8:  C_σ
                rgb_mode[0], rgb_mode[1], rgb_mode[2],       # 9-11: C_mode
                state["temperature"],                         # 12: T
                state["humidity"],                            # 13: H
                t_e,                                          # 14: t_e
                t_rem,                                        # 15: t_rem
            ]

            if self.state_variant == "C":
                max_pool = state.get(
                    "max_pool",
                    np.zeros(self.max_pool_dim, dtype=np.float32),
                )
                obs_list.extend(max_pool.tolist())            # 16-19: max_pool

        return np.array(obs_list, dtype=np.float32)

    def _compute_reward(
        self,
        state: dict,
        action: int,
        obs: np.ndarray,
    ) -> float:
        """Compute the three-component reward signal.

        r_t = r_track + r_progress + c_safety

        1. r_track: Rate-tracking reward — rewards the agent for matching
           the desired ripening rate (penalises deviation from ideal).
               desired_rate = (X_target - X) / max(t_rem, ε)   [per day]
               dx_dt_daily  = dX/dt × steps_per_day             [per day]
               r_track = -λ × |dx_dt_daily - desired_rate|
           Properly unit-matched (both per-day).  Agent is rewarded
           when its control drives dX/dt close to the schedule.

        2. r_progress: Progress reward — positive signal for moving X
           toward the ripe threshold.  Computed as the change in X
           scaled by a weight factor.
               r_progress = β × (prev_X - X)
           This gives a gentle positive reward for every step of
           ripening, counteracting the "do nothing" pathology.

        3. c_safety: Progressive penalty for consecutive thermal boundary
           violations, with a per-step cap to prevent catastrophic blowup.
               c_safety = -α × min(n, n_cap)²
           where n is the number of consecutive violations.

        Terminal bonus (applied in step(), not here):
           +harvest_bonus when auto-harvest triggers (X ≤ threshold),
           scaled by (1 - timing_error / max_days) to reward on-time harvest.

        Args:
            state: Raw simulator state dict.
            action: Executed action (0/1/2).
            obs: Observation vector.

        Returns:
            Scalar reward.
        """
        temp = state["_true_temperature"]
        x = state["ripeness"]
        dx_dt_step = obs[1]  # dX/dt per step (hourly finite difference)

        t_rem = self.target_day - state["days_elapsed"]

        # --- 1. Rate-tracking reward (r_track) ---
        # Convert dX/dt from per-step to per-day to match desired_rate units
        dx_dt_daily = dx_dt_step * self.steps_per_day

        x_target = self.ripe_threshold
        t_rem_clamped = max(t_rem, self.t_rem_epsilon)
        desired_rate = (x_target - x) / t_rem_clamped  # per day (negative)

        r_track = -self.lambda_rate * abs(dx_dt_daily - desired_rate)

        # --- 2. Progress reward (r_progress) ---
        # Positive reward for ripening progress (prev_X > X means progress)
        delta_x = self.prev_ripeness - x  # Positive when ripening
        r_progress = self.progress_weight * delta_x

        # --- 3. Progressive safety penalty (capped) ---
        if temp > 35.0 or temp < 12.5:
            self.consecutive_safety_violations += 1
            n = min(self.consecutive_safety_violations, self.safety_cap)
            c_safety = -self.safety_alpha * (n ** 2)
        else:
            self.consecutive_safety_violations = 0
            c_safety = 0.0

        reward = r_track + r_progress + c_safety
        return reward

    def render(self) -> None:
        """Print current state to console."""
        state = self.simulator.get_state()
        x = state["ripeness"]

        # ROYG labels (high X = Green, low X = Red)
        if x > 0.8:
            label = "Green"
        elif x > 0.6:
            label = "Breaker"
        elif x > 0.4:
            label = "Turning"
        elif x > 0.2:
            label = "Pink"
        elif x > 0.1:
            label = "Light Red"
        else:
            label = "Red"

        rgb = state["rgb"]
        t_rem = self.target_day - state["days_elapsed"]
        print(
            f"Day {state['days_elapsed']:.1f} | "
            f"t_rem={t_rem:.1f}d | "
            f"X: {x:.3f} ({label}) | "
            f"RGB: [{rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f}] | "
            f"Temp: {state['temperature']:.1f}°C"
        )
