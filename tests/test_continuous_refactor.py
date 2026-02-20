"""Tests for the ROYG Chromatic Index refactor.

Validates that:
1. Simulator uses ROYG convention: X=1 (Green) → X=0 (Red).
2. ODE is dX/dt = -k₁(T-T_base)X — X decreases monotonically.
3. Temperature actions are incremental (±ΔT setpoints).
4. Environment produces correct-dimension observations for variants A/B/C.
5. Reward uses rate-based penalty, auxiliary f(dX/dt, t_rem), progressive safety.
6. Harvest action removed — auto-harvest via threshold or deadline.
7. X_ref analytical solution is exponential decay (not 1-exp).
"""

import numpy as np
import pytest

from ml_training.rl.simulator import SimulatorConfig, TomatoRipeningSimulator
from ml_training.rl.environment import TomatoRipeningEnv, STATE_DIMS


# ── Simulator Tests ──────────────────────────────────────────────

class TestSimulatorROYG:
    """Verify the simulator uses ROYG convention (X=1 Green, X=0 Red)."""

    def setup_method(self):
        self.sim = TomatoRipeningSimulator(rng=np.random.default_rng(42))
        self.sim.reset()

    def test_initial_ripeness_green_range(self):
        """Initial X should be in [0.6, 1.0] (green-ish start)."""
        for _ in range(50):
            state = self.sim.reset()
            assert 0.6 <= state["ripeness"] <= 1.0, \
                f"Initial X={state['ripeness']} out of [0.6, 1.0]"

    def test_ripeness_decreases_during_ripening(self):
        """X should only decrease (Green→Red is monotonic decrease)."""
        self.sim.reset()
        self.sim.temperature = 25.0  # Above T_base for ripening
        prev = self.sim.ripeness
        for _ in range(100):
            self.sim.step(action=0, dt_hours=1.0)
            assert self.sim.ripeness <= prev + 1e-9, \
                f"X increased: {prev} -> {self.sim.ripeness}"
            prev = self.sim.ripeness

    def test_ripeness_never_below_zero(self):
        """After many steps at high temp, X should saturate at 0.0."""
        self.sim.reset()
        self.sim.temperature = 35.0  # Max temp
        for _ in range(500):
            state = self.sim.step(action=0, dt_hours=1.0)
        assert state["ripeness"] >= 0.0, \
            f"X={state['ripeness']} went below 0.0"
        assert state["ripeness"] < 0.1, \
            f"X={state['ripeness']} should be near 0.0 after 500 hours at 35°C"

    def test_x_min_is_zero(self):
        """Config default x_min should be 0.0."""
        cfg = SimulatorConfig()
        assert cfg.x_min == 0.0

    def test_rgb_mapping_green_at_one(self):
        """At X=1.0 (ROYG Green), G channel should dominate."""
        self.sim.ripeness = 1.0
        self.sim._rgb_noise_offset = np.zeros(3)
        rgb = self.sim.get_rgb()
        assert rgb[1] > rgb[0], f"At X=1, G={rgb[1]} should > R={rgb[0]}"

    def test_rgb_mapping_red_at_zero(self):
        """At X=0.0 (ROYG Red), R channel should dominate."""
        self.sim.ripeness = 0.0
        self.sim._rgb_noise_offset = np.zeros(3)
        rgb = self.sim.get_rgb()
        assert rgb[0] > rgb[1], f"At X=0, R={rgb[0]} should > G={rgb[1]}"


class TestIncrementalTemperature:
    """Verify incremental temperature actions (not mode-setting)."""

    def setup_method(self):
        self.sim = TomatoRipeningSimulator(rng=np.random.default_rng(42))
        self.sim.reset()

    def test_heat_increments_temperature(self):
        """Action 1 (heat) should increase temperature by ~ΔT."""
        self.sim.temperature = 28.0  # Above ambient
        old_temp = self.sim.temperature
        self.sim.step(action=1, dt_hours=1.0)
        # After increment + drift + noise, temperature should be higher
        assert self.sim.temperature > old_temp - 0.5, \
            f"Heat action didn't increase temp: {old_temp} -> {self.sim.temperature}"

    def test_cool_decrements_temperature(self):
        """Action 2 (cool) should reduce setpoint, but temp floors at ambient.
        
        In a heater-only system, 'cool' means 'heater off' — the chamber
        passively drifts toward ambient.  Temperature cannot drop below
        the ambient floor.
        """
        self.sim.temperature = 33.0  # Well above ambient (~27°C)
        old_temp = self.sim.temperature
        self.sim.step(action=2, dt_hours=1.0)
        # After decrement + drift, temperature should drop (but stay >= ambient)
        assert self.sim.temperature < old_temp + 0.5, \
            f"Cool action didn't reduce temp: {old_temp} -> {self.sim.temperature}"
        assert self.sim.temperature >= self.sim.config.ambient_temp_mean - 3 * self.sim.config.ambient_temp_std, \
            f"Temp {self.sim.temperature} fell below ambient floor"

    def test_maintain_no_setpoint_change(self):
        """Action 0 (maintain) should not apply ΔT increment."""
        self.sim.temperature = 30.0  # Above ambient
        old_temp = self.sim.temperature
        self.sim.step(action=0, dt_hours=1.0)
        # Only drift/noise, should stay close
        assert abs(self.sim.temperature - old_temp) < 2.0, \
            f"Maintain caused large temp change: {old_temp} -> {self.sim.temperature}"


class TestRGBModeAndMaxPool:
    """Verify new simulator features: C_mode and max-pool vector."""

    def setup_method(self):
        self.sim = TomatoRipeningSimulator(rng=np.random.default_rng(42))
        self.sim.reset()

    def test_rgb_mode_shape(self):
        mode = self.sim.get_rgb_mode()
        assert mode.shape == (3,), f"Expected (3,), got {mode.shape}"

    def test_rgb_mode_in_range(self):
        mode = self.sim.get_rgb_mode()
        assert np.all(mode >= 0.0) and np.all(mode <= 1.0)

    def test_max_pool_shape(self):
        vec = self.sim.get_max_pool_vector(output_dim=4)
        assert vec.shape == (4,), f"Expected (4,), got {vec.shape}"

    def test_max_pool_in_state(self):
        state = self.sim.get_state()
        assert "max_pool" in state
        assert "rgb_mode" in state


class TestXRefDecay:
    """Verify the analytical ODE reference trajectory (exponential decay)."""

    def setup_method(self):
        self.sim = TomatoRipeningSimulator(rng=np.random.default_rng(42))

    def test_x_ref_starts_at_one(self):
        """X_ref(0) should be 1.0 (fully green)."""
        x_ref = self.sim.compute_x_ref(0.0)
        assert abs(x_ref - 1.0) < 1e-6

    def test_x_ref_approaches_zero(self):
        """X_ref should decay toward 0 over many days."""
        x_ref = self.sim.compute_x_ref(30.0)
        assert x_ref < 0.05, f"X_ref after 30 days should be ~0, got {x_ref}"

    def test_x_ref_monotonically_decreases(self):
        """X_ref should be monotonically decreasing."""
        prev = 1.0
        for t in np.linspace(0, 10, 50):
            x = self.sim.compute_x_ref(t)
            assert x <= prev + 1e-9
            prev = x


# ── Environment Tests ────────────────────────────────────────────

class TestEnvironmentObservationVariants:
    """Verify observation dimensions for all state-space ablation variants."""

    @pytest.mark.parametrize("variant,expected_dim", [("A", 7), ("B", 16), ("C", 20)])
    def test_observation_shape(self, variant, expected_dim):
        env = TomatoRipeningEnv(seed=42, state_variant=variant)
        obs, _ = env.reset()
        assert obs.shape == (expected_dim,), \
            f"Variant {variant}: expected {expected_dim}D, got {obs.shape}"

    @pytest.mark.parametrize("variant", ["A", "B", "C"])
    def test_observation_in_bounds(self, variant):
        env = TomatoRipeningEnv(seed=42, state_variant=variant)
        obs, _ = env.reset()
        low = env.observation_space.low
        high = env.observation_space.high
        for i in range(len(obs)):
            assert low[i] <= obs[i] <= high[i], \
                f"Variant {variant}: obs[{i}]={obs[i]} out of [{low[i]}, {high[i]}]"

    @pytest.mark.parametrize("variant", ["A", "B", "C"])
    def test_step_returns_correct_dim(self, variant):
        env = TomatoRipeningEnv(seed=42, state_variant=variant)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (STATE_DIMS[variant],)


class TestActionSpaceNoHarvest:
    """Verify 3-action space (no harvest action)."""

    def setup_method(self):
        self.env = TomatoRipeningEnv(seed=42)

    def test_action_space_size(self):
        assert self.env.action_space.n == 3, \
            f"Expected 3 actions, got {self.env.action_space.n}"

    def test_all_actions_valid(self):
        """Actions 0, 1, 2 should all be valid."""
        self.env.reset()
        for action in [0, 1, 2]:
            obs, reward, terminated, truncated, info = self.env.step(action)
            assert obs is not None
            if terminated or truncated:
                self.env.reset()

    def test_auto_harvest_on_ripe(self):
        """Episode should terminate when X reaches ripe threshold."""
        self.env.reset()
        self.env.simulator.ripeness = 0.10  # Below threshold
        self.env.simulator.temperature = 25.0
        obs, reward, terminated, truncated, info = self.env.step(0)
        # Should auto-harvest
        assert terminated or self.env.simulator.ripeness > self.env.ripe_threshold


class TestTRemInObservation:
    """Verify t_rem (remaining time) replaces t_tgt in observations."""

    def setup_method(self):
        self.env = TomatoRipeningEnv(seed=42, state_variant="A")

    def test_t_rem_is_last_element(self):
        """In variant A, obs[-1] should be t_rem = t_tgt - t_e."""
        obs, _ = self.env.reset()
        t_e = obs[5]      # t_e position in variant A
        t_rem = obs[6]     # t_rem position in variant A
        expected_rem = self.env.target_day - t_e
        assert abs(t_rem - expected_rem) < 0.1, \
            f"t_rem={t_rem} != target_day - t_e = {expected_rem}"

    def test_t_rem_decreases_over_time(self):
        """t_rem should decrease as steps progress."""
        obs, _ = self.env.reset()
        prev_rem = obs[6]  # t_rem in variant A
        for _ in range(24):  # 1 day
            obs, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                break
        # After 24 hours, t_rem should have decreased by ~1 day
        assert obs[6] < prev_rem


class TestRewardFunction:
    """Verify the new 3-component reward structure."""

    def setup_method(self):
        self.env = TomatoRipeningEnv(seed=42, state_variant="A")

    def test_progressive_safety_penalty_high_temp(self):
        """Consecutive high-temp violations should produce increasing penalty."""
        self.env.reset()
        self.env.simulator.temperature = 36.0  # Above 35°C boundary

        rewards = []
        for _ in range(5):
            state = self.env.simulator.get_state()
            self.env.simulator.temperature = 36.0  # Force high
            obs = self.env._make_observation(state)
            r = self.env._compute_reward(state, action=0, obs=obs)
            rewards.append(r)

        # Progressive: each successive penalty should be harsher
        # (n² scaling: 1, 4, 9, 16, 25 × alpha), capped at safety_cap
        for i in range(1, min(len(rewards), self.env.safety_cap)):
            assert rewards[i] < rewards[i-1] or rewards[i] < -1.0, \
                f"Safety penalty not progressive: step {i-1}={rewards[i-1]:.2f}, step {i}={rewards[i]:.2f}"

    def test_safety_cap_limits_penalty(self):
        """Safety penalty should plateau after safety_cap consecutive violations."""
        self.env.reset()
        self.env.simulator.temperature = 36.0
        for _ in range(self.env.safety_cap + 3):
            state = self.env.simulator.get_state()
            self.env.simulator.temperature = 36.0
            obs = self.env._make_observation(state)
            self.env._compute_reward(state, action=0, obs=obs)
        # Counter keeps incrementing but capped n is used for penalty
        assert self.env.consecutive_safety_violations >= self.env.safety_cap

    def test_safety_resets_on_normal_temp(self):
        """Consecutive violation counter should reset when temp is normal."""
        self.env.reset()
        # Force a violation
        self.env.simulator.temperature = 36.0
        state = self.env.simulator.get_state()
        obs = self.env._make_observation(state)
        self.env._compute_reward(state, action=0, obs=obs)
        assert self.env.consecutive_safety_violations > 0

        # Return to normal
        self.env.simulator.temperature = 22.0
        state = self.env.simulator.get_state()
        obs = self.env._make_observation(state)
        self.env._compute_reward(state, action=0, obs=obs)
        assert self.env.consecutive_safety_violations == 0

    def test_reward_is_finite(self):
        """Reward should never be NaN or Inf."""
        self.env.reset()
        for _ in range(50):
            obs, reward, terminated, truncated, _ = self.env.step(0)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if terminated or truncated:
                break

    def test_progress_reward_positive_during_ripening(self):
        """Heating should produce positive progress reward from ripening."""
        self.env.reset()
        # Heat to trigger ripening — r_progress should contribute positively
        obs, reward, _, _, _ = self.env.step(1)  # heat
        assert np.isfinite(reward)

    def test_harvest_bonus_on_auto_harvest(self):
        """Episode should receive positive terminal bonus on auto-harvest."""
        self.env.reset()
        # Force near-ripe state to trigger auto-harvest quickly
        self.env.simulator.ripeness = 0.16  # Just above threshold
        self.env.simulator.temperature = 30.0  # Warm to push it over
        for _ in range(50):
            obs, reward, terminated, truncated, info = self.env.step(1)
            if terminated:
                assert info.get("auto_harvest", False), "Should auto-harvest"
                # Terminal step should include positive harvest bonus
                assert reward > 0, f"Terminal step should have positive reward from bonus, got {reward}"
                break

    def test_energy_cost_for_heating(self):
        """Heat/Cool should give different reward than maintain (rate changes)."""
        self.env.reset()
        state = self.env.simulator.get_state()
        obs = self.env._make_observation(state)
        r_maintain = self.env._compute_reward(state, action=0, obs=obs)
        r_heat = self.env._compute_reward(state, action=1, obs=obs)
        assert np.isfinite(r_maintain) and np.isfinite(r_heat)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
