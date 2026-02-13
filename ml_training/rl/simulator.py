"""Physics-based tomato ripening simulator.

Implements the ripening ODE:
    dR/dt = k₁ × (T - T_base) × (1 - R/R_max)

where:
    R = ripeness stage [0.0 - 5.0]
    T = temperature [12.5°C - 25.0°C]
    k₁ = ripening rate constant
    T_base = base temperature for ripening (12.5°C)
    R_max = maximum ripeness (5.0)

Includes domain randomization for sim-to-real transfer:
    - Temperature sensor noise
    - Humidity sensor noise
    - Initial condition variation
    - Ripening rate variation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulatorConfig:
    """Configuration for the ripening simulator."""
    k1: float = 0.08                # Ripening rate constant
    t_base: float = 12.5            # Base temperature (°C)
    r_max: float = 5.0              # Maximum ripeness stage
    temp_noise_std: float = 0.5     # Temperature sensor noise (°C)
    humidity_noise_std: float = 2.0  # Humidity sensor noise (%)
    k1_variation: float = 0.02      # Per-episode k1 randomization range
    initial_ripeness_range: tuple[float, float] = (0.0, 2.0)
    initial_temp_range: tuple[float, float] = (18.0, 22.0)
    initial_humidity_range: tuple[float, float] = (65.0, 85.0)


class TomatoRipeningSimulator:
    """Physics-based tomato ripening simulation.

    Simulates the ripening process of a tomato batch over time,
    including temperature control actions and environmental noise.

    Args:
        config: Simulator configuration.
        rng: NumPy random generator for reproducibility.
    """

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config or SimulatorConfig()
        self.rng = rng or np.random.default_rng()

        # State
        self.ripeness: float = 0.0
        self.temperature: float = 20.0
        self.humidity: float = 75.0
        self.hours_elapsed: float = 0.0

        # Per-episode randomization
        self._episode_k1: float = self.config.k1

    def reset(self) -> dict[str, float]:
        """Reset simulator to random initial conditions.

        Returns:
            Dict with initial state values.
        """
        cfg = self.config

        # Randomize initial conditions
        self.ripeness = self.rng.uniform(*cfg.initial_ripeness_range)
        self.temperature = self.rng.uniform(*cfg.initial_temp_range)
        self.humidity = self.rng.uniform(*cfg.initial_humidity_range)
        self.hours_elapsed = 0.0

        # Per-episode ripening rate variation (domain randomization)
        self._episode_k1 = cfg.k1 + self.rng.uniform(
            -cfg.k1_variation, cfg.k1_variation
        )

        return self.get_state()

    def step(self, action: int, dt_hours: float = 1.0) -> dict[str, float]:
        """Advance simulation by one timestep.

        Args:
            action: Control action:
                0 = maintain (do nothing)
                1 = heat (+2°C target adjustment)
                2 = cool (-2°C target adjustment)
                3 = harvest (terminal action)
            dt_hours: Timestep in hours.

        Returns:
            Dict with updated state values.
        """
        # Apply temperature control action
        if action == 1:  # heat
            self.temperature = min(25.0, self.temperature + 1.0 * dt_hours)
        elif action == 2:  # cool
            self.temperature = max(12.5, self.temperature - 1.0 * dt_hours)
        # action 0 (maintain) and 3 (harvest) don't change temperature

        # Small natural temperature drift
        self.temperature += self.rng.normal(0, 0.1) * dt_hours
        self.temperature = np.clip(self.temperature, 12.5, 25.0)

        # Humidity drift (inversely correlated with temperature changes)
        self.humidity += self.rng.normal(0, 0.5) * dt_hours
        self.humidity = np.clip(self.humidity, 50.0, 99.0)

        # Ripening ODE: dR/dt = k1 * (T - T_base) * (1 - R/R_max)
        cfg = self.config
        if self.temperature > cfg.t_base and self.ripeness < cfg.r_max:
            dR = (
                self._episode_k1
                * (self.temperature - cfg.t_base)
                * (1.0 - self.ripeness / cfg.r_max)
                * dt_hours
            )
            self.ripeness = min(cfg.r_max, self.ripeness + dR)

        self.hours_elapsed += dt_hours

        return self.get_state()

    def get_state(self) -> dict[str, float]:
        """Get current state with simulated sensor noise.

        Returns:
            Dict with noisy sensor readings and true ripeness.
        """
        cfg = self.config
        return {
            "ripeness": self.ripeness,
            "temperature": self.temperature + self.rng.normal(0, cfg.temp_noise_std),
            "humidity": self.humidity + self.rng.normal(0, cfg.humidity_noise_std),
            "hours_elapsed": self.hours_elapsed,
            "days_elapsed": self.hours_elapsed / 24.0,
            # True values (not available on real hardware, for evaluation only)
            "_true_temperature": self.temperature,
            "_true_humidity": self.humidity,
            "_true_ripeness": self.ripeness,
        }

    def get_ripeness_stage(self) -> int:
        """Get discrete ripeness stage (0-5).

        Returns:
            Integer ripeness stage.
        """
        return int(np.clip(np.round(self.ripeness), 0, 5))

    def is_overripe(self) -> bool:
        """Check if tomato has exceeded optimal ripeness."""
        return self.ripeness >= self.config.r_max

    def compute_quality_score(self, target_ripeness: float = 4.0) -> float:
        """Compute quality score based on ripeness at harvest.

        Args:
            target_ripeness: Desired ripeness at harvest.

        Returns:
            Quality score in [0, 1].
        """
        error = abs(self.ripeness - target_ripeness)
        return max(0.0, 1.0 - error / self.config.r_max)
