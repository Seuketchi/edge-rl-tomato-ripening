"""Physics-based tomato ripening simulator.

Implements the Chromatic Evolution ODE (ROYG spectral mapping):
    dX/dt = -k₁ × (T - T_base) × X

where:
    X = Continuous Chromatic Index [0.0 - 1.0] (Red → Green, ROYG spectral)
        X ≈ 1.0 : Green / unripe  (long-wavelength end of visible)
        X ≈ 0.0 : Red   / ripe    (short-wavelength end of visible)
    T = temperature [12.5°C - 35.0°C]
    k₁ = cultivar-specific rate constant
    T_base = 12.5°C

Ripening drives X downward from ~1.0 toward 0.0.

Includes domain randomization for sim-to-real transfer:
    - Temperature sensor noise
    - Humidity sensor noise
    - Initial condition variation
    - Ripening rate variation

Temperature actions are INCREMENTAL (ΔT = ±1°C per step),
modelling a supervisory RL controller issuing setpoints to a
low-level PID temperature controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulatorConfig:
    """Configuration for the ripening simulator."""
    k1: float = 0.08                # Ripening rate constant
    t_base: float = 12.5            # Base temperature (°C)
    x_min: float = 0.0              # Minimum chromatic index (fully ripe / Red)
    temp_noise_std: float = 0.5     # Temperature sensor noise (°C)
    humidity_noise_std: float = 2.0  # Humidity sensor noise (%)
    k1_variation: float = 0.02      # Per-episode k1 randomization range
    initial_ripeness_range: tuple[float, float] = (0.6, 1.0)  # Green-ish start
    initial_temp_range: tuple[float, float] = (25.0, 30.0)  # At/above ambient (Philippines)
    initial_humidity_range: tuple[float, float] = (65.0, 85.0)
    delta_t_step: float = 1.0       # Incremental temperature change per action (°C)
    # Ambient environment (heater-only system: cannot cool below ambient)
    ambient_temp_mean: float = 27.0  # Mean ambient temperature (°C) — Philippines climate
    ambient_temp_std: float = 2.0    # Ambient variation std (diurnal cycle + noise)
    # Max-pool simulation: spatial heterogeneity parameters
    spatial_grid_size: int = 4      # NxN grid for spatial pixel simulation
    spatial_std: float = 0.06       # Std of per-patch chromatic variation


class TomatoRipeningSimulator:
    """Physics-based tomato ripening simulation.

    Simulates the ripening process of a tomato batch over time,
    including incremental temperature control and environmental noise.

    Convention (ROYG spectral mapping):
        X = 1.0 → Green (unripe)
        X = 0.0 → Red   (ripe)
        Ripening decreases X.

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
        self.ripeness: float = 1.0   # Start green (X≈1)
        self.temperature: float = 20.0
        self.humidity: float = 75.0
        self.hours_elapsed: float = 0.0

        # Rate tracking (for dX/dt observation)
        self.last_ripeness: float = 1.0
        self.current_rate: float = 0.0

        # Per-episode randomization
        self._episode_k1: float = self.config.k1
        self._rgb_noise_offset: np.ndarray = self.rng.normal(0, 0.02, 3)
        # Per-episode spatial variation offsets (for max-pool simulation)
        self._spatial_offsets: np.ndarray = np.zeros(
            (self.config.spatial_grid_size, self.config.spatial_grid_size)
        )

    def reset(self) -> dict[str, float | np.ndarray]:
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

        self.last_ripeness = self.ripeness
        self.current_rate = 0.0
        self._rgb_noise_offset = self.rng.normal(0, 0.02, 3)

        # Spatial heterogeneity offsets for max-pool simulation
        gs = cfg.spatial_grid_size
        self._spatial_offsets = self.rng.normal(0, cfg.spatial_std, (gs, gs))

        # Per-episode ripening rate variation (domain randomization)
        self._episode_k1 = cfg.k1 + self.rng.uniform(
            -cfg.k1_variation, cfg.k1_variation
        )

        return self.get_state()

    def step(self, action: int, dt_hours: float = 1.0) -> dict[str, float | np.ndarray]:
        """Advance simulation by one timestep.

        Heater-only actuation (no active cooling hardware):
            0 = maintain (no setpoint change)
            1 = heat (+ΔT °C) — heater relay ON
            2 = cool (-ΔT °C) — heater relay OFF, passive drift toward ambient

        Temperature is floored at ambient — the system cannot cool
        below the surrounding environment temperature.

        Args:
            action: Control action (0=maintain, 1=heat, 2=cool).
            dt_hours: Timestep in hours.

        Returns:
            Dict with updated state values.
        """
        cfg = self.config

        # --- Incremental temperature control ---
        if action == 1:  # heat: increment setpoint
            self.temperature += cfg.delta_t_step
        elif action == 2:  # cool: decrement setpoint
            self.temperature -= cfg.delta_t_step
        # action == 0: maintain — no setpoint change

        # Natural drift toward ambient (models imperfect insulation)
        ambient_temp = cfg.ambient_temp_mean + self.rng.normal(0, cfg.ambient_temp_std)
        diff = ambient_temp - self.temperature
        self.temperature += diff * 0.05 * dt_hours  # Slow drift

        # Small stochastic noise
        self.temperature += self.rng.normal(0, 0.1) * dt_hours

        # Heater-only system: cannot cool below ambient (no active cooling)
        self.temperature = max(self.temperature, ambient_temp)
        self.temperature = np.clip(self.temperature, 10.0, 40.0)

        # Humidity drift
        self.humidity += self.rng.normal(0, 0.5) * dt_hours
        self.humidity = np.clip(self.humidity, 40.0, 99.0)

        # --- Ripening ODE: dX/dt = -k1 * (T - T_base) * X ---
        # X decreases as the tomato ripens (Green → Red in ROYG).
        # We use the negative sign so that positive k1 drives X down.
        dX = 0.0
        if self.temperature > cfg.t_base and self.ripeness > cfg.x_min:
            dX = (
                -self._episode_k1
                * (self.temperature - cfg.t_base)
                * self.ripeness
                * (dt_hours / 24.0)  # Convert hours to days
            )
            self.last_ripeness = self.ripeness
            self.ripeness = max(cfg.x_min, self.ripeness + dX)

            # Instantaneous rate (per day) for observation
            if dt_hours > 0:
                self.current_rate = dX * (24.0 / dt_hours)

        self.hours_elapsed += dt_hours
        return self.get_state()

    def get_rgb(self) -> np.ndarray:
        """Simulate RGB sensor reading based on chromatic index.

        ROYG convention: X=1 is Green, X=0 is Red.
        Interpolates between:
        - X=1.0 (Green):   [0.2, 0.8, 0.2]
        - X=0.6 (Turning): [0.6, 0.7, 0.2]
        - X=0.0 (Red):     [0.9, 0.1, 0.1]

        Returns:
            Unnormalized RGB vector (approx [0-1] range) with noise.
        """
        x = self.ripeness

        # Base colors (ROYG: high X = Green, low X = Red)
        if x > 0.6:
            # Green → Turning
            alpha = (x - 0.6) / 0.4   # 0 at x=0.6, 1 at x=1.0
            red = 0.6 - (0.4 * alpha)  # 0.6 → 0.2
            grn = 0.7 + (0.1 * alpha)  # 0.7 → 0.8
            blu = 0.2
        else:
            # Turning → Red
            alpha = x / 0.6            # 0 at x=0, 1 at x=0.6
            red = 0.9 - (0.3 * alpha)  # 0.9 → 0.6
            grn = 0.1 + (0.6 * alpha)  # 0.1 → 0.7
            blu = 0.1 + (0.1 * alpha)  # 0.1 → 0.2

        rgb = np.array([red, grn, blu], dtype=np.float32)

        # Add sensor noise and bias
        noise = self.rng.normal(0, 0.02, 3)
        return np.clip(rgb + noise + self._rgb_noise_offset, 0.0, 1.0)

    def get_rgb_mode(self) -> np.ndarray:
        """Compute per-channel mode approximation (most vivid pixel value).

        Simulates a spatial pixel distribution across the tomato ROI
        and returns the per-channel maximum frequency bin (mode).
        In practice, this highlights the most vivid colour present.

        Returns:
            Mode RGB vector of shape (3,).
        """
        gs = self.config.spatial_grid_size
        base_rgb = self.get_rgb()

        # Each patch has a slightly different colour
        patch_rgbs = np.tile(base_rgb, (gs * gs, 1))  # (N*N, 3)
        spatial_flat = self._spatial_offsets.ravel()
        for ch in range(3):
            patch_rgbs[:, ch] += spatial_flat * (0.5 if ch == 0 else 0.3)

        patch_rgbs = np.clip(patch_rgbs, 0.0, 1.0)

        # Mode ≈ value with highest frequency (bin with most patches)
        mode_rgb = np.zeros(3, dtype=np.float32)
        for ch in range(3):
            counts, bin_edges = np.histogram(patch_rgbs[:, ch], bins=8)
            max_bin = np.argmax(counts)
            mode_rgb[ch] = 0.5 * (bin_edges[max_bin] + bin_edges[max_bin + 1])

        return mode_rgb

    def get_max_pool_vector(self, output_dim: int = 4) -> np.ndarray:
        """Simulate a max-pooling operation on a spatial feature map.

        Generates a synthetic NxN spatial grid of pixel intensities
        (representing the feature map), then applies max-pooling with
        a stride sized to produce a 1×output_dim vector.

        This mirrors the CNN max-pooling idea: identify the most
        activated pixels across spatial regions.

        Args:
            output_dim: Desired output vector length (N < 5 recommended).

        Returns:
            Max-pooled vector of shape (output_dim,).
        """
        gs = self.config.spatial_grid_size
        base_rgb = self.get_rgb()

        # Build a single-channel "feature map" by combining RGB
        # weighted by ripeness-relevance (R-G difference is most
        # informative for ripening).
        feature = np.zeros((gs, gs), dtype=np.float32)
        for i in range(gs):
            for j in range(gs):
                offset = self._spatial_offsets[i, j]
                local_r = np.clip(base_rgb[0] + offset * 0.5, 0, 1)
                local_g = np.clip(base_rgb[1] - offset * 0.3, 0, 1)
                feature[i, j] = local_r - local_g  # R-G difference

        # Flatten and apply max-pool to get `output_dim` values
        flat = feature.ravel()  # length gs*gs
        if len(flat) < output_dim:
            flat = np.pad(flat, (0, output_dim - len(flat)))

        # Divide into output_dim regions and take max of each
        indices = np.array_split(np.arange(len(flat)), output_dim)
        result = np.array(
            [flat[idx].max() for idx in indices],
            dtype=np.float32,
        )
        return result

    def get_state(self) -> dict[str, float | np.ndarray]:
        """Get current state with simulated sensor noise.

        Returns:
            Dict with noisy sensor readings and true chromatic index.
        """
        cfg = self.config
        rgb = self.get_rgb()
        rgb_mode = self.get_rgb_mode()
        max_pool_vec = self.get_max_pool_vector()

        return {
            "ripeness": self.ripeness,          # Ground truth X (ROYG: 1=Green, 0=Red)
            "rgb": rgb,                          # Noisy mean RGB
            "rgb_mode": rgb_mode,                # Per-channel mode
            "max_pool": max_pool_vec,            # Max-pooled spatial vector
            "ripening_rate": self.current_rate,   # dX/dt (negative = ripening)
            "temperature": self.temperature + self.rng.normal(0, cfg.temp_noise_std),
            "humidity": self.humidity + self.rng.normal(0, cfg.humidity_noise_std),
            "hours_elapsed": self.hours_elapsed,
            "days_elapsed": self.hours_elapsed / 24.0,
            # True values (for evaluation only — not available on real HW)
            "_true_temperature": self.temperature,
            "_true_humidity": self.humidity,
            "_true_ripeness": self.ripeness,
        }

    def compute_x_ref(self, t_days: float, t_ideal: float = 20.0) -> float:
        """Compute reference chromatic index from ODE at ideal temperature.

        Analytical solution of dX/dt = -k1*(T_ideal - T_base)*X:
            X_ref(t) = X_0 * exp(-k1 * (T_ideal - T_base) * t)

        where X_0 ≈ 1.0 (green start).

        In ROYG convention, X_ref decays from ~1.0 toward 0.0 over time.

        Args:
            t_days: Time elapsed in days.
            t_ideal: Ideal reference temperature (°C).

        Returns:
            Reference chromatic index X_ref ∈ [0, 1].
        """
        cfg = self.config
        return float(
            np.exp(-cfg.k1 * (t_ideal - cfg.t_base) * t_days)
        )

    def is_ripe(self, threshold: float = 0.15) -> bool:
        """Check if tomato has reached target ripeness (X near 0)."""
        return self.ripeness <= threshold

    def compute_quality_score(self, target_ripeness: float = 0.1) -> float:
        """Compute quality score based on chromatic index at harvest.

        In ROYG convention, lower X = more ripe = higher quality
        (when close to target).

        Args:
            target_ripeness: Desired chromatic index at harvest (near 0 = ripe).

        Returns:
            Quality score in [0, 1].
        """
        error = abs(self.ripeness - target_ripeness)
        return max(0.0, 1.0 - error)
