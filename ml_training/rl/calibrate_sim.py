#!/usr/bin/env python3
"""Sim-to-Real Calibration Tool for Edge-RL

This script optimizes the simulator ODE parameters (k1, T_base) against
real-world physical tomato ripening data. It ensures that the digital
twin accurately reflects the specific hardware, enclosure insulation,
and local tomato variants used in physical deployments.

Usage:
    python -m ml_training.rl.calibrate_sim --csv data/real_run_1.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml

def ripening_ode(t_hours, temp_array, k1, t_base, x_init):
    """
    Simulate the Chromatic Index (X) over time given an array of hourly temperatures.
    ODE: dX/dt = -k1 * (T - T_base) * X
    """
    x = np.zeros(len(t_hours))
    x[0] = x_init
    
    # Simple Euler integration (1 hour steps)
    dt_days = 1.0 / 24.0
    for i in range(1, len(t_hours)):
        # Temperature is bounded by t_base (no reverse ripening)
        eff_temp = max(0.0, temp_array[i-1] - t_base)
        dx = -k1 * eff_temp * x[i-1] * dt_days
        
        new_x = x[i-1] + dx
        x[i] = np.clip(new_x, 0.0, 1.0)
        
    return x

def objective(params, t_hours, temp_array, x_real):
    """Loss function: Mean Squared Error between sim and real X."""
    k1, t_base = params
    
    # Hard bounds penalty to guide optimizer
    if k1 <= 0.001 or k1 > 0.5 or t_base < 5.0 or t_base > 20.0:
        return 1e6
        
    x_sim = ripening_ode(t_hours, temp_array, k1, t_base, x_real[0])
    mse = np.mean((x_sim - x_real) ** 2)
    return mse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV file with columns: hour, temp_c, chromatic_x")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml")
    parser.add_argument("--out", type=str, default="outputs/calibration")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading real-world data from {args.csv}...")
    try:
        df = pd.read_csv(args.csv)
        t_hours = df['hour'].values
        temp_array = df['temp_c'].values
        x_real = df['chromatic_x'].values
    except Exception as e:
        print(f"Error reading CSV {args.csv}: {e}")
        # Create a dummy CSV for the user to see the expected format
        dummy_df = pd.DataFrame({
            "hour": np.arange(0, 120),  # 5 days
            "temp_c": np.random.normal(25, 2, 120),
            "chromatic_x": np.linspace(0.9, 0.2, 120) + np.random.normal(0, 0.02, 120)
        })
        dummy_path = out_dir / "example_format.csv"
        dummy_df.to_csv(dummy_path, index=False)
        print(f"Generated example CSV format at {dummy_path}")
        return

    # Load initial guesses from config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    sim_cfg = config.get("rl", {}).get("simulator", {})
    k1_init = sim_cfg.get("k1", 0.02)
    t_base_init = sim_cfg.get("t_base", 12.5)
    
    print(f"Initial parameters from config: k1={k1_init:.4f}, T_base={t_base_init:.1f}°C")
    
    # Run Scipy Optimizer
    print("Running Sim-to-Real calibration optimizer...")
    res = minimize(
        objective, 
        x0=[k1_init, t_base_init],
        args=(t_hours, temp_array, x_real),
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )
    
    k1_opt, t_base_opt = res.x
    
    print("\n" + "="*40)
    print("  CALIBRATION RESULTS")
    print("="*40)
    if res.success:
        print(f"Optimization Status: SUCCESS")
        print(f"Final MSE Loss:      {res.fun:.6f}")
        print("\nUpdated Parameters:")
        print(f"  k1 (Ripening rate): {k1_opt:.5f} (was {k1_init})")
        print(f"  T_base (Base temp): {t_base_opt:.2f}°C (was {t_base_init}°C)")
    else:
        print("Optimization FAILED to converge.")
        print(res.message)
    print("="*40)

    # Generate fitting plot
    x_sim_init = ripening_ode(t_hours, temp_array, k1_init, t_base_init, x_real[0])
    x_sim_opt  = ripening_ode(t_hours, temp_array, k1_opt, t_base_opt, x_real[0])
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Sim-to-Real Calibration: Chromatic Index (X)")
    plt.scatter(t_hours, x_real, color='black', s=10, label='Real Hardware Sensor', alpha=0.5)
    plt.plot(t_hours, x_sim_init, '--', color='red', label=f'Initial ODE (k1={k1_init})')
    plt.plot(t_hours, x_sim_opt, '-', color='blue', linewidth=2, label=f'Calibrated ODE (k1={k1_opt:.4f})')
    plt.ylabel("Chromatic Index X")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t_hours, temp_array, color='orange', label='Chamber Temp (°C)')
    plt.axhline(t_base_opt, color='blue', linestyle=':', label=f'Calibrated T_base ({t_base_opt:.1f}°C)')
    plt.xlabel("Time (Hours)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    
    plot_path = out_dir / "calibration_fit.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved calibration plot to {plot_path}")
    
    print("\nNext step: Update `ml_training/config.yaml` -> `rl.simulator` with the Calibrated ODE values.")

if __name__ == "__main__":
    main()
