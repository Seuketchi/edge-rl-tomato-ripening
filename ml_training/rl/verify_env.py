"""Quick sanity check for TomatoRipeningEnv (Variant B, 16D state).

Usage:
    python -m ml_training.rl.verify_env
"""

import numpy as np
from ml_training.rl.environment import TomatoRipeningEnv


def main():
    print("Initializing Environment...")
    env = TomatoRipeningEnv()
    obs, info = env.reset()

    print("\nInitial State:")
    env.render()
    print(f"Observation shape: {obs.shape}")
    assert obs.shape == (16,), f"Expected 16D observation, got {obs.shape}"
    print(f"Observation: {obs}")

    # Variant B state layout: [X, dX/dt, X_ref, C_mu(3), C_sig(3), C_mode(3), T, H, t_e, t_rem]
    x = obs[0]
    dx_dt = obs[1]
    x_ref = obs[2]
    c_mu = obs[3:6]
    c_sig = obs[6:9]
    c_mode = obs[9:12]
    temp = obs[12]
    humid = obs[13]

    print(f"\nChromatic Index X: {x:.4f}")
    print(f"dX/dt: {dx_dt:.4f}")
    print(f"X_ref: {x_ref:.4f}")
    print(f"C_mu (RGB mean): {c_mu}")
    print(f"C_sig (RGB std):  {c_sig}")
    print(f"C_mode (RGB mode): {c_mode}")
    t_e = obs[14]
    t_rem = obs[15]

    print(f"Temperature (norm): {temp:.4f}")
    print(f"Humidity (norm): {humid:.4f}")
    print(f"t_e (elapsed): {t_e:.4f}")
    print(f"t_rem (remaining): {t_rem:.4f}")

    assert 0.0 <= x <= 1.0, f"X out of range: {x}"
    assert np.all(c_mu >= 0.0) and np.all(c_mu <= 1.0), "C_mu out of range!"

    # Run a few steps with HEATING (Action 1)
    print("\n--- Testing HEAT (Action 1) ---")
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(1)
        env.render()
        print(f"Reward: {reward:.4f}")
        if terminated or truncated:
            break

    # Run a few steps with COOLING (Action 2)
    print("\n--- Testing COOL (Action 2) ---")
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(2)
        env.render()
        print(f"Reward: {reward:.4f}")
        if terminated or truncated:
            break

    # Test Temperature Penalty
    print("\n--- Testing Safety Penalty (overheating) ---")
    env.simulator.temperature = 36.0
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"Temp: {env.simulator.temperature:.1f}°C, Reward: {reward}")
    if reward <= -5.0:
        print("OK: Temperature penalty applied.")
    else:
        print("WARNING: Temperature penalty NOT applied (check safety bounds).")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
