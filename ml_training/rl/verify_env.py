
import numpy as np
from ml_training.rl.environment import TomatoRipeningEnv

def main():
    print("Initializing Environment...")
    env = TomatoRipeningEnv()
    obs, info = env.reset()
    
    print("\nInitial State:")
    env.render()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    # Test RGB values
    rgb = obs[0:3]
    print(f"RGB: {rgb}")
    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0), "RGB values out of range!"
    
    # Test dR/dt
    rate = obs[3]
    print(f"Ripening Rate: {rate}")
    
    # Run a few steps with HEATING (Action 1)
    print("\n--- Testing HEATING (Action 1) ---")
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(1)
        env.render()
        print(f"Reward: {reward:.4f}")
        if terminated or truncated:
            break
            
    # Run a few steps with COOLING (Action 2)
    print("\n--- Testing COOLING (Action 2) ---")
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(2)
        env.render()
        print(f"Reward: {reward:.4f}")
        if terminated or truncated:
            break

    # Test Temperature Penalty
    print("\n--- Testing Safety Penalty (Please wait, overheating...) ---")
    # Force high temp
    env.simulator.temperature = 36.0 
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"Temp: {env.simulator.temperature:.1f}°C, Reward: {reward}")
    if reward <= -5.0:
        print("SUCCESS: Temperature penalty applied.")
    else:
        print("FAILURE: Temperature penalty NOT applied.")

if __name__ == "__main__":
    main()
