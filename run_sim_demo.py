
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ml_training.rl.simulator import TomatoRipeningSimulator, SimulatorConfig
import numpy as np
import matplotlib.pyplot as plt

def run_visual_demo():
    print("=== Edge-RL Digital Twin Simulator Visualization ===")
    
    sim = TomatoRipeningSimulator()
    state = sim.reset()
    
    history = {
        "hours": [],
        "ripeness": [],
        "temperature": [],
        "humidity": [],
        "actions": []
    }
    
    # Run for 120 hours (5 days)
    for hour in range(0, 121):
        # Log state
        history["hours"].append(hour)
        history["ripeness"].append(state["_true_ripeness"])
        history["temperature"].append(state["_true_temperature"])
        history["humidity"].append(state["_true_humidity"])
        
        # Simple policy: keep at 15°C until day 3, then 22°C
        target_temp = 15.0 if hour < 72 else 22.0
        
        if state["_true_temperature"] < target_temp - 0.5:
            action = 1 # heat
        elif state["_true_temperature"] > target_temp + 0.5:
            action = 2 # cool
        else:
            action = 0 # maintain
            
        history["actions"].append(action)
        state = sim.step(action)
        
        if sim.is_overripe():
            print(f"Episode terminated at hour {hour} (overripe)")
            break
            
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    color = 'tab:red'
    ax1.set_ylabel('Ripeness Stage (0-5)', color=color)
    ax1.plot(history["hours"], history["ripeness"], color=color, linewidth=2, label="Ripeness")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 5.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Tomato Ripening Trajectory (Digital Twin)")
    
    ax1b = ax1.twinx()
    color = 'tab:blue'
    ax1b.set_ylabel('Temperature (°C)', color=color)
    ax1b.plot(history["hours"], history["temperature"], color=color, linestyle='--', alpha=0.7, label="Temperature")
    ax1b.tick_params(axis='y', labelcolor=color)
    ax1b.set_ylim(10, 30)
    
    # Action labels
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Action')
    action_names = ["Maintain", "Heat", "Cool"]
    ax2.scatter(history["hours"], history["actions"], c=history["actions"], cmap='viridis', s=10)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(action_names)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "outputs/ripening_trajectory_demo.png"
    plt.savefig(output_path)
    print(f"\nSaved visualization to {output_path}")
    print("==================================================")

if __name__ == "__main__":
    run_visual_demo()
