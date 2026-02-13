
import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from pathlib import Path

from ml_training.rl.environment import TomatoRipeningEnv
from ml_training.rl.distill import StudentPolicy

def load_student_policy(model_path, env):
    # Load metadata to get architecture
    checkpoint = torch.load(model_path)
    
    state_dim = checkpoint.get("state_dim", env.observation_space.shape[0])
    action_dim = checkpoint.get("action_dim", env.action_space.n)
    hidden_sizes = checkpoint.get("hidden_sizes", [64, 64])
    
    student = StudentPolicy(state_dim, action_dim, hidden_sizes)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()
    return student

def items_to_text(items):
    return "\n".join([f"{k}: {v}" for k, v in items.items()])

def get_tomato_color(ripeness):
    # Map ripeness (1.0 - 6.0) to RGB color
    # 1=Green, 2=Breaker, 3=Turning, 4=Pink, 5=LightRed, 6=Red
    # Simplified gradient: Green -> Yellow -> Red
    
    # Normalize 1.0-6.0 to 0.0-1.0
    norm = np.clip((ripeness - 1.0) / 5.0, 0.0, 1.0)
    
    # Green to Red interpolation
    # Green: (0, 1, 0), Red: (1, 0, 0)
    # But tomatoes go Green -> Yellow/Orange -> Red
    
    if norm < 0.5:
        # Green to Yellow (1, 1, 0)
        r = 2 * norm
        g = 1.0
        b = 0.0
    else:
        # Yellow to Red
        r = 1.0
        g = 1.0 - 2 * (norm - 0.5)
        b = 0.0
        
    return (r, g, b)

def simulate_episode(env, policy, output_path, seed=None):
    obs, info = env.reset(seed=seed)
    frames = []
    
    # Setup plot
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    ax_tomato = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_action = fig.add_subplot(gs[1, :])
    
    # Tomato visual
    tomato_circle = Circle((0.5, 0.5), 0.4, color='green')
    ax_tomato.add_patch(tomato_circle)
    ax_tomato.set_xlim(0, 1)
    ax_tomato.set_ylim(0, 1)
    ax_tomato.axis('off')
    ax_tomato.set_title("Tomato State (Digital Twin)")
    
    # Stats text
    stats_text = ax_stats.text(0.1, 0.9, "", transform=ax_stats.transAxes, verticalalignment='top', fontsize=12)
    ax_stats.axis('off')
    ax_stats.set_title("Environment Physics")
    
    # Action Timeline
    actions_hist = []
    times_hist = []
    action_colors = ['gray', 'red', 'blue', 'green'] # Maintain, Heat, Cool, Harvest
    action_names = ['Maintain', 'Heat (+1°C)', 'Cool (-1°C)', 'Harvest']
    
    done = False
    step = 0
    total_reward = 0
    
    print("Simulating episode...")
    
    while not done:
        # Get simulation state directly for visualization
        sim_state = env.simulator.get_state()
        
        # Decide action
        action = policy.predict(obs)
        if hasattr(policy, 'predict') and not isinstance(action, int) and len(action.shape) > 0:
             # Handle SB3 predict return type if using teacher
             action = int(action[0])
        
        # Color update
        color = get_tomato_color(sim_state["ripeness"])
        
        # Record frame data
        state_str = f"""
Day: {sim_state['days_elapsed']:.2f} / {env.target_day:.1f}
Ripeness: {sim_state['ripeness']:.2f} (Target: 4.0-5.0)
Temperature: {sim_state['temperature']:.1f}°C
Humidity: {sim_state['humidity']:.1f}%
Action: {action_names[action]}
Reward so far: {total_reward:.2f}
        """
        print(f"Step {step}: {state_str.strip()}")
        
        def update_frame(frame_idx, c=color, s=state_str, a=action, t=sim_state['days_elapsed']):
            tomato_circle.set_color(c)
            stats_text.set_text(s)
            
            # Update timeline (simplified for animation function limitation)
            # In a real animation we'd append, but here we just show current state
            ax_action.clear()
            ax_action.set_xlim(0, 7) # Max days
            ax_action.set_ylim(0, 1)
            ax_action.set_yticks([])
            ax_action.set_xlabel("Days Elapsed")
            
            # Draw history
            for i, (act, time) in enumerate(zip(actions_hist[:frame_idx+1], times_hist[:frame_idx+1])):
                 rect = Rectangle((time, 0), 1/24.0, 1, color=action_colors[act], alpha=0.6)
                 ax_action.add_patch(rect)
                 
            # Legend
            patches = [Rectangle((0,0),1,1, color=c) for c in action_colors]
            ax_action.legend(patches, action_names, loc='upper left', ncol=4, fontsize='small')

        # Store closure for animation
        frames.append((color, state_str, action, sim_state['days_elapsed']))
        
        actions_hist.append(action)
        times_hist.append(sim_state['days_elapsed'])
        
        # Step env
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1

    # Create animation
    print(f"Generating animation ({len(frames)} frames)...")
    
    def animate(i):
        c, s, a, t = frames[i]
        tomato_circle.set_color(c)
        stats_text.set_text(s)
        
        ax_action.clear()
        ax_action.set_xlim(0, 7)
        ax_action.set_ylim(0, 1)
        ax_action.set_yticks([])
        ax_action.set_xlabel("Days Elapsed")
        ax_action.set_title("Action History")
        
        # Draw all history up to i
        for idx in range(i+1):
            act = actions_hist[idx]
            time = times_hist[idx]
            # Width is 1 step (1/24 day)
            ax_action.add_patch(Rectangle((time, 0), 1/24.0, 1, color=action_colors[act], alpha=0.8))
        
        ax_action.legend([Rectangle((0,0),1,1, color=c) for c in action_colors], action_names, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize='small')

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=100)
    
    # Save
    writer = animation.PillowWriter(fps=10)
    ani.save(output_path, writer=writer)
    print(f"Saved animation to {output_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_training/config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to student_policy.pth")
    parser.add_argument("--output", type=str, default="tomato_sim.gif")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for environment")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    env = TomatoRipeningEnv(config=config)
    student = load_student_policy(args.model, env)
    
    # Pass seed manually since we modified simulate_episode signature implicitly?
    # No, simulate_episode calls env.reset(seed=42). We need to change that too.
    simulate_episode(env, student, args.output, seed=args.seed)

if __name__ == "__main__":
    main()
