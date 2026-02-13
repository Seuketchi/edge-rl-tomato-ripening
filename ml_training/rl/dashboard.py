
import time
import yaml
import torch
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
import sys
import os
sys.path.append(os.getcwd())

from ml_training.rl.environment import TomatoRipeningEnv
from ml_training.rl.distill import StudentPolicy

# --- Dashboard Config ---
st.set_page_config(
    page_title="Tomato Digital Twin",
    page_icon="🍅",
    layout="wide",
)

# --- Helper Functions ---
@st.cache_resource
def load_model(config_path, model_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    env = TomatoRipeningEnv(config=config)
    
    # Load student policy
    checkpoint = torch.load(model_path)
    state_dim = checkpoint.get("state_dim", env.observation_space.shape[0])
    action_dim = checkpoint.get("action_dim", env.action_space.n)
    hidden_sizes = checkpoint.get("hidden_sizes", [64, 64])
    
    student = StudentPolicy(state_dim, action_dim, hidden_sizes)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()
    
    return env, student

def get_tomato_color_hex(ripeness):
    # Ripeness 1.0 (Green) to 6.0 (Red)
    # Norm 0.0 to 1.0
    norm = np.clip((ripeness - 1.0) / 5.0, 0.0, 1.0)
    
    # Green (0, 255, 0) -> Yellow (255, 255, 0) -> Red (255, 0, 0)
    if norm < 0.5:
        # Green to Yellow
        r = int(255 * (2 * norm))
        g = 255
        b = 0
    else:
        # Yellow to Red
        r = 255
        g = int(255 * (1.0 - 2 * (norm - 0.5)))
        b = 0
        
    return f"#{r:02x}{g:02x}{b:02x}"

# --- UI Layout ---
st.title("🍅 Digital Twin: Tomato Ripening Control")
st.markdown("### AI Agent (Student Policy) Monitoring Dashboard")

# Sidebar
st.sidebar.header("Simulation Controls")
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
speed = st.sidebar.slider("Animation Speed (sec/step)", 0.01, 1.0, 0.1)
init_ripeness = st.sidebar.slider("Initial Ripeness Stage", 0.0, 5.0, 0.0, step=0.1) # New Control
run_btn = st.sidebar.button("Run Simulation", type="primary")

# Load model (Cached)
MODEL_PATH = "outputs/distill_20260213_092014/student_policy.pth"
CONFIG_PATH = "ml_training/config.yaml"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Please run distillation first.")
    st.stop()

env, policy = load_model(CONFIG_PATH, MODEL_PATH)

# Main Area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Tomato State")
    tomato_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.subheader("Environment Telemetry")
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

# --- Simulation Loop ---
if run_btn:
    # Reset and Override Ripeness
    obs, info = env.reset(seed=int(seed))
    
    # NEW: Override initial ripeness based on slider
    env.simulator.ripeness = float(init_ripeness)
    # Recalculate observation after state change
    state = env.simulator.get_state()
    # Need to manually update env internal state too
    env.prev_ripeness = state["ripeness"]
    # Re-generate observation
    obs = env._make_observation(state)
    
    done = False
    total_reward = 0
    
    history = {
        "day": [],
        "temperature": [],
        "humidity": [],
        "ripeness": [],
        "action": []
    }
    
    action_names = ['Rest', 'Heat', 'Cool', 'Harvest']
    
    with st.spinner("Simulating..."):
        while not done:
            # Action
            action = policy.predict(obs)
            if hasattr(policy, 'predict') and not isinstance(action, int) and len(action.shape) > 0:
                 action = int(action[0])
            
            # Step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Get physical state
            state = env.simulator.get_state()
            
            # Update History
            history["day"].append(state["days_elapsed"])
            history["temperature"].append(state["temperature"])
            history["humidity"].append(state["humidity"])
            history["ripeness"].append(state["ripeness"])
            history["action"].append(action)
            
            # 1. Update Tomato Visual
            color = get_tomato_color_hex(state["ripeness"])
            # SVG Circle
            svg = f"""
            <div style="text-align: center;">
                <svg height="200" width="200">
                    <circle cx="100" cy="100" r="90" fill="{color}" stroke="black" stroke-width="3" />
                </svg>
                <h3>{state['ripeness']:.2f} / 6.0</h3>
            </div>
            """
            tomato_placeholder.markdown(svg, unsafe_allow_html=True)
            
            # 2. Update Status Text
            status_text = f"**Current Action:** {action_names[action]}\n\n"
            if action == 3: # Harvest
                status_text += "🚀 **DISPATCHED!**"
            elif action == 1:
                status_text += "🔥 Heating..."
            elif action == 2:
                status_text += "❄️ Cooling..."
            else:
                status_text += "Unchanged"
            status_placeholder.info(status_text)

            # 3. Update Metrics
            action_delta_val = 1.0 # Default delta
            delta_disp = None
            if action == 1:
                delta_disp = f"+{action_delta_val:.1f}"
            elif action == 2:
                delta_disp = f"-{action_delta_val:.1f}"

            with metrics_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Temperature", f"{state['temperature']:.1f} °C", delta=delta_disp)
                c2.metric("Humidity", f"{state['humidity']:.1f} %")
                c3.metric("Reward", f"{total_reward:.2f}")

            # 4. Update Charts
            df = pd.DataFrame(history)
            with chart_placeholder.container():
                st.line_chart(df, x="day", y=["temperature", "ripeness"])
            
            time.sleep(speed)
            
    st.success(f"Simulation Complete! Final Reward: {total_reward:.2f}")
    if action == 3:
        st.balloons()
