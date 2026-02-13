"""
WebSocket backend for the Digital Twin visualization.

Runs the ACTUAL trained DQN model and Python TomatoRipeningSimulator,
streaming real state to the browser frontend every simulation step.

Usage:
    python digital_twin_viz/server.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import websockets
from stable_baselines3 import DQN

from ml_training.rl.environment import TomatoRipeningEnv
from ml_training.rl.simulator import TomatoRipeningSimulator, SimulatorConfig
import yaml

# ── Load config ──────────────────────────────────────────────────────
CONFIG_PATH = ROOT / "ml_training" / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# ── Load trained DQN ─────────────────────────────────────────────────
MODEL_PATH = None
for p in sorted(ROOT.glob("outputs/rl_*/final_model.zip"), reverse=True):
    MODEL_PATH = p
    break
if MODEL_PATH is None:
    print("ERROR: No trained RL model found in outputs/rl_*/final_model.zip")
    sys.exit(1)

print(f"Loading DQN model from: {MODEL_PATH}")
model = DQN.load(str(MODEL_PATH).replace(".zip", ""))
print("✅ DQN model loaded successfully")

# ── Simulation state ─────────────────────────────────────────────────
env: TomatoRipeningEnv | None = None
obs: np.ndarray | None = None
episode_info: dict = {}
step_count: int = 0
running: bool = False
speed: int = 1
mode: str = "rl"  # "rl", "fixed", "manual"
manual_action: int = 0
episode_history: list[dict] = []
last_q_values: list[float] = [0.0, 0.0, 0.0, 0.0]


def reset_env():
    """Reset the environment and return initial state."""
    global env, obs, episode_info, step_count, running, episode_history
    env = TomatoRipeningEnv(config=config)
    obs, info = env.reset()
    episode_info = info
    step_count = 0
    episode_history = []
    return build_state_msg("reset")


def do_step():
    """Execute one actual simulation step using the real model."""
    global obs, step_count, running, episode_history

    if env is None:
        return build_state_msg("error", error="Environment not initialized")

    # Get action from the ACTUAL trained model or baseline
    global last_q_values
    if mode == "rl":
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        # Extract Q-values from the actual DQN network
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_vals = model.q_net(obs_tensor).squeeze().tolist()
            last_q_values = [float(v) for v in q_vals]
    elif mode == "fixed":
        # Fixed-day baseline: harvest at target day
        days_elapsed = env.simulator.hours_elapsed / 24.0
        if days_elapsed >= env.target_day:
            action = 3
        else:
            action = 0
        last_q_values = [0.0, 0.0, 0.0, 0.0]
    elif mode == "manual":
        action = manual_action
        last_q_values = [0.0, 0.0, 0.0, 0.0]
    else:
        action = 0

    # Step the REAL environment
    obs_new, reward, terminated, truncated, info = env.step(action)
    obs = obs_new
    step_count += 1

    # Record history
    sim = env.simulator
    step_data = {
        "step": step_count,
        "hours": sim.hours_elapsed,
        "days": sim.hours_elapsed / 24.0,
        "ripeness": float(sim.ripeness),
        "temperature": float(sim.temperature),
        "humidity": float(sim.humidity),
        "action": action,
        "reward": float(reward),
    }
    episode_history.append(step_data)

    if terminated or truncated:
        running = False
        return build_state_msg("done", action=action, reward=reward, info=info)

    return build_state_msg("step", action=action, reward=reward, info=info)


def build_state_msg(event: str, **kwargs) -> dict:
    """Build a state message to send to the frontend."""
    if env is None:
        return {"event": event, "error": "no environment"}

    sim = env.simulator
    action_names = ["maintain", "heat", "cool", "harvest"]

    msg = {
        "event": event,
        "step": step_count,
        "hours": round(float(sim.hours_elapsed), 2),
        "days": round(float(sim.hours_elapsed) / 24.0, 2),
        "targetDay": round(float(env.target_day), 1),
        "ripeness": round(float(sim.ripeness), 4),
        "ripenessStage": int(sim.get_ripeness_stage()),
        "temperature": round(float(sim.temperature), 2),
        "humidity": round(float(sim.humidity), 2),
        "quality": round(float(sim.compute_quality_score()), 4),
        "isOverripe": bool(sim.is_overripe()),
        "mode": mode,
        "speed": speed,
        "totalReward": round(float(sum(h["reward"] for h in episode_history)), 2),
        "history": {
            "hours": [float(h["hours"]) for h in episode_history[-200:]],
            "ripeness": [float(h["ripeness"]) for h in episode_history[-200:]],
            "temperature": [float(h["temperature"]) for h in episode_history[-200:]],
            "actions": [int(h["action"]) for h in episode_history[-200:]],
            "rewards": [float(h["reward"]) for h in episode_history[-200:]],
        },
        "observation": [float(v) for v in obs] if obs is not None else [],
        "qValues": last_q_values,
    }

    if "action" in kwargs:
        msg["action"] = action_names[int(kwargs["action"])]
        msg["actionId"] = int(kwargs["action"])
    if "reward" in kwargs:
        msg["reward"] = round(float(kwargs["reward"]), 4)
    if "info" in kwargs:
        info = kwargs["info"]
        if "harvest_quality" in info:
            msg["harvestQuality"] = round(float(info["harvest_quality"]), 4)
        if "timing_error" in info:
            msg["timingError"] = round(float(info["timing_error"]), 2)
    if "error" in kwargs:
        msg["error"] = kwargs["error"]

    return msg


# ── WebSocket handler ────────────────────────────────────────────────
async def handler(websocket):
    """Handle a WebSocket connection from the frontend."""
    global running, speed, mode, manual_action

    print(f"🌐 Client connected: {websocket.remote_address}")

    # Send initial state
    state = reset_env()
    await websocket.send(json.dumps(state))

    async def sim_loop():
        """Run simulation steps while running=True."""
        while running:
            for _ in range(speed):
                state = do_step()
                if state.get("event") in ("done", "error"):
                    await websocket.send(json.dumps(state))
                    return
            await websocket.send(json.dumps(state))
            await asyncio.sleep(0.05)  # ~20 FPS update rate

    sim_task = None

    try:
        async for raw in websocket:
            msg = json.loads(raw)
            cmd = msg.get("cmd")

            if cmd == "start":
                if not running:
                    running = True
                    sim_task = asyncio.create_task(sim_loop())

            elif cmd == "pause":
                running = False
                if sim_task:
                    await sim_task
                    sim_task = None

            elif cmd == "reset":
                running = False
                if sim_task:
                    await sim_task
                    sim_task = None
                state = reset_env()
                await websocket.send(json.dumps(state))

            elif cmd == "step":
                # Single step
                if not running:
                    state = do_step()
                    await websocket.send(json.dumps(state))

            elif cmd == "harvest":
                manual_action = 3
                state = do_step()
                await websocket.send(json.dumps(state))
                running = False

            elif cmd == "set_mode":
                mode = msg.get("mode", "rl")
                print(f"  Mode changed to: {mode}")

            elif cmd == "set_speed":
                speed = max(1, min(20, msg.get("speed", 1)))

            elif cmd == "set_action":
                manual_action = msg.get("action", 0)

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        running = False
        if sim_task:
            sim_task.cancel()
        print(f"🔌 Client disconnected")


async def main():
    port = 8765
    print(f"\n🚀 Digital Twin WebSocket server starting on ws://localhost:{port}")
    print(f"   Model: {MODEL_PATH.name}")
    print(f"   Open the frontend at http://localhost:8080\n")

    async with websockets.serve(handler, "localhost", port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
