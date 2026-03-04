"""
WebSocket backend for the Digital Twin visualization — multi-agent edition.

Supports:
  - Single-agent mode:  any one policy (DQN/PPO/A2C/baselines) + any variant (A/B/C)
  - Compare mode:       multiple agents run on the same seed in lockstep

Usage:
    python digital_twin_viz/server.py
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import websockets
from stable_baselines3 import DQN, PPO, A2C
import yaml

from ml_training.rl.environment import TomatoRipeningEnv

# ── Agent catalogue ───────────────────────────────────────────────────
_OUT = ROOT / "outputs"

AGENT_DEFS: dict[str, dict] = {
    "dqn": {
        "label": "DQN", "algo": "dqn",
        "path": _OUT / "rl_20260303_174001" / "best_model" / "best_model.zip",
        "variant": "B", "color": "#2a7a3b",
    },
    "ppo": {
        "label": "PPO", "algo": "ppo",
        "path": _OUT / "algo_comparison_20260226_210137" / "ppo" / "best_model" / "best_model.zip",
        "variant": "B", "color": "#c84b38",
    },
    "a2c": {
        "label": "A2C", "algo": "a2c",
        "path": _OUT / "algo_comparison_20260226_210137" / "a2c" / "best_model" / "best_model.zip",
        "variant": "B", "color": "#3a6abf",
    },
    "dqn_a": {
        "label": "DQN Var-A", "algo": "dqn",
        "path": _OUT / "rl_20260303_204723" / "best_model" / "best_model.zip",
        "variant": "A", "color": "#c8a838",
    },
    "dqn_c": {
        "label": "DQN Var-C", "algo": "dqn",
        "path": _OUT / "rl_20260303_212521" / "best_model" / "best_model.zip",
        "variant": "C", "color": "#7a4a8a",
    },
    "fixed_day": {
        "label": "Fixed-Day", "algo": "baseline", "policy": "fixed_day",
        "variant": "B", "color": "#888888",
    },
    "fixed_stage5": {
        "label": "Fixed-Stage5", "algo": "baseline", "policy": "fixed_stage5",
        "variant": "B", "color": "#e8883a",
    },
    "random": {
        "label": "Random", "algo": "baseline", "policy": "random",
        "variant": "B", "color": "#aaaaaa",
    },
}

# ── Load all ML models at startup ─────────────────────────────────────
_models: dict[str, Any] = {}
_CLS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}

print("\nLoading models...")
for _aid, _defn in AGENT_DEFS.items():
    if _defn["algo"] == "baseline":
        continue
    _path = Path(_defn["path"])
    if not _path.exists():
        print(f"  WARNING: {_aid} — model not found ({_path.parent.parent.name})")
        continue
    _models[_aid] = _CLS[_defn["algo"]].load(str(_path).replace(".zip", ""))
    print(f"  ✅ {_aid} ({_defn['algo'].upper()} {_defn['variant']}): {_path.parent.parent.name}")
print(f"Models ready: {list(_models.keys())}\n")

with open(ROOT / "ml_training" / "config.yaml") as f:
    _BASE_CONFIG = yaml.safe_load(f)

ACTION_NAMES = ["maintain", "heat", "cool"]


# ── Per-connection session ────────────────────────────────────────────
class SimSession:
    """All simulation state for one WebSocket connection."""

    def __init__(self) -> None:
        self.running = False
        self.speed = 3
        self.compare_mode = False
        self.active_agent = "dqn"          # single mode
        self.compare_agents: list[str] = ["dqn", "ppo", "a2c"]  # compare mode
        self.manual_action = 0
        self.shared_seed = 42

        # Single-agent state
        self.env: TomatoRipeningEnv | None = None
        self.obs: np.ndarray | None = None
        self.step_count = 0
        self.episode_history: list[dict] = []
        self.last_q_values: list[float] = [0.0, 0.0, 0.0]

        # Compare state
        self.cstates: dict[str, dict] = {}

    # ── Reset ─────────────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> dict:
        seed = seed if seed is not None else random.randint(0, 99999)
        self.shared_seed = seed
        return self._reset_compare(seed) if self.compare_mode else self._reset_single(seed)

    def _reset_single(self, seed: int) -> dict:
        defn = AGENT_DEFS[self.active_agent]
        self.env = TomatoRipeningEnv(config=_BASE_CONFIG, state_variant=defn["variant"], seed=seed)
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.episode_history = []
        self.last_q_values = [0.0, 0.0, 0.0]
        return self._single_msg("reset")

    def _reset_compare(self, seed: int) -> dict:
        self.cstates = {}
        for aid in self.compare_agents:
            defn = AGENT_DEFS[aid]
            env = TomatoRipeningEnv(config=_BASE_CONFIG, state_variant=defn["variant"], seed=seed)
            obs, _ = env.reset()
            self.cstates[aid] = {
                "env": env, "obs": obs, "done": False,
                "step": 0, "total_reward": 0.0,
                "history": [], "final_info": None,
            }
        return self._compare_msg("compare_reset")

    # ── Step ──────────────────────────────────────────────────────────
    def step(self) -> dict:
        return self._step_compare() if self.compare_mode else self._step_single()

    def _get_action(self, aid: str, obs: np.ndarray) -> tuple[int, list[float]]:
        defn = AGENT_DEFS[aid]
        q_vals = [0.0, 0.0, 0.0]

        if defn["algo"] == "baseline":
            pol = defn["policy"]
            if pol == "fixed_day":
                action = 0
            elif pol == "fixed_stage5":
                action = 1 if float(obs[0]) > 0.3 else 0
            else:  # random
                action = random.randint(0, 2)
            return action, q_vals

        model = _models.get(aid)
        if model is None:
            return 0, q_vals
        act, _ = model.predict(obs, deterministic=True)
        action = int(act)
        if defn["algo"] == "dqn":
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_vals = [float(v) for v in model.q_net(t).squeeze().tolist()]
        return action, q_vals

    def _step_single(self) -> dict:
        if self.env is None:
            return {"event": "error", "error": "No environment"}

        if self.active_agent == "manual":
            action, q_vals = self.manual_action, [0.0, 0.0, 0.0]
        else:
            action, q_vals = self._get_action(self.active_agent, self.obs)
        self.last_q_values = q_vals

        obs_new, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs_new
        self.step_count += 1
        sim = self.env.simulator
        self.episode_history.append({
            "hours": sim.hours_elapsed,
            "ripeness": float(sim.ripeness),
            "temperature": float(sim.temperature),
            "humidity": float(sim.humidity),
            "action": action,
            "reward": float(reward),
        })
        if terminated or truncated:
            self.running = False
            return self._single_msg("done", action=action, reward=reward, info=info)
        return self._single_msg("step", action=action, reward=reward, info=info)

    def _step_compare(self) -> dict:
        all_done = True
        for aid, cs in self.cstates.items():
            if cs["done"]:
                continue
            all_done = False
            action, _ = self._get_action(aid, cs["obs"])
            obs_new, reward, terminated, truncated, info = cs["env"].step(action)
            cs["obs"] = obs_new
            cs["step"] += 1
            cs["total_reward"] += float(reward)
            sim = cs["env"].simulator
            cs["history"].append({
                "hours": sim.hours_elapsed,
                "ripeness": float(sim.ripeness),
                "temperature": float(sim.temperature),
                "action": action,
                "reward": float(reward),
            })
            if terminated or truncated:
                cs["done"] = True
                cs["final_info"] = info

        if all_done:
            self.running = False
            return self._compare_msg("compare_done")
        return self._compare_msg("compare_step")

    # ── Message builders ──────────────────────────────────────────────
    def _single_msg(self, event: str, **kw) -> dict:
        if self.env is None:
            return {"event": event, "error": "no env"}
        sim = self.env.simulator
        defn = AGENT_DEFS.get(self.active_agent, {})
        hist = self.episode_history

        msg: dict = {
            "event": event, "mode": "single",
            "agentId": self.active_agent,
            "agentLabel": defn.get("label", self.active_agent),
            "agentColor": defn.get("color", "#888"),
            "step": self.step_count,
            "hours": round(float(sim.hours_elapsed), 2),
            "days": round(float(sim.hours_elapsed) / 24.0, 2),
            "targetDay": round(float(self.env.target_day), 1),
            "ripeness": round(float(sim.ripeness), 4),
            "ripenessStage": int(min(5, int((1.0 - float(sim.ripeness)) * 6))),
            "temperature": round(float(sim.temperature), 2),
            "humidity": round(float(sim.humidity), 2),
            "hourOfDay": round(sim.hours_elapsed % 24.0, 1),
            "quality": round(float(sim.compute_quality_score()), 4),
            "isOverripe": bool(float(sim.ripeness) < 0.05),
            "speed": self.speed,
            "totalReward": round(float(sum(h["reward"] for h in hist)), 2),
            "history": {
                "hours": [h["hours"] for h in hist[-200:]],
                "ripeness": [h["ripeness"] for h in hist[-200:]],
                "temperature": [h["temperature"] for h in hist[-200:]],
                "humidity": [h["humidity"] for h in hist[-200:]],
                "actions": [h["action"] for h in hist[-200:]],
                "rewards": [h["reward"] for h in hist[-200:]],
            },
            "observation": [float(v) for v in self.obs] if self.obs is not None else [],
            "qValues": self.last_q_values,
        }
        if "action" in kw:
            msg["action"] = ACTION_NAMES[int(kw["action"])]
            msg["actionId"] = int(kw["action"])
        if "reward" in kw:
            msg["reward"] = round(float(kw["reward"]), 4)
        if "info" in kw:
            info = kw["info"]
            if "harvest_quality" in info:
                msg["harvestQuality"] = round(float(info["harvest_quality"]), 4)
            if "timing_error" in info:
                msg["timingError"] = round(float(info["timing_error"]), 2)
        return msg

    def _compare_msg(self, event: str) -> dict:
        agents_out: dict[str, dict] = {}
        target_day = None

        for aid, cs in self.cstates.items():
            defn = AGENT_DEFS[aid]
            sim = cs["env"].simulator
            if target_day is None:
                target_day = round(float(cs["env"].target_day), 1)
            hist = cs["history"][-200:]
            last_action = int(hist[-1]["action"]) if hist else 0
            entry: dict = {
                "label": defn["label"],
                "color": defn["color"],
                "step": cs["step"],
                "days": round(sim.hours_elapsed / 24.0, 2),
                "ripeness": round(float(sim.ripeness), 4),
                "temperature": round(float(sim.temperature), 2),
                "action": last_action,
                "totalReward": round(cs["total_reward"], 2),
                "done": cs["done"],
                "history": {
                    "hours": [h["hours"] for h in hist],
                    "ripeness": [h["ripeness"] for h in hist],
                    "temperature": [h["temperature"] for h in hist],
                    "actions": [h["action"] for h in hist],
                },
            }
            if cs["done"] and cs["final_info"]:
                fi = cs["final_info"]
                entry["harvestQuality"] = round(float(fi.get("harvest_quality", 0)), 4)
                entry["timingError"] = round(float(fi.get("timing_error", 0)), 2)
                entry["harvestDay"] = round(sim.hours_elapsed / 24.0, 2)
            agents_out[aid] = entry

        return {
            "event": event, "mode": "compare",
            "targetDay": target_day,
            "seed": self.shared_seed,
            "agents": agents_out,
            "speed": self.speed,
        }

    # ── Catalogue helper (sent to frontend on connect) ─────────────────
    @staticmethod
    def catalogue_msg() -> dict:
        return {
            "event": "catalogue",
            "agents": {
                aid: {
                    "label": d["label"],
                    "color": d["color"],
                    "variant": d["variant"],
                    "available": d["algo"] == "baseline" or aid in _models,
                }
                for aid, d in AGENT_DEFS.items()
            },
        }


# ── WebSocket handler ─────────────────────────────────────────────────
async def handler(websocket):
    print(f"  Client connected: {websocket.remote_address}")
    session = SimSession()

    # Send catalogue + initial state
    await websocket.send(json.dumps(SimSession.catalogue_msg()))
    state = session.reset()
    await websocket.send(json.dumps(state))

    sim_task: asyncio.Task | None = None

    async def sim_loop():
        while session.running:
            for _ in range(session.speed):
                s = session.step()
                ev = s.get("event", "")
                if ev in ("done", "compare_done", "error"):
                    await websocket.send(json.dumps(s))
                    return
            await websocket.send(json.dumps(s))
            await asyncio.sleep(0.05)

    try:
        async for raw in websocket:
            msg = json.loads(raw)
            cmd = msg.get("cmd")

            if cmd == "start":
                if not session.running:
                    session.running = True
                    sim_task = asyncio.create_task(sim_loop())

            elif cmd == "pause":
                session.running = False
                if sim_task:
                    await sim_task
                    sim_task = None

            elif cmd == "reset":
                session.running = False
                if sim_task:
                    await sim_task
                    sim_task = None
                seed = msg.get("seed")  # optional pinned seed
                state = session.reset(seed)
                await websocket.send(json.dumps(state))

            elif cmd == "step":
                if not session.running:
                    s = session.step()
                    await websocket.send(json.dumps(s))

            elif cmd == "set_speed":
                session.speed = max(1, min(20, int(msg.get("speed", 3))))

            elif cmd == "set_action":
                session.manual_action = int(msg.get("action", 0))

            elif cmd == "set_single":
                # Switch to single-agent mode with given agent
                session.running = False
                if sim_task:
                    sim_task.cancel()
                    sim_task = None
                session.compare_mode = False
                session.active_agent = msg.get("agent", "dqn")
                state = session.reset()
                await websocket.send(json.dumps(state))

            elif cmd == "set_compare":
                # Switch to compare mode with given agent list
                session.running = False
                if sim_task:
                    sim_task.cancel()
                    sim_task = None
                session.compare_mode = True
                session.compare_agents = msg.get("agents", ["dqn", "ppo", "a2c"])
                state = session.reset()
                await websocket.send(json.dumps(state))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        session.running = False
        if sim_task:
            sim_task.cancel()
        print(f"  Client disconnected")


async def main():
    port = 8765
    print(f"\n  Digital Twin WebSocket server on ws://localhost:{port}")
    print(f"  Open: digital_twin_viz/index.html\n")
    async with websockets.serve(handler, "0.0.0.0", port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
