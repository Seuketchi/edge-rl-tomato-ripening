# Edge-RL Development Roadmap

Based on reviewing the full thesis, codebase, and related literature. Below are concrete directions organized by **effort** and **impact**.

---

## Where You Stand Now

| Component | Status | Evidence |
|---|---|---|
| Digital twin (ODE sim) | ✅ Done | `environment.py`, `simulator.py` |
| DQN teacher training | ✅ Done | 1M steps, reward +4.05 ± 1.48 |
| Policy distillation | ✅ Done | 97.8% fidelity, 30× compression |
| State-space ablation (A/B/C) | ✅ Done | Variant B selected (16D) |
| Domain randomization | ✅ Done | 6 randomized parameters |
| ESP32-S3 pure-C inference | ✅ Done | 7ms latency, 237KB binary |
| Golden vector verification | ✅ Done | 20/20 match |
| On-device ODE simulation | ✅ Done | 3-phase test passing |
| Real tomato hardware test | ❌ Not done | All results are sim-based |
| INT8 quantization | ❌ Mentioned in conclusion | Deployment uses FP32 |

---

## 🟢 Quick Wins (1–2 weeks each)

### 1. Actually Quantize to INT8
Your conclusion claims INT8 but deployment is FP32. Implementing this would:
- Shrink policy from **21.8 KB → ~5.5 KB**
- Potentially halve inference from **7ms → 3-4ms**
- Make the thesis claims fully honest

**How:** Use the `quant_bits: 8` already in `config.yaml`. Add a post-training quantization step in `export_policy_c.py` that converts FP32 weights to INT8 with a scale factor per layer.

**Repo to study:** [tensorflow/model-optimization](https://github.com/tensorflow/model-optimization) or just implement symmetric per-tensor quantization manually (it's ~50 lines of Python).

---

### 2. Try PPO or SAC as Teacher
You only tested DQN. A quick comparison with PPO (also in SB3) would strengthen the ablation and potentially find a better teacher:

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
```

**Why it matters:** Reviewers will ask "why DQN?" — having a comparison table (DQN vs PPO vs SAC) is a strong answer. PPO is generally more stable and may converge faster.

---

### 3. Power Consumption Profiling
You report inference duty cycle (0.0002%) but not **actual power draw**. Measure:
- Idle power (deep sleep between decisions)
- Active inference power (7ms burst)
- Camera capture power
- Heater relay power

This turns a "feasibility" story into an **"autonomous solar-powered deployment"** story.

---

## 🟡 Medium Effort (2–4 weeks each)

### 4. End-to-End Vision Pipeline on ESP32
The pipeline is now based on Direct Pixel feature extraction. This matches the biological setup and uses microsecond logic without needing ESP-DL memory footprint. Make sure that the camera initialization (I2C) and pin allocations are properly resolved by integrating it with real tomatoes.

---

### 5. Real Tomato Validation
The single most impactful thing for thesis defense. Even a small experiment:
- Buy 10 tomatoes at the same ripeness stage
- Run 5 with the Edge-RL system, 5 at room temperature (control)
- Photograph daily, measure actual X (chromatic index)
- Compare harvest timing and quality

**This converts your entire thesis from "simulation study" to "validated system."**

---

### 6. Sim-to-Real Gap Analysis
Use the repos I found to do a proper sim-to-real comparison:

1. Collect real sensor data from the chamber (even without RL control)
2. Replay the sensor traces through your simulator
3. Measure the **reality gap** (predicted X vs actual X)
4. Adjust `k1`, `T_base` to minimize the gap
5. Report the calibrated vs uncalibrated performance

**Repo:** [gabrieletiboni/dropo](https://github.com/gabrieletiboni/dropo) — DROPO can automatically find optimal domain randomization ranges from offline data.

---

## 🔴 Ambitious Extensions (1–3 months)

### 7. Multi-Fruit Batch Control
Currently single-tomato. Real-world value is a **batch of tomatoes at different stages**. This turns the problem into:
- Multi-agent RL (one agent per tomato zone) or
- Single agent with aggregate state (mean/variance of X across batch)

This is a **publishable extension** beyond the thesis.

---

### 8. On-Device Continual Learning
Your conclusion recommends this. The idea: fine-tune the policy on-device as the system observes real tomato behavior.

**Feasible approach for ESP32:**
- Don't do full backprop on-device (too expensive)
- Instead, maintain a small lookup table of state→action corrections
- When the farmer overrides the agent's decision (manual button), log it as a correction
- Periodically upload corrections to a phone app that triggers cloud retraining + OTA update

---

### 9. Transfer to Other Crops
The ODE model `dX/dt = -k1(T - T_base)X` is generic Arrhenius kinetics. By changing `k1` and `T_base`, you can model:
- **Bananas** (k1 ≈ 0.12, very temperature-sensitive)
- **Mangoes** (k1 ≈ 0.06, slower ripening)
- **Avocados** (k1 ≈ 0.04, ethylene-dependent)

**One config change, new crop.** This massively broadens the impact story.

---

### 10. Ethylene Sensor Integration
Your conclusion mentions this. Adding an MQ-3 or SGP30 gas sensor would:
- Provide a **direct biochemical signal** (ethylene = ripening trigger)
- Add 1-2 dimensions to the state space
- Enable detection of **climacteric burst** (the moment ethylene spikes = harvest now)

Cost: ~$5 for the sensor module. Wiring: one ADC pin on the ESP32.

---

## Publication Opportunities

| Venue | Focus | Your Angle |
|---|---|---|
| **IEEE Sensors Letters** | Short paper | Edge deployment + sensor fusion |
| **Computers & Electronics in Agriculture** | Full paper | RL for postharvest management |
| **NeurIPS TinyML Workshop** | Workshop paper | Policy distillation for MCUs |
| **IROS / CoRL** | Conference | Sim-to-real for agricultural robotics |
| **Philippine Computing Conference** | Local | Cost-effective precision agriculture |

---

## Suggested Priority Order

```
1. [Quick]  INT8 quantization         → Validates thesis claim
2. [Quick]  PPO/SAC comparison         → Strengthens ablation
3. [Medium] Real tomato validation     → THE most impactful for defense
4. [Medium] Color histogram on ESP32   → Closes the vision gap cheaply
5. [Quick]  Power profiling            → Enables solar-powered story
6. [Medium] Sim-to-real calibration    → Academic rigor
7. [Ambit.] Multi-fruit batching       → Publishable extension
```
