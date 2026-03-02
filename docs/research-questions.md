# Research Questions — Detailed Specification

## RQ1: Edge RL Technical Feasibility

> **Can a distilled reinforcement learning policy, trained in simulation and deployed on ESP32-S3 hardware, make harvest timing decisions with sub-2-second total inference latency while maintaining ≥85% ripeness classification accuracy?**

### Hypothesis
Policy distillation and direct pixel feature extraction enable deploying an RL decision policy on a $33 ESP32-S3 microcontroller with pure C inference (no ML runtime), achieving speeds suitable for real-time agricultural monitoring.

### Variables
- **Independent:** Policy architecture (teacher/student size), state variant (A/B/C)
- **Dependent:** Inference latency, distillation fidelity, memory usage, power consumption
- **Controlled:** Hardware platform (ESP32-S3 N16R8), vision pipeline (direct pixel stats)

### Success Criteria
| Metric | Target | Measurement Method |
|---|---|---|
| Distillation fidelity | ≥ 95% action agreement | Student vs teacher on rollout data |
| Combined inference latency | < 2 seconds | ESP32 timer (`esp_timer_get_time`) |
| Vision pipeline size | < 2KB code | Source file size (no model weights) |
| RL policy size (FP32) | < 25KB | C header array size |
| Peak SRAM usage | < 512KB | ESP-IDF `heap_caps_get_info` |
| Golden vector test | 20/20 pass | On-device verification at boot |

### Baselines for Comparison
1. **Fixed-Stage5** — harvest when X ≤ threshold (rule-based, no RL)
2. **Fixed-Day** — maintain temperature, let natural ripening occur
3. **Random** — random action selection

---

## RQ2: Rapid Validation Methodology

> **Can simulation-based RL training combined with 5–10 real tomato trials enable system validation within an 8-week timeline?**

### Hypothesis
A physics-based digital twin simulator with domain randomization provides sufficient training data for RL policy learning, and a small number of real-world trials (5–10 fruits) suffices to validate the sim-to-real pipeline for an undergraduate proof-of-concept.

### Variables
- **Independent:** Simulator fidelity, domain randomization range, number of real trials
- **Dependent:** Sim-to-real transfer quality, validation timeline feasibility
- **Controlled:** Hardware platform, vision pipeline (direct pixel stats), RL algorithm (DQN)

### Success Criteria
| Metric | Target | Measurement Method |
|---|---|---|
| Simulation training convergence | > 80% success rate | Mean episode reward |
| Ablation variants trained | ≥ 3 (A/B/C state variants) | Training logs |
| Real trial completion | 5–10 fruits | Experiment log |
| End-to-end pipeline time | < 8 weeks | Calendar tracking |
| Hardware assembly + validation | < 2 weeks | Calendar tracking |

### Ablation Studies
1. **State Variant A (7D)** — minimal state: X, dX/dt, X_ref, T, H, t_e, t_rem
2. **State Variant B (16D)** — A + RGB statistics (mean/std/mode)
3. **State Variant C (20D)** — B + max-pool spatial features

---

## RQ3: Sim-to-Real Policy Transfer

> **Does the RL policy trained entirely in a physics-based digital twin produce reasonable harvest timing recommendations when deployed with real sensor data?**

### Hypothesis
A calibrated physics-based ripening simulator with domain randomization produces RL policies that generalize to real-world sensor inputs, demonstrating that simulation-only training is a viable approach for agricultural RL.

### Variables
- **Independent:** Simulator parameters, domain randomization range, training steps
- **Dependent:** Policy action quality, sim-to-real gap, harvest timing accuracy
- **Controlled:** RL algorithm (DQN), reward function, policy architecture

### Success Criteria
| Metric | Target | Measurement Method |
|---|---|---|
| Simulation success rate | > 80% | % episodes where $X \leq 0.15$ (ripe) before deadline |
| Real-world plausibility | Actions match expert judgment in >75% of cases | Expert evaluation checklist |
| Policy vs. fixed baseline | Outperforms fixed-heat-only and fixed-maintain baselines | Average total reward comparison |
| Sim-to-real performance gap | < 20% | Compare sim metrics to real deployment |

### Evaluation Protocol
1. Run system with real tomatoes for 5-7 day batches
2. Record RL policy recommendations at each decision point
3. Compare to:
   - What an expert would recommend (researcher's judgment)
   - Fixed-rule baseline outputs (e.g., heat-only, maintain-only)
   - Simulator predictions for same initial conditions
4. Quantify gap between simulated expected outcomes and actual observed outcomes

---

## Dropped Research Question (Future Work)

> ~~**RQ (Original):** Does incorporating causal structure learning (NOTEARS) versus purely correlational ML improve cross-environment generalization?~~

### Why Dropped
- Causal inference is a separate, deep research track
- NOTEARS implementation + validation adds 3-4 weeks minimum
- Difficult to demonstrate meaningful results with only 2-3 tomato batches
- Cross-environment generalization requires multiple distinct environments to test
- Does not directly contribute to the core "edge RL" system novelty

### Future Work Framing
> "Future research could investigate whether incorporating causal structure learning improves cross-environment generalization of ripening models. Preliminary analysis suggests that learning causal relationships between temperature and ripening rate (rather than purely correlational patterns) may enable better zero-shot transfer to unseen deployment conditions."
