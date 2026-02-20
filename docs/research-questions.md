# Research Questions — Detailed Specification

## RQ1: Edge RL Technical Feasibility

> **Can a distilled reinforcement learning policy, trained in simulation and deployed on ESP32-S3 hardware, make harvest timing decisions with sub-2-second total inference latency while maintaining ≥85% ripeness classification accuracy?**

### Hypothesis
Modern neural network quantization (INT8) and policy distillation techniques enable deploying both a vision classifier and an RL decision policy on a $33 ESP32-S3 microcontroller, achieving inference speeds suitable for real-time agricultural monitoring.

### Variables
- **Independent:** Model architecture, quantization level, input resolution
- **Dependent:** Inference latency, classification accuracy, memory usage, power consumption
- **Controlled:** Hardware platform (ESP32-S3 N16R8), dataset (Kaggle + grocery-store)

### Success Criteria
| Metric | Target | Measurement Method |
|---|---|---|
| Classification accuracy | ≥ 85% on grocery-store test set | Confusion matrix, F1 per class |
| Combined inference latency | < 2 seconds | ESP32 timer (`esp_timer_get_time`) |
| Vision model size (INT8) | < 400KB | File size of `.tflite` |
| RL policy size (INT8) | < 50KB | File size of `.tflite` |
| Peak SRAM usage | < 512KB | ESP-IDF `heap_caps_get_info` |
| System uptime | > 90% over 7 days | Log analysis |

### Baselines for Comparison
1. **Cloud inference** — same model on laptop (accuracy upper bound, latency lower bound)
2. **Simpler model** — basic color histogram + threshold (no ML baseline)

---

## RQ2: Rapid Validation Methodology

> **Can transfer learning from public datasets combined with few-shot fine-tuning on commercially available tomatoes enable model validation within an 8-week timeline?**

### Hypothesis
Pre-training on Kaggle tomato images provides a strong feature foundation, and fine-tuning with as few as 60 real images (10 per ripeness stage) captured with the actual deployment camera achieves sufficient accuracy for a usable system.

### Variables
- **Independent:** Pre-training dataset size, fine-tuning image count, learning rate
- **Dependent:** Accuracy on grocery-store test set, training time
- **Controlled:** Model architecture, augmentation pipeline, camera hardware

### Success Criteria
| Metric | Target | Measurement Method |
|---|---|---|
| Kaggle validation accuracy | ≥ 88% | Standard train/val split evaluation |
| Grocery-store test accuracy | ≥ 85% | Held-out grocery images |
| Accuracy drop (Kaggle → real) | < 10 percentage points | Direct comparison |
| Data collection time | < 2 hours total | Time log |
| End-to-end pipeline time | < 1 week | Calendar tracking |

### Ablation Studies
1. **No pre-training** — train from scratch on 60 images only (expected: very poor)
2. **Pre-training only** — no fine-tuning on real images (expected: moderate drop)
3. **Full pipeline** — Kaggle pre-train + few-shot fine-tune (expected: best)

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
