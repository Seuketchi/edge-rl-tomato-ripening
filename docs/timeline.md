# 8-Week Implementation Timeline

> Start date assumed: Week 1 of project approval

## Phase Overview

```
Week 1-2  ████████░░░░░░░░  Foundation & Data Pipeline
Week 3-4  ░░░░████████░░░░  ML Training & RL Simulation
Week 5-6  ░░░░░░░░████████  Edge Deployment & Integration
Week 7-8  ░░░░░░░░░░░░████  Validation & Thesis Writing
```

---

## Week 1: Project Setup & Data Collection

**Goal:** Working development environment + Kaggle dataset ready

| Task | Deliverable |
|---|---|
| Set up ESP-IDF development environment | `idf.py build` compiles clean project |
| Order & receive hardware (ESP32-S3 + OV2640 + DHT22) | Components on hand |
| Camera "hello world" — capture & display images | JPEG captures saved to SD/serial |
| Download & organize Kaggle tomato dataset | Dataset in `data/kaggle/` with train/val/test splits |
| Purchase grocery-store tomatoes (all 6 stages) | 10+ images per stage captured with OV2640 |
| Set up PyTorch training environment | Training script runs on laptop/GPU |

## Week 2: Vision Model Training

**Goal:** Quantized vision model achieving ≥85% accuracy

| Task | Deliverable |
|---|---|
| Train MobileNetV2 0.35x on Kaggle dataset | Model with >88% val accuracy |
| Data augmentation pipeline (rotation, flip, color jitter, crop) | Augmentation config file |
| Fine-tune on grocery-store images (few-shot, LR=1e-5) | >85% on grocery-store test set |
| Quantize to INT8 (TFLite or ONNX) | `model.tflite` < 300KB |
| Export as C array for ESP32 embedding | `model_data.h` ready for firmware |
| Evaluate: confusion matrix, per-class accuracy | Thesis figures saved |

## Week 3: Digital Twin & RL Training

**Goal:** Trained RL policy performing well in simulation

| Task | Deliverable |
|---|---|
| Implement ripening physics simulator | `simulator.py` matching dR/dt model |
| Wrap simulator as Gymnasium environment | `TomatoRipeningEnv` class |
| Implement reward function (quality - timing error - energy) | Tuned reward weights |
| Train SAC policy (500K steps) | Trained policy with >80% sim success rate |
| Evaluate against baselines (fixed-rule, random) | Comparison table for thesis |
| Ablation: vary environment parameters | Robustness analysis |

## Week 4: Policy Distillation & Export

**Goal:** Compressed RL policy ready for edge deployment

| Task | Deliverable |
|---|---|
| Distill teacher → student MLP (256×256 → 64×64) | Student retains >95% of teacher performance |
| Quantize student to INT8 | `policy.tflite` ~35KB |
| Export as C array | `policy_data.h` ready for firmware |
| Validate distilled policy in simulation | Performance comparison logged |
| Begin thesis writing (Chapters 1-2) | Draft of Introduction + Literature Review |

## Week 5: Edge Firmware Development

**Goal:** Working inference pipeline on ESP32-S3

| Task | Deliverable |
|---|---|
| Integrate vision model into ESP-IDF firmware | Classification output on serial monitor |
| Integrate RL policy inference | Action recommendations displayed |
| Implement DHT22 sensor reading task | Temperature + humidity logged |
| Implement camera capture task (30-min intervals) | Automated image capture working |
| FreeRTOS task architecture (all 5 tasks running) | Stable multi-task operation |
| Measure inference latency | Latency < 2 seconds confirmed |

## Week 6: System Integration & Initial Validation

**Goal:** End-to-end system running autonomously

| Task | Deliverable |
|---|---|
| Connect all components on breadboard/PCB | Physical prototype assembled |
| MQTT telemetry output (optional, serial is fine) | Telemetry data flowing |
| Run continuous 48-hour stability test | Uptime > 90% logged |
| First validation batch: place tomatoes, run system 5-7 days | Batch 1 data collected |
| Memory profiling & optimization | Memory usage documented |
| Continue thesis writing (Chapter 3: Methodology) | Draft of Methodology complete |

## Week 7: Validation & Analysis

**Goal:** Statistical results for thesis

| Task | Deliverable |
|---|---|
| Second validation batch (5-7 days) | Batch 2 data collected |
| Analyze classification accuracy on real-world images | Confusion matrices, F1 scores |
| Analyze RL policy recommendations vs. expert judgment | Sim-to-real gap quantified |
| System benchmark analysis (latency, memory, power) | Performance tables |
| Generate all thesis figures and tables | Figures saved to `thesis/figures/` |
| Write Chapter 4: Results and Discussion | Draft complete |

## Week 8: Thesis Completion & Defense Prep

**Goal:** Submitted thesis + defense ready

| Task | Deliverable |
|---|---|
| Third validation batch (if time allows) | Batch 3 data (bonus) |
| Write Chapter 5: Conclusions and Recommendations | Final chapter done |
| Abstract, acknowledgments, references | All front/back matter |
| Proofread, format, and submit thesis | PDF submitted |
| Prepare defense slides (15-20 slides) | Slide deck ready |
| Practice defense presentation | Rehearsed 2-3 times |

---

## Critical Path & Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Vision model doesn't fit on ESP32-S3 | Medium | High | Use smaller MobileNetV2 variant; proven on ESP32 |
| RL policy doesn't transfer from sim | Medium | Medium | Have fixed-rule baseline as fallback; still shows edge inference |
| Hardware delivery delays | Low | High | Order in advance; all components are standard |
| Kaggle dataset insufficient quality | Low | Medium | Multiple Kaggle tomato datasets available; combine if needed |
| Grocery-store tomatoes not covering all stages | Medium | Low | Visit multiple stores; stages 1-2 may need produce markets |
| 8-week timeline too tight for 3 batches | High | Low | 2 batches minimum is sufficient for proof-of-concept |

## Minimum Viable Thesis (If Behind Schedule)

If significantly behind by Week 5, the **minimum deliverable** thesis demonstrates:
1. ✅ Vision model running on ESP32-S3 with real-time classification
2. ✅ RL policy trained in simulation with documented performance
3. ✅ At least 1 real-world validation batch
4. ⬜ RL edge deployment (can present simulation results only)
5. ⬜ Multi-batch statistical analysis (single batch is acceptable)

This still satisfies all 3 RQs at a reduced evidence level.
