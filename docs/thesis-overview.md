# Edge-RL: Reinforcement Learning-Enhanced Edge Intelligence for Post-Harvest Tomato Ripening Optimization

> Undergraduate Computer Engineering Thesis — 8-Week Timeline

## Problem Statement

Post-harvest losses account for 20-40% of tomato production in developing countries. Commercial ripening facilities solve this with $10,000-$50,000 infrastructure, but small-scale farmers cannot afford this. Existing IoT solutions provide *monitoring* but not *intelligent decision-making* — they tell you the temperature but can't tell you *when to harvest given a target delivery date*.

## Thesis Novelty

**Core Novelty: First demonstration of a complete sim-to-edge RL pipeline for agricultural post-harvest optimization on a sub-$50 microcontroller.**

This thesis combines three things that have individually been explored but never integrated together:

1. **Edge-deployed RL policy inference** — RL for agriculture exists (greenhouse climate control), but only on cloud/server hardware. Deploying a distilled RL policy on an ESP32-S3 MCU for agricultural decision-making is novel.

2. **Sim-to-real transfer for post-harvest** — While sim-to-real is established in robotics, applying it specifically to post-harvest ripening physics (with a calibrated digital twin) is unexplored.

3. **Grocery-store rapid validation protocol** — A methodological contribution enabling agricultural AI validation without growing seasons, using transfer learning from public Kaggle datasets + few-shot fine-tuning on store-bought tomatoes.

### What Makes This an Undergraduate-Appropriate Novelty

- It's a **systems integration novelty** — combining known techniques (RL, quantization, edge deployment, transfer learning) in a new application domain and demonstrating end-to-end feasibility
- It produces **measurable, demonstrable results** within 8 weeks
- The individual components use established libraries (Stable Baselines3, ESP-IDF, PyTorch)
- The hardware cost is low ($33), making it reproducible

## Refined Research Questions

### RQ1: Edge RL Feasibility
> Can a distilled reinforcement learning policy, trained in simulation and deployed on ESP32-S3 hardware, make harvest timing decisions with sub-2-second total inference latency while maintaining ≥85% classification accuracy?

**Measurable Outcomes:**
- Inference latency < 2 seconds (classification + RL policy combined)
- 6-stage ripeness classification accuracy ≥ 85% on grocery-store test set
- System uptime > 90% over continuous 7-day operation
- Total hardware cost < $50

### RQ2: Rapid Validation Methodology
> Can transfer learning from public datasets combined with few-shot fine-tuning on commercially available tomatoes enable model validation within an 8-week timeline?

**Measurable Outcomes:**
- Achieve ≥ 85% accuracy using Kaggle pre-training + 60 real images (10/class)
- Complete 3 validation batches within project timeline
- Demonstrate < 10% accuracy drop from Kaggle validation to grocery-store test

### RQ3: Sim-to-Real Policy Transfer *(Simplified from Causal Inference)*
> Does the RL policy trained entirely in a physics-based digital twin produce reasonable harvest timing recommendations when deployed with real sensor data?

**Measurable Outcomes:**
- RL policy recommends harvest within ±2 days of human expert judgment
- Policy outperforms fixed-rule baselines (e.g., "harvest at stage 5")
- Sim-to-real gap in policy performance < 20%

> [!IMPORTANT]
> **Dropped from original scope:** The causal inference component (NOTEARS-based causal discovery) was removed. While intellectually interesting, it adds significant complexity (separate research track), is hard to validate in 8 weeks, and doesn't directly serve the core systems-integration novelty. It can be listed as "future work."

## Key Simplifications from Original Documents

| Original Scope | Refined Scope | Rationale |
|---|---|---|
| EdgeViT (280KB vision transformer) | MobileNetV2-tiny or EfficientNet-Lite0 (quantized) | EdgeViT on ESP32-S3 is risky; MobileNet family has proven ESP32 support |
| Causal inference (NOTEARS) | Dropped → future work | Separate research track, hard to validate in 8 weeks |
| 7-chapter thesis structure | 5-chapter structure | Standard undergrad format, less writing overhead |
| Prototypical networks for few-shot | Simple fine-tuning with small LR | Prototypical networks add complexity without clear benefit here |
| Cloud-based RL training + Firebase dashboard | Local RL training + simple serial/MQTT monitoring | Cloud infra is overhead; thesis focuses on edge intelligence |
| 3 complete tomato batches | 2-3 batches (2 minimum) | Reduces timeline risk |

## Thesis Chapter Outline (5 Chapters)

1. **Introduction** — Problem, RQs, contributions, scope
2. **Review of Related Literature** — Edge AI in agriculture, RL for agricultural control, post-harvest science, sim-to-real transfer
3. **Methodology** — System architecture, ML pipeline, RL training, validation protocol
4. **Results and Discussion** — Classification performance, RL policy evaluation, system benchmarks, sim-to-real gap analysis
5. **Conclusions and Recommendations** — Summary, limitations, future work (including causal inference direction)
