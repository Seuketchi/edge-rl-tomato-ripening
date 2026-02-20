# Edge-RL: Tomato Ripening Optimization Thesis

## What This Project Is

An undergraduate Computer Engineering thesis demonstrating **reinforcement learning-based harvest timing optimization on a $33 ESP32-S3 microcontroller** for post-harvest tomato ripening. The novelty is the end-to-end integration of sim-trained RL + quantized vision model + edge deployment for agriculture.

## Key Constraints

- **Timeline:** 8 weeks total
- **Hardware budget:** ~$33 (ESP32-S3 + OV2640 camera + DHT22 sensor)
- **Target platform:** ESP32-S3-DevKitC-1 (N16R8): 240MHz, 512KB SRAM, 8MB PSRAM, 16MB Flash
- **Thesis level:** Undergraduate — systems integration novelty, not theoretical novelty

## Architecture

```
[Kaggle Dataset] → [PyTorch: MobileNetV2 training] → [INT8 Quantize] ──┐
                                                                         ├→ [ESP32-S3 Firmware]
[Digital Twin Sim] → [SB3: DQN RL training] → [Distill + INT8] ────────┘
                                                                         │
ESP32-S3 runs: OV2640 → Vision Model → RL Policy → Action (maintain/heat/cool)
```

## Repository Structure

```
thesis-it/
├── claude.md              ← You are here
├── docs/                  ← Source of truth for thesis scope & design
│   ├── thesis-overview.md   # Refined thesis idea, novelty, RQs (START HERE)
│   ├── architecture.md      # System architecture, memory maps, tasks
│   ├── technical-stack.md   # Hardware, firmware, ML tools, model options
│   ├── timeline.md          # 8-week schedule with deliverables
│   └── research-questions.md # Detailed RQ specs, success criteria, ablations
├── edge_firmware/         ← ESP-IDF C firmware (ESP32-S3)
├── ml_training/           ← Python ML pipeline
│   ├── vision/              # MobileNetV2 training, quantization
│   ├── rl/                  # DQN training, digital twin, distillation
│   └── data/                # Dataset management
├── validation/            ← Test protocols and analysis
├── hardware/              ← Schematics, BOM
└── thesis/                ← LaTeX/Word thesis document
```

## Tech Stack Quick Reference

| Component | Choice | Why |
|---|---|---|
| Vision model | MobileNetV2 0.35x (INT8) | Proven on ESP32, ~200KB |
| RL algorithm | DQN (Deep Q-Network) | Simple, discrete actions, well-suited for 3-action space |
| RL library | Stable Baselines3 | Mature, well-documented |
| Edge framework | ESP-IDF v5.1+ | Official Espressif SDK |
| ML runtime | ESP-DL v3.x + esp-ppq | ~10x faster than TFLite Micro on ESP32-S3 |
| Training | PyTorch + TFLite converter | Standard pipeline |
| Language (edge) | C (C11) | Required by ESP-IDF |
| Language (training) | Python 3.10+ | Standard ML |

## Coding Conventions

### Python (ML Training)
- Use type hints for function signatures
- Docstrings for all public functions (Google style)
- Config via YAML files, not hardcoded values
- Save all experiment results to `results/` with timestamps
- Use `pathlib.Path` for file paths

### C (ESP32 Firmware)
- Follow ESP-IDF coding style
- Prefix all custom functions with `edge_rl_`
- Use `ESP_LOG*` macros for logging (not `printf`)
- All tasks must have watchdog integration
- Magic numbers → `#define` constants in `app_config.h`

### General
- Commit messages: `type(scope): description` (e.g., `feat(vision): add MobileNetV2 training script`)
- Branch naming: `feature/`, `fix/`, `docs/`
- All model artifacts go in `models/` with version suffixes

## Research Questions (Summary)

1. **RQ1 — Edge RL Feasibility:** Can distilled RL + quantized vision run on ESP32-S3 with <2s latency and ≥85% accuracy?
2. **RQ2 — Rapid Validation:** Can Kaggle pre-training + 60 real images enable validation in 8 weeks?
3. **RQ3 — Sim-to-Real Transfer:** Does sim-trained RL policy produce reasonable recommendations with real sensors?

## What Was Dropped (Future Work)

- **Causal inference (NOTEARS)** — too complex for 8-week timeline
- **EdgeViT** — risky on ESP32; using proven MobileNetV2 instead
- **Firebase cloud backend** — unnecessary overhead; serial/MQTT monitoring suffices
- **Prototypical networks** — simple fine-tuning is sufficient

## Important Files to Know About

- `docs/thesis-overview.md` — Start here for the full thesis design
- `docs/timeline.md` — Week-by-week implementation plan
- `docs/research-questions.md` — Detailed success criteria
- `Edge_RL_Chapter1_Research_Paper.docx` — Original Chapter 1 draft (reference only)
- `Edge_RL_Technical_Stack.docx` — Original tech stack doc (reference only)
