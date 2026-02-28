# Edge-RL: Tomato Ripening Optimization Thesis

## What This Project Is

An undergraduate Computer Engineering thesis demonstrating **reinforcement learning-based harvest timing optimization on a $33 ESP32-S3 microcontroller** for post-harvest tomato ripening. The novelty is the end-to-end integration of sim-trained RL + direct pixel feature extraction + edge deployment for agriculture.

## Key Constraints

- **Timeline:** 8 weeks total
- **Hardware budget:** ~$33 (ESP32-S3 + OV2640 camera + DHT22 sensor)
- **Target platform:** ESP32-S3-DevKitC-1 (N16R8): 240MHz, 512KB SRAM, 8MB PSRAM, 16MB Flash
- **Thesis level:** Undergraduate — systems integration novelty, not theoretical novelty

## Architecture

```
[Digital Twin Sim] → [SB3: DQN RL training] → [Distill to 64×64 MLP] → [Export to C header] ──┐
                                                                                                  ├→ [ESP32-S3 Firmware]
ESP32-S3 runs: OV2640 → Direct Pixel Statistics (RGB mean/std/mode + Chromatic Index X) ────────┘
                         → RL Policy (16D→64→64→3 MLP) → Action (maintain/heat/cool)
```

**Vision Module:** No CNN/ML model for vision. The camera captures an RGB frame, a 60% centre crop isolates the fruit, and RGB statistics (mean, std, mode) + Chromatic Index `X = G/(R+G)` are computed directly from pixel arrays. This guarantees perfect consistency between simulator observations and real hardware.

## Repository Structure

```
thesis-it/
├── CLAUDE.md              ← You are here
├── docs/                  ← Source of truth for thesis scope & design
│   ├── thesis-overview.md   # Refined thesis idea, novelty, RQs (START HERE)
│   ├── architecture.md      # System architecture, memory maps, tasks
│   ├── technical-stack.md   # Hardware, firmware, ML tools
│   ├── timeline.md          # 8-week schedule with deliverables
│   └── research-questions.md # Detailed RQ specs, success criteria, ablations
├── edge_firmware/         ← ESP-IDF C firmware (ESP32-S3)
├── ml_training/           ← Python ML pipeline
│   ├── vision/              # (Legacy — not used in deployment; direct pixel extraction replaced CNN)
│   ├── rl/                  # DQN training, digital twin, distillation
│   └── config.yaml          # Global hyperparameters
├── digital_twin_viz/      ← Web-based visualization dashboard
├── tests/                 ← Test protocols
└── docs/thesis/           ← LaTeX thesis document
```

## Tech Stack Quick Reference

| Component | Choice | Why |
|---|---|---|
| Vision pipeline | Direct pixel RGB statistics + Chromatic Index | Microsecond computation, zero covariate shift vs simulator |
| RL algorithm | DQN (Deep Q-Network) | Simple, discrete actions, well-suited for 3-action space |
| RL library | Stable Baselines3 | Mature, well-documented |
| Edge framework | ESP-IDF v5.1+ | Official Espressif SDK |
| Distilled policy | 16→64→64→3 MLP (5,443 params, ~21.8 KB) | Pure C inference, no ML runtime needed |
| Training | PyTorch | Standard pipeline |
| Language (edge) | C (C11) | Required by ESP-IDF |
| Language (training) | Python 3.10+ | Standard ML |

## Coding Conventions

### Python (ML Training)
- Use type hints for function signatures
- Docstrings for all public functions (Google style)
- Config via YAML files, not hardcoded values
- Save all experiment results to `outputs/` with timestamps
- Use `pathlib.Path` for file paths

### C (ESP32 Firmware)
- Follow ESP-IDF coding style
- Prefix all custom functions with `edge_rl_`
- Use `ESP_LOG*` macros for logging (not `printf`)
- All tasks must have watchdog integration
- Magic numbers → `#define` constants in `app_config.h`

### General
- Commit messages: `type(scope): description` (e.g., `feat(rl): add distillation pipeline`)
- Branch naming: `feature/`, `fix/`, `docs/`
- All model artifacts go in `outputs/models/` with version suffixes

## Research Questions (Summary)

1. **RQ1 — Edge RL Feasibility:** Can distilled RL + direct pixel extraction run on ESP32-S3 with <2s latency and ≥85% accuracy?
2. **RQ2 — Rapid Validation:** Can simulation-based training + 5–10 real tomato trials enable validation in 8 weeks?
3. **RQ3 — Sim-to-Real Transfer:** Does sim-trained RL policy produce reasonable recommendations with real sensors?

## What Was Dropped (Future Work)

- **Causal inference (NOTEARS)** — too complex for 8-week timeline
- **CNN-based vision (MobileNetV2/EdgeViT)** — direct pixel computation is faster, simpler, and avoids covariate shift
- **Firebase cloud backend** — unnecessary overhead; serial/MQTT monitoring suffices
- **Prototypical networks** — simple fine-tuning is sufficient

## Important Files to Know About

- `docs/thesis-overview.md` — Start here for the full thesis design
- `docs/timeline.md` — Week-by-week implementation plan
- `docs/research-questions.md` — Detailed success criteria
- `TODO.md` — Master task checklist with current project status
- `edge_firmware/main/app_config.h` — All firmware configuration constants
