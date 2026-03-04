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

## Current Project Stage

**Stage 5 — Model Finalization + Hardware Assembly** (as of March 2026). Stages 1–4 complete. Config regression fixed (k1_variation=0.008, ambient_temp_std=2.0). DQN retrain in progress (outputs/rl_20260303_*/). After retrain: distill → export C headers → recompile firmware → assemble prototype → Stage 6 (real tomato experiments).

## "Fix this" Shorthand

When the user says **"fix this"**, work through this prioritized list:

1. **Dashboard 4→3 action mismatch** — `digital_twin_viz/index.html` still shows a 4th "Harvest 🔪" Q-value bar and labels vision as "MobileNetV2 (0.35×)". Should be 3 actions and "Direct Pixel Statistics".
2. **After retrain: update thesis numbers** — Tables 2+3 in `docs/thesis/chapters/04_results.tex` and action distribution text (0.2%/6.5%/93.2%) — update to new eval JSON. Sync `thesis_report.tex` and `methodology_and_results.tex` which currently have stale 31.4%/43.5%/25.1%.
3. **After retrain: distill → export → recompile** — `python -m ml_training.rl.distill ... && python ml_training/rl/export_policy_c.py --verify && cd edge_firmware && idf.py build`
4. **Confidence-based action gating** — In firmware, if max softmax confidence < 0.6, default to MAINTAIN. ~10 lines of C in `policy_task.c`.
5. **k₁ online adaptation** — Exponential moving average of observed dX/dt on ESP32 to adapt the ripening rate constant. Addresses RQ3 sim-to-real transfer. Only if time permits.

## How to Run Things

### Python Environment
```bash
source .venv/bin/activate          # Python 3.13.11, plain venv
pip install -r requirements.txt    # torch, stable-baselines3, gymnasium, etc.
```

### ML Training (all from project root)
```bash
# DQN Training
python -m ml_training.rl.train_dqn --config ml_training/config.yaml

# DQN smoke test (fast verification)
python -m ml_training.rl.train_dqn --config ml_training/config.yaml --total-timesteps 1000 --smoke-test

# Policy distillation (teacher → student)
python -m ml_training.rl.distill --config ml_training/config.yaml --teacher outputs/rl_<timestamp>/final_model.zip

# Algorithm comparison (DQN vs PPO vs A2C)
python -m ml_training.rl.train_algo_comparison --config ml_training/config.yaml
python -m ml_training.rl.train_algo_comparison --config ml_training/config.yaml --smoke-test

# Export distilled policy to C header
python ml_training/rl/export_policy_c.py
python ml_training/rl/export_policy_c.py --student outputs/.../student_policy.pth
python ml_training/rl/export_policy_c.py --int8    # INT8 quantized
python ml_training/rl/export_policy_c.py --verify  # verify golden vectors

# Run simulation (unified runner — replaces run_simulation, run_evaluation, verify_env)
python -m ml_training.rl.run_sim --mode verify                                          # env sanity check
python -m ml_training.rl.run_sim --mode demo --model outputs/rl_<timestamp>/best_model/best_model.zip
python -m ml_training.rl.run_sim --mode eval --episodes 100 --model outputs/rl_<timestamp>/best_model/best_model.zip

# Generate thesis figures
python generate_thesis_figures.py --model-dir outputs/rl_<timestamp>
python generate_thesis_figures.py --figures episode envelope tracking comparison distillation training
```

### Visualization & Demos
```bash
python digital_twin_viz/server.py   # WebSocket dashboard → open http://localhost:8765
# run_sim_demo.py and run_box2d_viz.py deleted; use run_sim instead:
python -m ml_training.rl.run_sim --mode demo   # auto-discovers latest model
```

### Tests
```bash
python -m pytest tests/ -v
```

### ESP-IDF Firmware (requires ESP-IDF v5.1+)
```bash
cd edge_firmware
idf.py build
idf.py flash
idf.py monitor                     # Serial output at 115200 baud
```

## Thesis LaTeX Structure

Three document variants exist:

### 1. `docs/thesis/thesis_final.tex` — IEEE Conference Format (main, with chapter includes)
```bash
cd docs/thesis && pdflatex thesis_final.tex && pdflatex thesis_final.tex
```
Includes:
- `chapters/01_introduction.tex` → Chapter 1
- `chapters/02_rrl.tex` → Chapter 2 (Review of Related Literature)
- `chapters/03_methodology.tex` → Chapter 3
- `chapters/04_results.tex` → Chapter 4
- `conclusion.tex` → Chapter 5
- `references.bib`

### 2. `docs/thesis/thesis_report.tex` — IEEE Conference Format (self-contained, 453 lines)
```bash
cd docs/thesis && pdflatex thesis_report.tex && bibtex thesis_report && pdflatex thesis_report.tex && pdflatex thesis_report.tex
```
All 4 chapters inline (no `\input{}`). Good for quick single-file editing.

### 3. `docs/thesis/manuscript/methodology_and_results.tex` — University A4 Format
```bash
cd docs/thesis/manuscript && pdflatex methodology_and_results.tex
```
Includes `00_title_page` through `12_curriculum_vitae`. University submission format.

### Key rule: When told to "update the methodology", edit `chapters/03_methodology.tex`. When told to "update results", edit `chapters/04_results.tex`.

## Known Issues / Tech Debt

1. **Dashboard shows legacy 4-action space & MobileNetV2 label** — `digital_twin_viz/index.html` needs 3 actions and "Direct Pixel Statistics" label.
2. **`ml_training/vision/` directory is legacy** — Not used in deployment. Direct pixel extraction replaced CNN entirely.
3. **`export_onnx.py` is dead code** — The system uses direct C header export (`export_policy_c.py`), not ONNX.
4. **Distillation reward gap** — Student achieves 97.8% *action fidelity* vs teacher, but reward drops from teacher's 4.05 to student's −22.96. Thesis correctly frames this as action agreement, not reward equivalence. New retrain will produce canonical numbers.
5. **Best trained model path** — `outputs/rl_20260303_174001/` (Variant B, canonical). Use `best_model/best_model.zip`.
6. **Ablation Variants A/C** — `outputs/rl_2026030320*/` (A and C retrains, post state_variant bug fix). Table 2 in `04_results.tex` needs updating once these complete.
7. **state_variant config bug (fixed 2026-03-03)** — `TomatoRipeningEnv` takes `state_variant` as a keyword arg, NOT from config dict. `train_dqn.py` now explicitly reads it from `config["rl"]["environment"]["state_variant"]` and passes it through.
