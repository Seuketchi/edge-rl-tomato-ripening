# Edge-RL Thesis — Complete TODO Plan
**Tristan O. Jadman | BS Computer Engineering | MSU-IIT | March 2026**
**Reference standard: Heyrana (2025) — UV-Vis Spectrophotometer with ML**

---

## HOW TO USE THIS
Work top to bottom. Each phase must be reasonably complete before moving to the next.
Items marked `[NOW]` can be done today without any hardware or experiments.
Items marked `[HARDWARE]` require the physical prototype to be built first.
Items marked `[EXPERIMENT]` require real tomatoes and a running system.

---

## ══════════════════════════════════════
## STAGE 1 — STOP AND FIX (Do Right Now) ✅ COMPLETE
## ══════════════════════════════════════

### 1.1 Stop Ongoing Work That Is No Longer Needed
- [x] `[NOW]` Stop MobileNetV2 training. It is not needed. Direct pixel computation replaces it entirely.
- [x] `[NOW]` Do not purchase or set up any additional ML pipeline for vision. The camera + arithmetic is the full vision module.

### 1.2 Fix Writing Errors
- [x] `[NOW]` **COOL% mismatch** — Abstract now correctly says 28.2%, matching Table 4.3. Fixed in prior conversation.
- [x] `[NOW]` **Fixed-Day timing error** — Conclusion corrected to match Table 4.4 (Fixed-Day = 0.48 days, Edge-RL = 0.67 days).
- [x] `[NOW]` **Missing Fixed-Stage5 row** — Already present in Table 4.4 (reward −2288.03).
- [x] `[NOW]` **Unresolved (?) citations** — Added `sutton2018`, `bellman1957`, `jeannoob_yolov8`, `tomatod_dataset`, `ripeness_regression_cv` to References.bib. **0 unresolved citations in compiled PDF.**
- [x] `[NOW]` **Edge-RL timing error** — Added explanation: agent optimizes composite reward (quality + safety), not timing error alone. 0.67 days is within 1-day practical tolerance.
- [x] `[NOW]` **"timing timing" typo** — Not found in current manuscript (already fixed in prior conversation).
- [x] `[NOW]` **Recommendation 1 trial count** — Changed from 30 → 5–10 trials.
- [x] `[NOW]` **Figure ?? broken reference** — Created TikZ conceptual framework figure in Section 1.7. **0 unresolved Figure/Table references.**

### 1.3 Writing Style Pass (Heyrana Standard)
- [x] `[NOW]` "So what" sentences added to key Results numbers (7ms latency, 97.8% fidelity, resource utilization).
- [x] `[NOW]` Equations in Chapter 3 audited — all symbols defined inline. Action mapping (COOL→−1, MAINTAIN→0, HEAT→+1) already present.
- [x] `[NOW]` Section 1.5.2 limitations — all 7 limitations now end with concrete future-work paths.
- [x] `[NOW]` Section 4.7.2 expanded from 2 sentences to 4 paragraphs (cost, timing, compression, latency comparisons).

---

## ══════════════════════════════════════
## STAGE 2 — REWRITE CHAPTER 3 VISION SECTION ✅ COMPLETE
## ══════════════════════════════════════

### 2.1 Remove MobileNetV2 Content
- [x] `[NOW]` Sections 3.2.1–3.2.3 deleted (done in prior conversation).

### 2.2 Rewrite Section 3.2 as Direct Visual Feature Extraction
- [x] `[NOW]` Section 3.2 renamed to "Phase 1: Direct Visual Feature Extraction" (done in prior conversation).
- [x] `[NOW]` Full pipeline description written (capture → crop → stats → chromatic index → policy).
- [x] `[NOW]` X formula added as numbered Equation 3.1.
- [x] `[NOW]` No-CNN justification paragraph added.
- [x] `[NOW]` Formula-Based Ripeness Proxy limitation added to Section 1.5.2.

### 2.3 Update Related Sections
- [x] `[NOW]` Section 3.7 summary updated.
- [x] `[NOW]` Table 3.5 (FreeRTOS tasks) — Vision task already says "Direct pixel extraction" with 8KB stack.
- [x] `[NOW]` Table 4.7 (Resource Utilization) — no vision model entries present.
- [x] `[NOW]` TikZ architecture diagram updated: "Vision Model (MobileNetV2)" → "Feature Extraction (Pixel Statistics)".
- [x] `[NOW]` OV2640 resolution justification — no MobileNetV2 reference remains; current text is appropriate.

---

## ══════════════════════════════════════
## STAGE 3 — FIX RL METHODOLOGY GAPS ✅ COMPLETE
## ══════════════════════════════════════

- [x] `[NOW]` **Action-to-temperature mapping** — Defined in Section 3.3.2: COOL→−1, MAINTAIN→0, HEAT→+1.
- [x] `[NOW]` **Xref defined precisely** — Section 3.4.1 defines T_optimal = 20°C and Xref(t) = X₀·exp(−k₁(T_optimal − T_base)·t).
- [x] `[NOW]` **Exploration fraction justified** — Sentence added in Section 3.4.5.
- [x] `[NOW]` **Reward tension acknowledged** — Paragraph added in Section 3.4.3 (r_progress vs r_track).
- [x] `[NOW]` **Ablation training variance** — Sentence added in Section 4.2.
- [x] `[NOW]` **Distillation dataset bias** — Sentence added in Section 3.5.1.

---

## ══════════════════════════════════════
## STAGE 4 — CREATE MISSING FIGURES ⚠️ PARTIALLY COMPLETE
## ══════════════════════════════════════

- [x] `[NOW]` **Conceptual framework figure (Figure 1.1)** — TikZ block diagram created in Section 1.7.
- [ ] `[NOW]` **DQN training convergence plot (Appendix A.2)** — ⚠️ BLOCKED: No training log data found in repository. Need to rerun training with logging, or locate existing TensorBoard/CSV logs.
- [ ] `[NOW]` **Distillation convergence plot (Appendix A.3)** — ⚠️ BLOCKED: No distillation log data found in repository. Need to rerun distillation with logging, or locate existing logs.

---

## ══════════════════════════════════════
## STAGE 5 — BUILD THE HARDWARE ← **YOU ARE HERE**
## ══════════════════════════════════════

- [ ] `[HARDWARE]` Assemble the XPS foam enclosure per BOM dimensions (20×20×15 cm internal).
- [ ] `[HARDWARE]` Mount OV2640 camera at top center facing downward.
- [ ] `[HARDWARE]` Mount DHT22 sensor mid-chamber away from heater.
- [ ] `[HARDWARE]` Wire relay module between ESP32-S3 GPIO and heating element with opto-isolation.
- [ ] `[HARDWARE]` Wire 12V DC fan for passive ventilation across chamber.
- [ ] `[HARDWARE]` Verify firmware boots and golden vector test passes 20/20 (check serial output).
- [ ] `[HARDWARE]` Verify DHT22 readings are plausible (22–35°C, 40–80% RH).
- [ ] `[HARDWARE]` Verify relay clicks when HEAT action is selected.
- [ ] `[HARDWARE]` Verify fan activates when COOL action is selected.
- [ ] `[HARDWARE]` Verify JSON telemetry streams correctly at 115200 baud.

### Hardware Documentation for Thesis
- [ ] `[HARDWARE]` **Take prototype photo** — full system setup showing chamber, ESP32, and sensors. Add to Section 3.6.1. This is the single most impactful addition you can make to the hardware section.
- [ ] `[HARDWARE]` **Draw wiring block diagram** — ESP32-S3 GPIO → relay → heater, fan connections. Hand-drawn is acceptable. Add to Section 3.6.1.
- [ ] `[HARDWARE]` **Measure heating rate** — activate heater, log temperature every 5 minutes for 60 minutes. Derive actual ∆u. Compare to simulator value. Add 2–3 sentences to Section 3.3.2.
- [ ] `[HARDWARE]` **Measure cooling rate** — heat chamber to 35°C, deactivate heater, activate fan, log temperature decay. Derive actual k_loss. Compare to simulator value.
- [ ] `[HARDWARE]` **Report passive cooling ΔT** — measure how far below ambient the fan can bring chamber temperature. State this honestly as a hardware constraint in Section 3.6.1.2.
- [ ] `[HARDWARE]` **Verify centre crop framing** — confirm a standard tomato fills at least 40% of the 96×96 pixel frame at the camera's mounting height. Adjust mount if needed.

---

## ══════════════════════════════════════
## STAGE 6 — REAL TOMATO EXPERIMENTS
## ══════════════════════════════════════

5–10 fruits is sufficient for undergrad proof-of-concept.

### Setup
- [ ] `[EXPERIMENT]` Source mature green tomatoes (X₀ ≈ 0.85–0.95) from a consistent supplier. Diamante Max F1 preferred to match calibration data.
- [ ] `[EXPERIMENT]` For each fruit: record weight, take initial photo under standardized lighting, record initial X₀ from the system.
- [ ] `[EXPERIMENT]` Assign a target harvest day from the U[3,7] range used in training.

### Running Trials
- [ ] `[EXPERIMENT]` Place fruit in chamber. Power on system. Confirm telemetry is streaming.
- [ ] `[EXPERIMENT]` Check telemetry once daily — do not intervene.
- [ ] `[EXPERIMENT]` Harvest when system triggers (X ≤ 0.15) or at day 7 truncation.
- [ ] `[EXPERIMENT]` At harvest: record actual harvest day, take photo, note firmness and colour.

### What to Record Per Trial
- [ ] `[EXPERIMENT]` Initial X₀
- [ ] `[EXPERIMENT]` Target harvest day
- [ ] `[EXPERIMENT]` Actual harvest day
- [ ] `[EXPERIMENT]` Timing error = |actual − target| in days
- [ ] `[EXPERIMENT]` Action log from JSON telemetry (HEAT/MAINTAIN/COOL distribution)
- [ ] `[EXPERIMENT]` Harvest photo

### Adding Results to Thesis
- [ ] `[EXPERIMENT]` Add new subsection 4.6.4: "Preliminary Real Tomato Validation" with a simple results table.
- [ ] `[EXPERIMENT]` Report real-world action distribution and compare to simulation (51.7% HEAT / 20.1% MAINTAIN / 28.2% COOL). Note differences and explain why.
- [ ] `[EXPERIMENT]` Update Abstract — add one sentence with real trial results.
- [ ] `[EXPERIMENT]` Update Section 1.5.2.1 — change "Simulation-Only" limitation to reflect that preliminary validation was conducted.
- [ ] `[EXPERIMENT]` Be honest about sim-to-real gap in Section 4.7.3 — if timing error on real fruit differs from simulation, state it and note it as future work.

---

## ══════════════════════════════════════
## STAGE 7 — FINAL CLEANUP BEFORE SUBMISSION
## ══════════════════════════════════════

- [x] Check every Figure ?? and Table ?? reference — none are unresolved.
- [x] Check every (?) citation — none remain.
- [ ] Confirm Appendix A.2 and A.3 have actual figures, not just section headings. ⚠️ Depends on convergence plot data.
- [ ] Fill in Curriculum Vitae placeholders — date of birth, place of birth, email.
- [ ] Confirm GitHub repository is public and contains all referenced source files.
- [ ] Add README to GitHub repository with setup instructions and how to reproduce results.
- [ ] Read Abstract one final time — it should accurately reflect the completed system including real tomato results.
- [ ] Run spell check on full document.
- [ ] Confirm all table column numbers add up and are internally consistent.
- [ ] Confirm page numbers in Table of Contents match actual chapter locations.

---

## WHAT TO DO NEXT

| Priority | Task | Blocker |
|---|---|---|
| 🔴 **Now** | Generate convergence plots (rerun training/distillation with logging) | Need training data |
| 🔴 **Now** | Build hardware prototype | Physical components |
| 🟡 After hardware | Run 5–10 real tomato trials | Working prototype |
| 🟢 Final | Cleanup, CV, README, spell check | Everything else done |

---

*Stages 1–3 are complete. Stage 4 needs training data for plots. The next physical work is Stage 5 (hardware assembly).*