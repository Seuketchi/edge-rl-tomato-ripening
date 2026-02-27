# Edge-RL Thesis — Improvement Checklist (Updated)
**Tristan O. Jadman | BS Computer Engineering | MSU-IIT | March 2026**

---

## BEFORE ANYTHING ELSE — Fix These Writing Errors
30 minutes of work. Do these first.

- [ ] **COOL% mismatch** — Abstract says 93.2% COOL. Table 4.3 shows 28.2% COOL. Add one sentence in Section 4.5 clarifying which number refers to which evaluation condition.
- [ ] **Fixed-Day timing error** — Table 4.4 says 0.48 days. Abstract and Section 4.4 say 0.95 days. Pick the correct number and make it consistent everywhere.
- [ ] **Missing Fixed-Stage5 row** — Described in Section 4.4.1 with mean reward −2288.03 but not in Table 4.4. Add it.
- [ ] **Unresolved (?) citations** — DQN original paper, Stable Baselines3, RL framework definition all show as `(?)`. Find and add the BibTeX entries.
- [ ] **Edge-RL timing error explanation** — Table 4.4 shows Edge-RL has worse timing error (0.67 days) than Random (0.51) and Fixed-Day (0.48). Add one sentence explaining the agent optimizes a composite reward including safety penalties, not timing error alone.
- [x] **"timing timing" typo** — Section 4.8 summary says "timing timing accuracy." Fix it.

---

## PHASE 1 — Rewrite the Vision Module (Do This Now)

Stop training MobileNetV2. It is not needed and cannot be justified given your RL setup.

**Why:** Your simulator generates observations using direct pixel math — G/(R+G) for X and numpy statistics for the RGB features. Your RL policy learned from those values. Deploying MobileNetV2 at inference would feed the policy a different distribution than it was trained on, introducing covariate shift. The direct computation is the only consistent choice.

### What to Rewrite

- [x] **Rewrite Section 3.2 entirely** — rename it from "Continuous State Vision Perception" to "Direct Visual Feature Extraction." Describe the pipeline as:
  1. OV2640 captures RGB frame
  2. 60% centre crop isolates the fruit
  3. RGB means, standard deviations, and modes computed directly from pixel arrays
  4. Chromatic Index X computed as G/(R+G) from the mean values
  5. 10D feature vector passed to RL policy

- [x] **Remove Section 3.2.1 (Dataset Transformation)** — no dataset needed for direct computation.
- [x] **Remove Section 3.2.2 (MobileNetV2 Architecture)** — no CNN needed.
- [x] **Remove Section 3.2.3 (Training and INT8 Quantization)** — no model to train or quantize.
- [x] **Add one paragraph justifying no CNN** — state explicitly that a CNN would introduce approximation error and covariate shift into a pipeline that uses deterministic formula-based observations in both training and deployment. The direct computation guarantees consistency.
- [x] **Update Section 3.7 summary** — Phase 1 is now "direct pixel-based feature extraction" not "MobileNetV2 regression."
- [x] **Update Table 4.7 (Resource Utilization)** — remove any vision model flash/SRAM entries. Your resource numbers actually improve since there is no vision model occupying flash.
- [x] **Add X formula to the thesis** — Equation X: chromatic_x = G_mean / (R_mean + G_mean + ε) where ε = 1e-6 prevents division by zero. This should appear in Section 3.2 alongside the RGB statistics formulas.
- [x] **Add one limitation in Section 1.5.2** — X = G/(R+G) is a formula-based proxy, not a calibrated biological measurement. It is sensitive to lighting conditions and does not directly measure lycopene content. This is honest and your panel will appreciate it.

### What This Gains You
- No training time waiting on MobileNetV2
- No quantization error in the vision pipeline
- Microsecond computation instead of hundreds of milliseconds
- Perfect consistency between simulator observations and real hardware
- A cleaner, more defensible thesis

---

## PHASE 2 — RL Methodology Fixes

These are targeted additions to Chapter 3, not rewrites.

- [x] **Define the action-to-temperature mapping formally** — Equation 3.2 uses `Action` as a variable but never defines the mapping. Add: Action ∈ {−1, 0, +1} where COOL=−1, MAINTAIN=0, HEAT=+1, so ∆u·Action gives the signed temperature increment.
- [x] **Define Xref precisely** — Variant A and B both include Xref described as "the analytical ODE solution at ideal temperature." State exactly what ideal temperature means — is it 20°C? The target harvest temperature? Define it with a formula so the reader can reproduce it.
- [x] **Justify exploration fraction of 0.7** — Table 3.3 lists this but never explains it. Add one sentence: the high exploration fraction was chosen because the ripening environment has sparse rewards and delayed feedback, requiring extended random exploration before the reward signal becomes informative.
- [x] **Acknowledge the reward tension** — r_progress always rewards faster ripening regardless of the target day, which conflicts with r_track when the tomato is ahead of schedule. Add one sentence in Section 3.4.3 acknowledging this tension and explaining that r_track dominates near the deadline due to the 1/max(t_rem, ε) term.
- [x] **Acknowledge ablation training variance** — Each state-space variant trained a separate DQN teacher. Add one sentence noting that results may reflect training variance across runs rather than purely state-space differences, and that multi-seed evaluation is recommended in future work.
- [x] **Acknowledge distillation dataset bias** — 100,000 rollout samples reflect the teacher's action distribution. If COOL dominates, COOL states are overrepresented. Add one sentence in Section 3.5.1 noting this and its implication for per-class fidelity.

---

## PHASE 3 — Hardware Construction

- [ ] **Add prototype photo** — single most impactful thing you can add to Chapter 3. Real photo of the working chamber replaces a hundred words.
- [ ] **Add wiring block diagram** — ESP32-S3 → relay → heater and fan. Hand-drawn is fine.
- [ ] **Measure and report thermal behavior** — heat chamber to 35°C, turn heater off, log cooling rate. This validates k_loss in Equation 3.2. Add 2–3 sentences in Section 3.3.2.
- [ ] **Report actual passive cooling ΔT** — measure how far below ambient the fan can bring the chamber temperature. State this honestly as a hardware constraint.

---

## PHASE 4 — Real Tomato Experiment

5–10 fruits is enough for undergrad proof-of-concept.

- [ ] **Run trials with the complete system** — camera computing X directly, RL policy making decisions, relay controlling heater. Log everything via JSON telemetry.
- [ ] **Record per trial:** initial X₀, target harvest day, actual harvest day, photo at harvest.
- [ ] **Report sim-to-real gap honestly** — if real timing error is worse than simulation, that is expected and acceptable. State it and note it as future work.
- [ ] **Update Abstract** — add one sentence with real trial results once available.
- [ ] **Update Section 1.5.2.1** — change "Simulation-Only" limitation to reflect that preliminary real validation was conducted.

---

## Fill in the Empty Appendix Sections

Heyrana's appendix was one of the things that made his thesis feel complete. Yours lists sections with no content.

- [ ] **Appendix A.2** — add DQN training convergence plot (episode reward over timesteps).
- [ ] **Appendix A.3** — add distillation training convergence plot (action fidelity over epochs).
- [ ] **Fill in Curriculum Vitae** — Date of birth, place of birth, and email are still placeholders.

---

## Summary Table

| What | When | Must Fix? |
|---|---|---|
| Fix 6 writing errors | Now | Yes |
| Rewrite Section 3.2 (drop MobileNetV2) | Now | Yes |
| Fix RL methodology gaps (action mapping, Xref, reward tension) | Now | Yes |
| Add prototype photo + wiring diagram | After hardware build | Yes |
| Measure thermal behavior | After hardware build | Yes |
| Run 5–10 real tomato trials | After hardware | Yes |
| Update Abstract with real results | After experiment | Yes |
| Fill in Appendix A.2, A.3, CV | Anytime | Yes |

---

*The thesis is technically strong. These changes make it honest, consistent, and complete.*