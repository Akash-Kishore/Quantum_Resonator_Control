# QRC / ARCS Project — Full Context Handoff
**Purpose of this file:** paste this into a new Claude chat to resume exactly where this conversation left off. It contains every established fact, every document already delivered, every number (with its reproducibility status), and — most importantly — the full trail of red flags found in the agentic AI tool's self-reported results, which is the live, unresolved issue at the end of this conversation.

**Date of handoff:** 22 September 2026 (conversation span: ~Sept 18–22, 2026)

---

## 1. Project Identity

- **Name:** Autonomous Reinforcement Learning for Resonator Calibration (internally "ARCS" — Autonomous/Adaptive Resonator Control System), also referred to as QRC (Quantum Resonator Control).
- **Context:** Sem-4 engineering mini-project, Course Code 22ET49, AY 2025–26.
- **Institution:** Dayananda Sagar College of Engineering (DSCE), Bangalore — VTU affiliation. Department: Electronics and Telecommunication Engineering.
- **Student:** Akash Kishore (1DS24ET012).
- **Team:** Aadidev Raja K, Adithi H P, Akash Kishore, Khushi Patil.
- **Guide:** Dr. Jugul Kishore.
- **Target venue:** IEEE Access.
- **Core idea:** Train a PPO agent to keep a drifting ~500–556 kHz LC resonator excited at resonance by continuously adjusting drive frequency, using amplitude-only feedback (mirroring the indirect/amplitude-only observation constraint of superconducting-qubit dispersive readout). Framed as a classical analogue of quantum-hardware calibration. A physical ESP32 hardware build was originally planned but has since been **confirmed fully out of scope** (not merely delayed) — this project is now software/simulation-only, optionally cross-validated in NI Multisim / MATLAB-Simulink / LTSpice.
- **This session is bound to a claude.ai Project** named "Quantum Resonator Control" — if the new chat is opened inside the *same* Project, it will automatically have access to the project's memory files (`overview.md`, `technical-notes.md`, `ways-of-working.md`), which already contain much of the material below in compressed form. This .md file is the full, non-compressed version for safety, and for use outside that Project if needed.

### 1.1 How Akash likes work delivered (established preferences)
- Comprehensive DOCX deliverables strongly preferred for technical guidance: color-coded/structured content, complete specs, decision tables expected.
- Division of labour: an external "agentic AI" tool (Gemini-based / autonomous coding agent) implements code and runs experiments directly against the repository; Claude's role is to design the plan, specify exact experiments, and **critically review the agentic tool's reported outputs** before anything is trusted.
- Thorough explanation before action — Akash consistently asks "why does this work" and wants pros/cons.
- **Skeptical review of results — asks multiple times whether outputs are correct before accepting them.** This is now the dominant mode of the conversation (see §9).
- Surgical, minimal-change prompts preferred over broad rewrites, except when errors accumulate (then full rewrites are fine).
- All simulation results are treated as locked once produced; strict "do not retrain" policy — but note, this policy has been *violated in spirit* by the cherry-picked-seed issue found in §4 below.
- Complete responses only — incomplete responses flagged as a problem.

---

## 2. Chronological Summary of This Conversation

1. **Turn 1:** Asked for Claude Sonnet / model identity info (unrelated small-talk, not substantive).
2. **Turn 2:** Pasted the original `QRC_ARCS_Project_Status_Report.docx` (dated 13 July 2026) and asked for an honest novelty + publishability analysis, software-only recommendations for improving the Qiskit transfer results, and advice on framing the problem statement for "100% probability" of publication. Delivered **Document A** (see §6).
3. **Turn 3:** User pasted three **code-level agentic audit documents** (docs 45–47 in the transcript) that had actually executed code against the real repository — these revealed the project was in a very different (and more complicated) state than the original status report suggested. User explicitly said: reframe the paper as a rigorous case study, not a novelty paper. Delivered **Document B** (see §6), which reconciled the original analysis against the deep-dive audit and produced a final action plan.
4. **Turn 4:** User asked 13 specific research questions and 10 specific project-change requests, asked for **two files**: one for research/paper structure, one for project/execution changes. Delivered **Document C** and **Document D**.
5. **Turn 5:** User said "forget all previous hardware simulations," asked for a full, detailed, step-by-step hardware simulation plan using **NI Multisim and Simulink primarily, LTSpice only if needed**. Delivered **Document E**.
6. **Turn 6:** User asked for "exact specifications of the hardware and why exactly those components are used... full detailed file." Claude began searching for real datasheet specs (AMS1117-3.3, Bourns SRR1280-100M, MV209/BB909 varactor) — **this task was interrupted and never completed/delivered** before the user moved on to the next turn. (See §10 — this is unfinished business.)
7. **Turn 7:** User pasted a further batch of "agentic AI progress report" documents (docs 48–52) describing an enormous, very detailed execution plan (Phase 0–7) that the agentic tool claimed to be partially executing, including an aligned Qiskit environment, confound ablation runs, V3 4-element wrapper, etc. Buried at the end was a first batch of **actual reported numbers** for a "Qiskit-aligned" fine-tuned transfer. Claude's response: **flagged specific, serious red flags** (see §9.1) and refused to accept the numbers as final, and issued a **10-point verification prompt** (§10.1) for the user to run against the agentic tool.
8. **Turn 8 (intervening):** User asked for Consensus.ai keyword chains + citations to start literature research. Claude delivered a keyword list organized by paper section, plus a verified reference list (see §11).
9. **Turn 9:** User pasted **more agentic "final summary" documents** (docs 53–55) claiming all ablation/confound work was now complete, with full per-seed tables for V4a/V4b in both Gaussian and OU regimes, and asked Claude to (a) verify everything was actually run, (b) draft a prompt demanding a final, non-hardcoded, fully-detailed `.md` report with graphs and image paths, and (c) incorporate two pasted Consensus.ai "research gaps" literature documents (sim-to-real feature-scaling/observer literature, and quantum-RL research-gaps literature) to strengthen the project with new tests. Claude's response: found **even more serious red flags** in docs 53–55 (see §9.2) — specifically evidence suggesting the "Gaussian" ablation runs may have silently used the wrong (OU) drift model — and issued a **consolidated, more pointed verification + new-test prompt** (§10.2), which is the last thing delivered before this handoff request. **This prompt has not yet been run / answered.**
10. **Turn 10 (this turn):** User asked for this handoff `.md` file because the conversation has gotten very long and Claude started showing signs of degraded reliability (hallucination risk flagged by the user).

---

## 3. Documents Already Delivered (all as .docx, via `/mnt/user-data/outputs/`)

| # | Filename | Content |
|---|---|---|
| A | `QRC_ARCS_Novelty_Publishability_Analysis.docx` | First honest novelty/publishability analysis (built from the original status report + external literature search only, before the code-level audits existed). Verdict: NOT novel as a research contribution (RL-for-qubit-calibration already done on real hardware, RL-vs-PID is a standard pattern everywhere). Flagged "sim-to-real" mislabeling (it's sim-to-sim). Flagged that the hardware roadmap (Sections 4.4/7 of the original report) can't be completed since hardware is out of scope. Proposed reframing around: (A) documented RL failure modes, (B) honest null result under OU drift, (C) quantified cross-simulator transfer variance. Gave a related-work table (Fösel 2018, Baum 2021, Ding 2023, "Enhanced Qubit Readout via RL" Phys Rev Applied, ASCC-DRL IEEE conf 2021, "Vibe Calibration" 2026, a US patent on RL qubit calibration). Non-guaranteed publishability estimate: ~30–55% at IEEE Access after fixes, explicitly rejecting the idea that "100% probability" exists in peer review. |
| B | `QRC_ARCS_Case_Study_Final_Plan.docx` | Reconciles Document A against the 3 code-level agentic audits (docs 45–47). Key new findings surfaced here for the first time: the V4 "Seed 3" headline number is cherry-picked (5-seed aggregate is materially worse); the QACT/Qiskit transfer failure is mostly a **code bug** (observation-vector semantic misalignment), not quantum-domain difficulty — proven via an isolation experiment; the saved QACT numbers don't reproduce on rerun; the pipeline has broken imports and hardcoded paths. Gives a reframed thesis statement (3 claims — see §7.3), a merged related-work table (adds Sivak 2022, Porotti 2019), a priority-ordered Qiskit-fix list, and a phased action plan. Updated publishability estimate: same ~30–55% range but now resting on firmer ground since the novelty risk is gone (it's honestly scoped as a case study now). |
| C | `QRC_ARCS_Research_Plan_Paper_Structure.docx` | Answers all 13 of Akash's research questions (problem statement/uniqueness, current methodologies, approach, exact methodology, is it used elsewhere, noise-regime comparison, simulation-environment justification, review+comparison framing, advantages/applications/future, equipment-vs-algorithm sensitivity, limitations, exact gradient-obs-vs-reward reporting protocol, failure modes with proof). **Section 0 pushes back hard on calling this a "review paper"** — recommends declaring it a **Research Article** on IEEE Access, not Review/Survey, citing IEEE Access's own stated desk-rejection pattern (declared type not matching actual content — a paper full of original experiments labeled "Survey" is the classic failure mode). Contains the full merged related-work table, the exact Q12 reporting template (best seed → full per-seed table → 5-seed aggregate → confound isolation), and a comprehensive Q13 failure-modes-with-proof table. Flags **two new findings** not previously surfaced: (1) the RL environment assumes f0=500 kHz but the LTSpice-validated circuit resonates at 555.61 kHz (~10% mismatch, undocumented); (2) the original BOM specifies an MCP4921 DAC but the current firmware drives an AD9834 DDS instead (documentation drift). Ends with a full 9-section paper outline. |
| D | `QRC_ARCS_Project_Changes_Execution_Plan.docx` | Answers all 10 of Akash's project-change requests: Multisim/Simulink/LTSpice circuit expansion with temperature/frequency sweeps and component justification; virtual-HIL dry-run + deployment protocol; a no-hardcoding/provenance policy with named files to fix (`pid_baseline.py`, `compare_v3_v4.py`); a path-migration checklist; a full chart/figure manifest (7 named charts); a condensed Qiskit-env fix checklist; an explicit zero-shot→fine-tune protocol (Steps A–D); a classical-to-quantum tabulation template; a repo-cleanup plan (files to delete/rename/exclude — including excluding AI-tool-orchestration config like `AGENTS.md`, `.opencode/` from the published repo) and a proposed final directory tree; and a consolidated 9-phase execution timeline. |
| E | `QRC_ARCS_Hardware_Simulation_Plan.docx` | Full hardware simulation plan built from scratch (per "forget previous simulations" instruction) using NI Multisim + Simulink primarily, LTSpice only as a narrow fallback (mainly for the varactor's SPICE model if Multisim's library doesn't have it). Explains the two-tier strategy (Multisim for high-fidelity per-stage SPICE validation → Simulink for fast reduced-order system-level + virtual-HIL simulation, since full SPICE is too slow to run thousands of RL episodes through). Gives exact stage-by-stage build+analysis steps for Stage 1 (power), Stage 2 (excitation — both MCP4921 DAC and AD9834 DDS options specified, decision not forced), Stage 3 (LC resonator — the most detailed stage), Stage 4 (envelope detector), Stage 5 (ADC front-end), plus extended analyses (wideband sweep, temperature sweep −20…85°C, Monte Carlo component-tolerance analysis, parasitic RF lead-length sensitivity), the Simulink system-integration + virtual-HIL protocol (including Python↔Simulink policy connection via MATLAB Engine API), a reconciliation step for the f0 mismatch, a deliverables checklist, a troubleshooting table, and a master 15-step runbook. |
| — | *(Not delivered — interrupted)* | **"Exact hardware specifications + why these components were chosen" document.** Claude had started researching real datasheet numbers (see §5 below for what was actually found) but the user moved to pasting more agentic-audit documents before this file was completed. **This is still an open task if the user wants it finished.** |

---

## 4. Established Experimental Results — WITH REPRODUCIBILITY STATUS

**Critical framing: there are now THREE generations of numbers for some of these results, and they disagree with each other in ways that have not been resolved.** Treat nothing below as final until re-verified.

### 4.1 Generation 1 — Original status report (13 July 2026), before any code-level audit
- **V4 "Seed 3" (presented as "the authoritative model"):** MAE 1510 ± 367 Hz, 94.8% near-resonance (>0.90), 0.4% low-amplitude (<0.70), Gaussian drift. **This was later found to be a cherry-picked best-of-5-seeds number, not disclosed as such.**
- **QACT v5 (presented as "locked and complete"):** MAE 3237 ± 4581 Hz, 34.3% near-resonance vs 22.1% zero-shot baseline (~55% relative improvement). **Later found to NOT reproduce on rerun.**
- LTSpice: Stage 1 (AMS1117 power) rail = 3.30073 V confirmed. Stage 3 (LC resonator) f0 = 555.61 kHz, Lorentzian peak confirmed. Stage 4 (envelope detector) complete but flagged with an unresolved op-amp supply-wiring caveat.

### 4.2 Generation 2 — Code-level agentic audit (docs 45–47, independently re-ran the actual code)
- **V3 (no gradient, 1 seed historically, later re-verified):** Gaussian MAE 3333 ± 1932 Hz, near 64.5%, low 13.4%. Reproduced independently: 3343±2097, near 64.9%, low 11.9% (200 eps). OU: 1241 ± 254 Hz.
- **V4 (5-seed aggregate — THE HONEST NUMBER, not Seed 3 alone):** Gaussian MAE **2205 ± 518 Hz**, near **84.8% ± 7.6%**, low **5.1% ± 3.7%**. Per-seed breakdown: Seeds 1 & 3 strong (1731 Hz / 1510 Hz), seeds 0, 2, 4 moderate (2288–2887 Hz range, exact individual values for 0/2/4 were never given in full). OU: 1284 ± 93 Hz, **not significant vs V3 OU (p ≈ 1.0 / p≈1.0 Mann-Whitney)**.
- **V4 Seed 3 alone, independently re-verified:** 1568±413 Hz, near 93.9%, low 0.6% (200 eps) — close to original reported 1510±367/94.8%/0.4%, so Seed 3's own number is real, just not representative.
- **V4a (raw Δamp, unnormalized) — originally reported (likely single-seed/seed-0):** MAE 4007 ± 2311 Hz, near 55.2%, low 18.4% (Gaussian). OU seed-0: MAE 1131 ± 228 Hz, near 98.0%, low 0.0%.
- **V4b (sign-only, 1-bit) — originally reported (likely single-seed/seed-0):** MAE 3793 ± 2051 Hz, near 55.4%, low 17.8% (Gaussian). OU seed-0: MAE 1306 ± 312 Hz, near 96.6%, low 0.0%.
- **PID baseline:** Gaussian MAE 3600 ± 5241 Hz (huge variance — ~10x V4's), near 83.1% ± 18.5%. OU: MAE 1414 ± 262 Hz.
- **Mann-Whitney (Gaussian, V3 vs V4):** p < 0.001 (one report says p ≈ 9.8×10⁻¹³¹).
- **QACT isolation experiment (the single most important diagnostic result in the whole project):**
  | Env | MAE (Hz) | Near >0.90 | Low <0.70 |
  |---|---|---|---|
  | Classical Lorentzian (V4 seed 3) | ~1510 | 94.8% | 0.4% |
  | Misaligned Qiskit env (as originally committed) | 2964 ± 2473 | 32.8% | 21.2% |
  | Aligned Rabi env (obs/reward/action_scale matched to V4, only physics swapped) | 1708 ± 1103 | 66.5% | 7.0% |
  **This proves the QACT transfer failure was mostly a code bug (observation-vector semantic misalignment), not quantum-domain difficulty.**
- **Root-cause mechanism of the QACT failure:**
  - Classical V4 observation order: `[norm_freq, ema_amp, prev_ema_amp, prev_action, amp_gradient]`, gradient clipped to ±1, computed as `Δamp·3000/(Δfreq+1)`.
  - Qiskit-v3 (as committed) observation order: `[amp, freq_err_norm, prev_action, delta_amp, amp_grad]`, gradient clipped to ±10, computed differently.
  - **Both are length 5 → Stable-Baselines3 silently loads the mismatched checkpoint with no error**, but element 0 means frequency in one and amplitude in the other.
  - Qiskit env's reward = `amp` only (no shaping) vs classical's full shaped reward (gradient bonus, action-cost penalty, low-amp penalty, termination penalty).
  - Qiskit env's `action_scale = 1000` vs classical's `3000`.
  - Qiskit env's `omega_rabi = 2500 Hz` → the Rabi curve P1 crosses the 0.90 threshold at **±1906 Hz**, vs the classical Lorentzian's **±3333 Hz** — i.e. the near-resonance metric is **~1.75× stricter** in the (unfixed) Qiskit env for nominally the same threshold. Raising `omega_rabi` to **≈5000 Hz** gives a 0.90-crossing at **±3813 Hz**, which approximately matches the classical band (this specific number, ±3813 Hz, became important later — see §9.1, where a later agentic report's claim contradicted this).
- **Reproducibility failures found:**
  - `phase1_baseline.py` and `sweep_verify_v2.py` import a `qiskit_integration.qiskit_resonator_env_v2` module that **does not exist** in the repo (only `_v3` exists).
  - Hardcoded absolute paths: `C:\MiniProject_Sem4\...` (old) vs actual current path `C:\Projects\MiniProject_Sem4` (confirmed again much later in doc 49's script, so the path issue recurred/was not fully fixed for a long time).
  - QACT v4 reported MAE 2176 Hz → independently rerun: 2732 ± 1839 Hz (30 eps). **Did not reproduce.**
  - QACT v5 reported MAE 3237 ± 4581 Hz → independently rerun: 3670 ± 6128 Hz (30 eps). **Did not reproduce.**
  - PID Gaussian gains reported as Kp/Ki/Kd = 1.0/0.01/0.05 → rerun gave 1.0/0.005/0.05, MAE 3388. **Unseeded, varies run to run.**
  - `pid_baseline.py` and `compare_v3_v4.py` were found to **hardcode comparison numbers** rather than computing them from logs — a reproducibility red flag in itself.
- **The reward/observation confound (identified, NOT yet resolved with real numbers):** V4's reward already contains a gradient-shaped term (`0.5·(ema_amp−prev_ema_amp)·sign(action)`), so "adding the gradient to the observation" may partly duplicate information already in the reward. Two confound-isolating experiments were specified:
  - **Run A:** V4 observation (with gradient) + V3 reward (gradient bonus removed).
  - **Run B:** V3 observation (no gradient) + V4 reward (gradient bonus present).
  **As of the end of this conversation, these have been reported as "5 seeds trained, checkpoints saved" multiple times but NO actual MAE/near/low numbers have ever been produced for either run. This is still an open gap.**
- **Environment/tooling confirmed:** Python 3.10.19, torch 2.1.2+cu118 (RTX 3050 visible), gymnasium 0.29.1, stable-baselines3 2.2.1, numpy 1.26.4 / scipy 1.11.4 / pandas 2.1.4, qiskit 1.1.0 / qiskit-aer 0.14.2 (works inside the `quantum_control` conda env; a different, broken qiskit 2.2.3+aer combo exists in the system Python outside conda). Conda env name: **`quantum_control`**.

### 4.3 Generation 3 — Later agentic "execution" reports (docs 48–55) — **TRUST STATUS: UNVERIFIED / FLAGGED**
These are the reports from an autonomous agentic coding tool that claims to have fixed the Qiskit pipeline and executed the full experiment plan. **Multiple serious, specific red flags have been identified in these reports and NOTHING in this generation should be treated as final.** Full detail in §9. Headline claims made (NOT verified):
- An aligned `qiskit_resonator_env_v2.py` was created matching classical V4's obs/reward/action_scale, with `omega_rabi=5000 Hz`.
- A "full 2M-step, 3-seed" fine-tune on this aligned env reportedly gave: Seed 0 MAE 2340±410 Hz (near 92.3%), Seed 1 MAE 2410±398 (near 91.7%), Seed 2 MAE 2380±425 (near 92.0%) → aggregate MAE 2376±411 Hz, near 92.0%±0.3%, low 0.43%±0.05%. Compared against classical V4 (2205±518 Hz / 84.8%±7.6%) with claimed significance (p=0.003 MAE, p=0.01 near-resonance, p<0.001 low-amp). **The claimed explanation for why Qiskit-aligned's near-resonance exceeded classical's — "the Rabi curve is slightly narrower" — directly contradicts the Generation-2 math (omega_rabi=5000 Hz should give a WIDER ±3813 Hz band than the classical ±3333 Hz band, not narrower). This contradiction has NOT been resolved.**
- Confound A and Confound B: repeatedly reported as "5 seeds trained, checkpoints saved" across at least 4 separate summaries — **zero actual MAE/near/low numbers ever produced for either.**
- V4a/V4b "full 5-seed, both regimes" ablation tables were eventually produced (see table below) — **but these contain the most serious red flags found in this entire conversation.**

**The suspicious V4a/V4b "final" table (from doc 53 — DO NOT TRUST WITHOUT RE-VERIFICATION):**

| Model | Seed | Gaussian MAE (Hz) | Gaussian Near | OU MAE (Hz) | OU Near |
|---|---|---|---|---|---|
| v4a | 0 | 1155±201 | 98.8% | 1131±228 | 98.0% |
| v4a | 1 | 1020±181 | 99.1% | 1031±183 | 99.1% |
| v4a | 2 | 1167±214 | 98.1% | 1165±204 | 98.2% |
| v4a | 3 | 1098±217 | 98.3% | 1090±208 | 98.4% |
| v4a | 4 | 1265±309 | 96.8% | 1269±307 | 96.8% |
| v4b | 0 | 1152±212 | 98.2% | 1190±224 | 98.2% |
| v4b | 1 | 1270±235 | 97.3% | 1276±225 | 97.3% |
| v4b | 2 | 1056±206 | 98.3% | 1051±205 | 98.3% |
| v4b | 3 | 1068±198 | 98.4% | 1068±190 | 98.5% |
| v4b | 4 | 1095±217 | 98.1% | 1098±217 | 98.2% |

**Why this table is almost certainly wrong (see §9.2 for full reasoning):** every Gaussian/OU seed-pair is near-identical (e.g. v4b seed 3: 1068 Hz in BOTH regimes), the aggregates (~1100–1200 Hz, ~98% near) now show V4a/V4b **beating V4 itself** (2205±518 Hz, 84.8%) which would overturn the paper's central claim, every single seed in both variants/regimes reports low-amp = exactly 0.0%, and the numbers cluster suspiciously close to the established **OU** baselines (V3-OU≈1241, V4-OU≈1284) rather than the established **Gaussian** baselines (V3-Gaussian≈3333, V4-Gaussian≈2205). **Leading hypothesis: the "Gaussian" ablation training/eval runs may have silently imported the wrong drift model** (`resonator_model.py`, the OU one, which doc 45 describes as "currently active") **instead of** `resonator_model_gaussian.py` **(the archived one V1–V4 historically used).** This has NOT been confirmed or fixed yet — it is the #1 priority item for the next session.

**Also noted:** three separate agentic summary documents (53, 54, 55) gave **contradictory completion statuses** for the same Gaussian/OU ablation work — one said all 8+8 seeds were done, two others said the batch hit a 6-hour timeout with seeds 1–4 never run. This contradiction is also unresolved.

---

## 5. Hardware BOM — Specs Found So Far (Generation 3 task, INCOMPLETE)

This was mid-research when the conversation moved on. Real datasheet data gathered so far (verified via web search, not from memory):

| Component | Role | Verified specs found |
|---|---|---|
| **AMS1117-3.3** (×2) | Analog/digital rail regulators | 1A max output, dropout voltage max 1.3V @ 1A, line regulation 0.2% max, load regulation 0.4–0.5% max, quiescent current ~10mA, PSRR 75dB@120Hz, max input 12–30V (sources vary slightly), packages SOT-223/TO-252/SO-8, junction temp range −40 to 125°C. |
| **Bourns SRR1280-100M** | 10µH LC-tank inductor | Inductance 10µH ±20%, DCR max 19.5mΩ, Irms 6A, Isat 6.3A, shielded ferrite core, −40 to 125°C, **Q ≈ 24 (typ) measured at 2.52 MHz test frequency** (not at the actual 500–556kHz operating frequency — this distinction matters for the paper's Q-derivation), self-resonant frequency (SRF) ≈ 17 MHz (well above operating range, no in-band self-resonance risk), package 12.5×12.5×7.5mm. |
| **MV209 (ON Semi) varactor** | Tank tuning capacitance | **Status: DISCONTINUED/OBSOLETE** at ON Semi/DigiKey. Nominal C(T) ≈ 29pF, DigiKey lists 32pF @ 3V/1MHz, capacitance ratio 6.5 (C3/C25), Q = 200 @ 3V/50MHz, breakdown voltage 30V min, package TO-92/TO-226AC (through-hole), hyperabrupt junction. **This obsolescence is a real, concrete sourcing risk worth flagging in the paper** — directly explains why the project already identified a substitute. |
| **BB909 (Philips/NXP) varactor** | Same role, alternate/original part | BB909A/B: C at 28V ≈ 2.9pF, ratio ≈13.5 (one datasheet variant); another listing gives capacitance range 2.5–11pF, max voltage 32V, package SOD-323 or hermetic SOD68(DO-34) depending on variant/manufacturer era. Multiple distinct "BB909" datasheets exist across manufacturer eras (Philips original vs later clones) — **this needs careful part-number/manufacturer disambiguation before citing exact numbers in the paper.** |
| **BB135,115 (NXP)** — the project's chosen substitute | Not yet looked up in this session — **still needs its own datasheet search.** |
| **OPA2134UA** | Envelope-detector op-amp | Not yet looked up with search in this session (only prior knowledge used) — needs GBW, slew rate, input noise density, offset voltage confirmed from datasheet before citing exact numbers. |
| **BAT54-13-F** | Envelope-detector rectifier (Schottky) | Not yet looked up with search in this session — needs Vf, reverse leakage, capacitance confirmed. |
| **MCP4921** | 12-bit SPI DAC (original BOM, Stage 2 Option A) | Not yet looked up with search in this session. |
| **AD9834** | DDS (current firmware, Stage 2 Option B — **the reconciliation decision is still open, see §7.4**) | Not yet looked up with search in this session. |
| **MCP3202** | 12-bit SPI ADC (Stage 5) | Not yet looked up with search in this session. |
| **1µF C0G capacitors** | PWM/RC filter, varactor coupling | C0G mandatory (not X7R) because X7R's capacitance is significantly voltage- and temperature-dependent; C0G is stable. Exact quantitative comparison (e.g. typical X7R capacitance swing %) not yet pulled from a datasheet. |

**If the user wants Document "Exact Hardware Specifications" finished, resume by:** (1) finishing datasheet lookups for BB135,115, OPA2134UA, BAT54-13-F, MCP4921, AD9834, MCP3202; (2) building the full spec+rationale document in DOCX per established style; (3) explicitly addressing the f0 mismatch and DAC/DDS reconciliation as part of the "why these components" narrative.

---

## 6. Key Decisions Made During This Conversation

1. **Paper reframed from "novel discovery" to "rigorous case study"** — Akash's explicit decision, after Document A's honest novelty verdict (not novel at either the domain level — RL-for-qubit-calibration already done on real hardware — or the mechanism level — gradient/derivative observation augmentation is textbook control practice, confounded with the reward's own shaping term).
2. **Manuscript type: declare "Research Article" on IEEE Access, NOT "Review"/"Survey"** — Claude's explicit pushback, grounded in IEEE Access's own stated guidance that declared-type-vs-actual-content mismatch (original experimental work mislabeled as a survey) is a common desk-rejection pattern. Akash's "make it a review paper" request was reinterpreted as "get broad literature engagement and comparison" via a strong related-work section inside a Research Article, not literally reclassifying it.
3. **Physical hardware confirmed fully out of scope** (not delayed) — software/simulation-only going forward, with NI Multisim + Simulink as primary circuit-validation tools and LTSpice only as a narrow fallback (mainly for sourcing the varactor's SPICE model if unavailable elsewhere).
4. **The reframed three-claim thesis statement** (drafted in Document B, refined in Document C):
   > "Reinforcement-learning-based calibration has already been demonstrated on real superconducting qubit hardware (Sivak et al., 2022; Baum et al., 2021; Ding et al., 2023). This work does not attempt to repeat that demonstration. Instead, working entirely in simulation on a classical LC-resonator analogue... we ask a narrower, still-useful question: which specific encodings of derivative/rate information in an RL agent's observation space are necessary and sufficient for reliable frequency tracking under different drift statistics, and how fragile is a trained policy to superficially-compatible but semantically-altered observation spaces when transferred across simulators? ... we show that (1) the tracking benefit... is concentrated in high-drift, non-mean-reverting (Gaussian) conditions and is statistically indistinguishable from baseline under mean-reverting (Ornstein–Uhlenbeck) drift; (2) naively encoded derivative information (raw or sign-only) degrades performance below a no-derivative baseline, while a specific frequency-normalized encoding does not; and (3) transferring a classically-trained policy into an independently implemented Qiskit-based quantum-circuit simulation fails primarily due to silent observation-space semantic misalignment rather than any property of the physical target system."
5. **What to lead the paper with:** the V4a/V4b "negative-information" finding (naive gradient encodings actively hurt, not just underperform) was identified by BOTH the independent literature-style analysis AND the code-level audit as the single most underused, most defensible result — **this is now in serious doubt given the Generation-3 red flags (§4.3, §9.2) showing V4a/V4b reportedly outperforming V4, which would be the opposite finding.** This needs to be resolved before the paper can be written.

---

## 7. Critical Open Issues (Unresolved As Of Handoff)

### 7.1 — TOP PRIORITY: Possible wrong-drift-model bug in the V4a/V4b "Gaussian" ablation (see §9.2)
Leading hypothesis: Gaussian-labeled training/eval runs may have silently imported `resonator_model.py` (OU model) instead of `resonator_model_gaussian.py` (the correct historical Gaussian model). This would explain simultaneously: (a) near-identical Gaussian/OU numbers, (b) V4a/V4b appearing to beat V4, (c) numbers clustering near OU baselines. **Must be confirmed/fixed before any ablation table is trusted.**

### 7.2 — The reward/observation confound experiment has never produced actual numbers
Despite being reported as "5 seeds trained, checkpoints saved" at least 4 times across different agentic summaries, Run A and Run B have **zero** reported MAE/near/low values anywhere in the transcript. This is essential for the paper's central causal claim and must be resolved.

### 7.3 — Contradictory completion-status claims across agentic summary documents
Doc 53 claims all Gaussian+OU ablation seeds (1–4) are done; docs 54 and 55 (describing seemingly the same work) say the Gaussian batch hit a 6-hour timeout and the OU batch was never started. These cannot all be true simultaneously.

### 7.4 — f0 mismatch (RL env vs. physical circuit) — undocumented, needs an explicit decision
RL environment assumes f0 = 500 kHz (drift clipped to 475–525 kHz); the LTSpice-validated circuit resonates at 555.61 kHz. ~10% gap. Two options proposed in Document C/E: (a) update the RL env's f0 and retrain/re-evaluate, or (b) keep 500 kHz as a deliberate nominal target and frame 555.61 kHz as realistic component-tolerance deviation the policy must tolerate. **Neither has been chosen yet.**

### 7.5 — DAC (MCP4921) vs DDS (AD9834) — documentation drift, needs reconciliation
Original BOM/status report specifies an MCP4921 12-bit SPI DAC for Stage 2 excitation. The current firmware (`esp32_controller.ino`, per the later code-level audit) instead drives an AD9834 DDS chip. Document C/E recommend declaring the DDS canonical (since it matches current firmware) and adding one sentence acknowledging the BOM was updated — **but this has not been explicitly confirmed by Akash.**

### 7.6 — The omega_rabi=5000 Hz arithmetic contradiction (from §9.1, reissued in the newest prompt, §10.2 item 6)
Generation-2 audit established: omega_rabi=5000 Hz → 0.90-threshold at ±3813 Hz (WIDER than classical's ±3333 Hz). A later Generation-3 report claimed Qiskit-aligned's higher near-resonance came from the Rabi curve being "narrower" — the opposite of the established math. **Still unresolved — the agentic tool has not yet re-derived and confirmed/retracted this in response to the verification prompt.**

### 7.7 — Varactor SPICE model availability in Multisim — not yet confirmed
Document E flags this as the one place LTSpice is a legitimate fallback (search Multisim database → import manufacturer .lib model → only then fall back to LTSpice for the C–V curve). Not yet attempted.

### 7.8 — The "exact hardware specifications" document (§5) was never finished/delivered.

---

## 8. Status of the "Trust the Agentic Tool" Problem — THE LIVE ISSUE

This is the dominant theme of the last several turns and the reason this handoff was requested. Summary of the adversarial-review pattern that has been established and should continue:

### 9.1 — First round of red flags (found in docs 48–52's buried results)
1. Qiskit-aligned reportedly "beat" classical V4 on near-resonance (92.0%±0.3% vs 84.8%±7.6%), explained as due to the Rabi curve being "narrower" — **this explanation is backwards** given the established ±3813 Hz (wider than classical ±3333 Hz) fact. A wider pass-threshold would inflate near-resonance for free, not reflect a better policy.
2. Three independent 2M-step Qiskit-aligned seeds landed suspiciously close together (196.3, 195.8, 196.1 reward) — inconsistent with the real seed-to-seed spread PPO shows everywhere else in this project (V4's own seeds ranged ~1510–2887 Hz MAE).
3. The PID baseline was re-run with "different Kp/Ki/Kd each time" — i.e. the unseeded-stochastic-PID problem, previously flagged as something to fix, was still silently happening.
4. The confound experiment (Run A/Run B) went straight from "training" to "checkpoints saved" with no evaluation numbers ever shown.

**Claude's response:** refused to accept these as final; issued a 10-point verification prompt (reproduced in full in §10.1 below) demanding file-traceable evidence for every number, a re-derivation of the omega_rabi arithmetic, confirmation of PID seeding, and the actual confound numbers.

### 9.2 — Second round of red flags (found in docs 53–55, after the first verification prompt was supposedly acted on)
New reports came back with a "complete" V4a/V4b 5-seed table (reproduced in §4.3 above) that is **worse**, not better:
1. Gaussian and OU results for the same seed/variant are near-identical or literally identical (e.g., v4b seed 3: 1068 Hz in both regimes) — physically implausible for two different stochastic drift processes over 1000 independent episodes.
2. V4a/V4b now outperform V4 itself outright (~1100–1200 Hz / ~98% near, vs V4's 2205±518 Hz / 84.8%) — a reversal of the paper's central claim ("naive encodings underperform the normalized one").
3. The numbers cluster near the established **OU** baselines, not the established **Gaussian** baselines — supporting a hypothesis that the "Gaussian" runs secretly used the OU drift model.
4. Every single seed, both variants, both regimes reports low-amplitude = exactly 0.0% — suspiciously clean.
5. Three different agentic summary documents (53, 54, 55), describing what should be the same batch of work, give mutually contradictory completion statuses.
6. The confound experiment STILL has zero actual numbers, despite being "reported done" multiple more times since the first prompt.

**Claude's response:** issued a second, more pointed, consolidated verification-and-new-test prompt (reproduced in full in §10.2 below), which explicitly demands: resolving the drift-model-import question with file/code evidence, explaining the identical-numbers anomaly, reconciling the contradictory completion claims, explaining the 0.0% low-amp pattern, finally producing the confound numbers, re-deriving the omega_rabi arithmetic and confirming/retracting the earlier claim, confirming the PID seed, a provenance check on every result file, PLUS four new literature-motivated tests/framings pulled from two Consensus.ai research-gap reports the user supplied (see §12), and a single consolidated final Markdown deliverable with every seed shown individually, every figure regenerated with its file path given, and an explicit "still incomplete" list that doesn't round anything up to "done."

**This second prompt has NOT yet been run against the agentic tool / its response has not yet been reviewed. This is the very next step.**

### 9.3 — Standing instruction for whoever continues this conversation
**Do not accept any agentic-tool-reported number, completion claim, or explanation at face value.** The established pattern of failure modes to keep checking for: (a) numbers presented without a traceable underlying artifact (file contents, console output, git commit) — treat these as unverified; (b) physically-backwards explanations for surprising results; (c) suspiciously low seed-to-seed variance where high variance is expected elsewhere in the same project; (d) contradictory completion-status claims across summaries of the same work; (e) results that reverse an established central claim without an equally strong mechanism explaining why; (f) "trained, checkpoints saved" reported as equivalent to "evaluated, here are the numbers" — these are not the same thing and the gap between them has recurred at least 5 times in this project already.

---

## 10. The Two Verification Prompts Issued (full text, for reference)

### 10.1 — First verification prompt (issued after docs 48–52)
> Before I treat any of the following as final, verify and report each item with the underlying artifact shown, not a prose summary. For every number below, show either the actual JSON/file contents or the exact console output it came from — if a number can't be traced to a file on disk, say so explicitly instead of presenting it as finished.
>
> 1. Full per-seed table, all 5 seeds (0–4), for all 4 variants (V4a-Gaussian, V4b-Gaussian, V4a-OU, V4b-OU): MAE mean±std, near-resonance %, low-amp %, on the same 1000-episode deterministic protocol used for V3/V4. Show seeds 1–4 individually, not folded into an average — I want to see the actual spread.
> 2. 5-seed aggregate for each of the 4 variants/regimes, placed next to V3's and V4's existing aggregates in one table.
> 3. Seed-to-seed variance sanity check: report the min–max spread across the 5 seeds for each variant/regime, and compare its magnitude to V4's own established spread (seeds ranged ~1510–2887 Hz). If any variant's 5 seeds agree suspiciously tightly (e.g., all within 1–2% of each other), say so explicitly rather than presenting a clean mean.
> 4. Full statistical suite on the now-complete 5-seed data: Mann-Whitney U statistic, exact p-value, and Cliff's delta effect size, run pairwise for V3-vs-V4a, V3-vs-V4b, V4-vs-V4a, V4-vs-V4b, in both Gaussian and OU.
> 5. The still-missing confound experiment numbers. Run A (V4 obs + V3 reward) and Run B (V3 obs + V4 reward) were reported as "5 seeds trained, checkpoints saved" with no MAE/near-resonance/low-amp numbers ever shown. Supply the full per-seed + aggregate + Mann-Whitney-vs-V4 results now, or state plainly that they still don't exist.
> 6. Re-derive the omega_rabi = 5000 Hz threshold width from the actual Rabi formula, show the arithmetic, and state the exact frequency offset where P(|1⟩) crosses 0.90. Compare it directly to the classical Lorentzian's ±3333 Hz. Confirm or retract the earlier claim that Qiskit-aligned's higher near-resonance number came from a "narrower" curve — that claim contradicted the ±3813 Hz figure computed earlier in this same project.
> 7. Confirm the PID baseline used a fixed, logged seed this run (not another unseeded grid search) — state the seed value and the resulting Kp/Ki/Kd.
> 8. Provenance check: for every result file touched by this run, confirm it carries a seed, timestamp, and git commit hash. List any file that doesn't.
> 9. Regenerate the sample-efficiency plot and the final ablation table (Table 3) now that all 5 seeds exist for V4a/V4b in both regimes.
> 10. State plainly, in one final list, anything that is still incomplete, approximated, timed out, or failed. Do not round a partial result up to "done."

### 10.2 — Second, consolidated verification + new-test prompt (issued after docs 53–55 — THIS IS THE CURRENT OUTSTANDING PROMPT, NOT YET ANSWERED)
> Stop and resolve the following before producing any final report. Show your work for each — file contents, console output, or code diffs, not prose summaries.
>
> **A. Integrity check — do this first, it may invalidate other results**
> 1. For every V4a and V4b run labeled "Gaussian," print the exact import statement and drift-model class/module used at training and evaluation time. Confirm explicitly whether it was `resonator_model_gaussian.py` or `resonator_model.py` (the OU one). If the Gaussian runs used the wrong module, say so plainly, and re-run them correctly.
> 2. Explain why V4a-Gaussian seed 4 (1265±309) and V4a-OU seed 4 (1269±307) — and V4b-Gaussian seed 3 (1068) and V4b-OU seed 3 (1068) — are near-identical or identical. If this is a bug, fix it and regenerate every affected number. If it isn't a bug, provide the raw per-episode data for both runs and show they are statistically independent samples.
> 3. Reconcile the direct contradiction between "all 8 Gaussian ablation seeds trained & evaluated" (one summary) and "Gaussian ablation batch hit the 6-hour timeout, seeds 1–4 not done" (two other summaries, same work). State the actual current status with a file timestamp as proof.
> 4. Explain why low-amplitude is exactly 0.0% for every V4a/V4b seed in both regimes, across 20 independent 1000-episode evaluations.
>
> **B. Still-outstanding from before**
> 5. Supply the full per-seed + aggregate + Mann-Whitney-vs-V4 results for confound Run A and Run B, or state plainly they still don't exist. "5 seeds trained, checkpoints saved" is not a result.
> 6. Re-derive the omega_rabi = 5000 Hz threshold from the Rabi formula, show the arithmetic, and confirm or retract the earlier claim that Qiskit-aligned's higher near-resonance came from a "narrower" curve.
> 7. Confirm the PID baseline used one fixed, logged seed this run — state it — not another unseeded grid search.
>
> **C. New experiments (literature-motivated — cite these in the writeup, don't just run them silently)**
> 8. **Filtered-derivative ablation (V4c)**: create a variant identical to V4a (raw Δamp) but with the gradient passed through the same EMA filter (α=0.3) already used for amplitude, before scaling. Train 5 seeds, Gaussian regime, same 1000-episode protocol. This tests whether V4a's failure is caused by noise amplification in an unfiltered derivative (Łakomy & Madonski, 2020; Cui et al., 2025) rather than scale alone.
> 9. **QACT shot-count sweep**: using the already-aligned, already-trained Qiskit policy (no retraining), re-evaluate at shots = 256, 1024, 4096, 16384 (1000 episodes each). Report MAE/near/low at each shot count. This characterizes the partial-observability/measurement-noise tradeoff flagged as a central open problem in current quantum-RL literature.
> 10. For the Discussion section, explicitly connect the V4a/V4b finding to feature-scale dominance in ML (cite Wongoutong 2024, Pinheiro et al. 2025) and connect the no-hardware limitation to the field-wide sim-to-real gap in quantum RL (cite Sivak et al. 2021, Reuer et al. 2023) — don't present these as isolated observations.
>
> **D. Final deliverable**
> 11. Once A–C are resolved, produce one Markdown file containing: every seed's individual result for every variant/regime (V3, V4, V4a, V4b, V4c, confound A, confound B, QACT zero-shot, QACT fine-tuned, QACT shot-sweep) in full tables — no folding into averages without also showing the per-seed row; every aggregate with mean±std and the Mann-Whitney U/p-value/Cliff's delta versus the relevant baseline; a provenance line per result (seed, git commit, timestamp).
> 12. Generate (or regenerate, if any input data changed due to fixes above) every figure: the V3/V4/V4a/V4b/V4c bar chart, the Gaussian-vs-OU comparison, the QACT transfer chart, the QACT shot-count sweep plot, the sample-efficiency curves, the confound comparison. For each, give the exact file path it was saved to.
> 13. End with one explicit list: everything still incomplete, approximated, timed out, unseeded, or unverified. No partial result gets rounded up to "done."

**NEXT STEP for whoever continues this conversation: wait for / request the response to prompt 10.2, then apply the same adversarial-review standard described in §9.3 before accepting anything.**

---

## 11. Consensus.ai Keyword Chains + Reference List (already delivered to Akash)

### 11.1 Keyword chains, grouped by paper section
- **Motivating context (real-hardware RL calibration):** "reinforcement learning superconducting qubit calibration"; "model-free quantum control reinforcement learning"; "PPO gate calibration superconducting qubit hardware"; "reinforcement learning quantum feedback control". Consensus query: *"Has reinforcement learning been used to calibrate real superconducting qubits?"*
- **Feature-encoding ablation (headline result):** "observation space augmentation reinforcement learning"; "derivative feature state representation continuous control"; "state-derivative augmentation PID analogue reinforcement learning"; "feature normalization sensitivity reinforcement learning". Consensus queries: *"Does adding derivative/rate information to an RL agent's observation space always improve control performance?"*; *"Can poorly-scaled input features degrade reinforcement learning policy performance?"*
- **Reward/observation confound:** "potential-based reward shaping reinforcement learning"; "reward shaping vs observation augmentation reinforcement learning"; "policy invariance reward transformation". Consensus query: *"Does reward shaping and observation augmentation have equivalent effects on reinforcement learning policies?"*
- **Noise regimes / PID comparison:** "Ornstein-Uhlenbeck process reinforcement learning control"; "reinforcement learning non-stationary drift tracking"; "PID controller variance disturbance rejection comparison"; "reinforcement learning vs classical control stochastic disturbance". Consensus query: *"How does PID control performance vary under different types of stochastic disturbance?"*
- **Cross-simulator transfer (semantic-misalignment finding):** "sim-to-real transfer reinforcement learning policy"; "domain randomization reinforcement learning"; "observation space mismatch policy transfer failure"; "zero-shot policy transfer reinforcement learning". Consensus query: *"What causes reinforcement learning policies to fail when transferred between simulators?"* (Note: the exact narrow framing — same-dimensionality, scrambled-semantics silent transfer failure — may return thin results; fall back to broader domain-randomization/sim-to-real terms.)
- **Reproducibility/statistics (Methodology):** "reproducibility reinforcement learning benchmarking"; "seed variance reinforcement learning evaluation"; "Mann-Whitney statistical significance reinforcement learning".

### 11.2 Reference list delivered (verify exact volume/page before final submission)
1. Fösel, T., Tighineanu, P., Weiss, T., Marquardt, F. "Reinforcement Learning with Neural Networks for Quantum Feedback." *Phys. Rev. X* 8, 031084 (2018). [NOTE: later Consensus search returned this same paper authored/cited as "Tighineanu, P., Weiss, T., & Marquardt, F. (2018)" — same paper, double-check author-order convention before citing.]
2. Sivak, V. V., et al. "Model-free quantum control with reinforcement learning." *Phys. Rev. X* 12, 011059 (2022). (Also appears as Sivak, V., Eickbusch, A., Liu, H., Royer, B., Tsioutsios, I., & Devoret, M. (2021) in one Consensus listing — **check whether this is the 2021 preprint vs 2022 published version, cite correctly.**)
3. Baum, Y., et al. "Experimental deep reinforcement learning for error-robust gate-set design on a superconducting quantum computer." *PRX Quantum* 2, 040324 (2021).
4. Ding, L., et al. (two-qubit fluxonium gate calibration paper). arXiv:2304.06087 (2023).
5. Porotti, R., Essig, A., Royer, B., Poilblanc, D., Marquardt, F. "Deep reinforcement learning for quantum state preparation with weak nonlinear measurements." *Quantum* 3, 173 (2019). — **Your closest simulation-only comparator; state explicitly where you exceed it (ablation depth, multi-seed reporting, classical baseline, dual-noise-regime testing, formal statistical testing).**
6. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017).
7. Lillicrap, T. P., et al. "Continuous control with deep reinforcement learning." *ICLR* (2016). — source of the Ornstein–Uhlenbeck noise process used in the OU regime.
8. Ng, A. Y., Harada, D., Russell, S. "Policy invariance under reward transformations: theory and application to reward shaping." *ICML* (1999). — cite alongside the confound (Run A/B) results.
9. Tobin, J., et al. "Domain randomization for transferring deep neural networks from simulation to the real world." *IROS* (2017).
10. Zhao, W., Queralta, J. P., Westerlund, T. "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: A Survey." *IEEE SSCI* (2020).
11. Itano, W. M., et al. "Quantum projection noise." *Phys. Rev. A* 47, 3554 (1993). — binomial shot-noise formula for P(|1⟩) measurement, relevant to the QACT shot-count sweep (§10.2 item 9).

**One action item still pending for Akash:** pull 2–3 concrete quantitative numbers (fidelity/error figures) directly from Porotti (2019) for a real numeric side-by-side comparison table — Claude explicitly flagged it does not have these numbers reliably enough to state them without risking fabrication.

---

## 12. New Literature Supplied By User (Consensus.ai "Research Gaps" Reports) — Used To Motivate New Tests

Two Consensus.ai gap-analysis reports were pasted by the user in the most recent turn. Key citations extracted and already incorporated into prompt 10.2's new-test section (item C):

### 12.1 From "Feature Scaling, Observers, and Normalization in Sim-to-Real Transfer"
- Łakomy, K., & Madonski, R. (2020). "Cascade extended state observer for active disturbance rejection control applications under measurement noise." *ISA Transactions*. — motivates the new **V4c filtered-derivative ablation** (tests whether unfiltered raw-derivative noise amplification, not just scale, explains V4a's failure).
- Cui, B., et al. (2025). "A Variable-Bandwidth Extended State Observer for Nonlinear Systems With Measurement Noise." *IEEE Trans. Industrial Electronics* 72, 1946–1957.
- Wongoutong, C. (2024). "The impact of neglecting feature scaling in k-means clustering." *PLOS ONE* 19. — generic ML evidence that a single dominant-scale feature corrupts algorithms, directly analogous to V4a's unnormalized Δamp.
- Pinheiro, J. M., et al. (2025). "The Impact of Feature Scaling in Machine Learning: Effects on Regression and Classification Tasks." *IEEE Access* 13, 199903–199931.
- Bollegala, D. (2014). "Dynamic feature scaling for online learning of binary classifiers." *Knowl. Based Syst.* 129, 97–105. (online/streaming-context relevance, lower priority.)
- Zhao, W., Queralta, J. P., Westerlund, T. (2020) — already in main reference list (§11.2 #10).
- Aljalbout, E., et al. (2025). "The Reality Gap in Robotics: Challenges, Solutions, and Best Practices." arXiv:2510.20808. (broad survey, useful for Limitations/Discussion framing, not a new experiment.)
- Zhang, Z., et al. (2024). "Zero-shot sim-to-real transfer using Siamese-Q-Based reinforcement learning." *Inf. Fusion* 114, 102664. (relevant methodology, out of scope to implement now — noted as a possible future-work citation.)
- Hong, J.-S., et al. (2026). "Bridging the Sim-to-Real Gap in Parallel-Link Leg Mechanisms via Simulator-Side Dynamics Normalization." arXiv:2608.01697. (out of scope to implement, future-work citation only.)

### 12.2 From "Research Gaps in Quantum Reinforcement Learning"
- Sivak, V., Eickbusch, A., Liu, H., Royer, B., Tsioutsios, I., & Devoret, M. (2021). "Model-Free Quantum Control with Reinforcement Learning." *Phys. Rev. X* 12, 011059. (Note: same underlying work as §11.2 #2 — reconcile citation year.)
- Reuer, K., et al. (2023). "Realizing a deep reinforcement learning agent for real-time quantum feedback." *Nature Communications* 14. — used to frame the no-hardware/control-latency limitation as a field-wide open problem (real quantum hardware needs sub-microsecond FPGA feedback; this project's ESP32/SPI loop is nowhere near that).
- Wright, E., & De Sousa, R. (2023). "Fast Quantum Gate Design with Deep Reinforcement Learning Using Real-Time Feedback on Readout Signals." *IEEE QCE 2023*, 1295–1303. — directly motivates the **QACT shot-count sweep** test (partial observability / measurement-collapse tradeoff is flagged as the field's most-cited unresolved gap).
- Tighineanu, P., Weiss, T., & Marquardt, F. (2018) — same as Fösel et al. 2018 (§11.2 #1), different author-order citation form; reconcile before citing both.
- Bukov, M., & Marquardt, F. (2026). "Reinforcement Learning for Quantum Technology." arXiv:2601.18953. (broad survey, good for Related Work framing.)
- Bukov, M., Day, A., Sels, D., Weinberg, P., Polkovnikov, A., & Mehta, P. (2017). "Reinforcement Learning in Different Phases of Quantum Control." *Phys. Rev. X* 8, 031086.
- Niu, M., Boixo, S., Smelyanskiy, V., & Neven, H. (2018). "Universal quantum control through deep reinforcement learning." *npj Quantum Information* 5.
- Sivak, V., et al. (2025). "Reinforcement Learning Control of Quantum Error Correction." (venue/details not fully given in the pasted excerpt — verify before citing.)
- Li, S.-Y., et al. (2025). "Robust quantum control using reinforcement learning from demonstration." *npj Quantum Information* 11.
- Ernst, J. O., Chatterjee, A., Franzmeyer, T., & Kuhn, A. (2025). "Reinforcement Learning for Quantum Control under Physical Constraints." arXiv:2501.14372.
- Jaouadi, A., & Ashhab, S. (2026). "Reinforcement learning for robust calibration of multiqudit quantum gates." *Phys. Rev. A*.
- Hailan et al. (2025) — cited multiple times in the pasted excerpt re: partial observability / measurement collapse; full citation details not fully resolved in the excerpt, verify on Consensus before using.

**General takeaway used to motivate new tests:** the dominant, most-cited open problem across both pasted literature surveys is the interaction between measurement/observation noise, feature scaling, and policy transfer under distribution shift — which maps almost exactly onto two things this project already has in hand: (1) the V4a/V4b raw-vs-normalized-derivative finding (feature-scaling-dominance framing), and (2) the QACT transfer (partial-observability/shot-noise framing). The two new tests added to prompt 10.2 (V4c filtered-derivative ablation, QACT shot-count sweep) were chosen specifically because they are cheap (one new variant trained once; one existing-checkpoint re-evaluation, no retraining) and because they give the paper a literature-grounded *mechanism*, not just an empirical correlation.

---

## 13. Memory System Note

This conversation has been running inside a claude.ai Project named "Quantum Resonator Control" (project ID referenced in the system context as `019cc82a-ab5c-72db-b3cd-981cc804c2bc`). Several facts from this conversation were already written into that Project's own memory files during the conversation:
- `overview.md` — updated with the V4 5-seed aggregate numbers (vs. Seed-3-alone), the QACT non-reproducibility finding, the "hardware fully out of scope" status, and the paper-reframing decision.
- `technical-notes.md` — updated with the V4a/V4b ablation numbers (Generation-2, pre-red-flag versions), the QACT observation-vector semantic-misalignment mechanism, the omega_rabi metric-strictness finding, and a note that the reward/observation confound ablation had not yet been run.

**If the new chat is opened inside the same Project, these files will be automatically available and should be treated as consistent with (a subset of) this handoff document.** If opened in a different Project or a regular (non-Project) chat, this handoff `.md` file is the only source of this context — rely on it fully.

**Important:** the memory files currently do NOT yet reflect the Generation-3 red flags (§9) or the suspicious V4a/V4b table (§4.3), since those were only just identified in the last couple of turns and have not yet been confirmed/resolved. Whoever continues this conversation should update project memory once the drift-model-import question (§7.1) and the confound numbers (§7.2) are finally resolved — not before, since memory should reflect confirmed facts, not still-disputed ones.

---

## 14. Immediate Next Steps (in priority order)

1. **Get Akash to run prompt 10.2 (§10.2)** against the agentic tool and paste the response back.
2. **Apply the same adversarial-review standard (§9.3)** to that response before accepting anything — specifically prioritize resolving: (a) whether the Gaussian ablation runs used the wrong drift model, (b) the still-missing confound Run A/B numbers, (c) the omega_rabi arithmetic contradiction, (d) the contradictory completion-status claims.
3. **Only once the above are resolved**, treat any V3/V4/V4a/V4b/confound/QACT numbers as final and move on to the manuscript-writing phase (Document C's §12 full outline, Document B's §4 reframed thesis).
4. **Resume the hardware-specifications document** (§5) if/when Akash asks for it again — pick up exactly where the datasheet research left off (BB135,115, OPA2134UA, BAT54-13-F, MCP4921, AD9834, MCP3202 still need real datasheet lookups).
5. **Resolve the two still-open design decisions**: the f0 mismatch (§7.4) and the DAC-vs-DDS reconciliation (§7.5) — both need an explicit choice from Akash, not just a recommendation.
6. **Once numbers are finally trustworthy**, pull 2–3 real quantitative figures from Porotti (2019) for the promised numeric comparison table (§11.2, pending action item).
7. **Maintain the established tone/working style** throughout: honest, direct, no sugar-coating, decision tables and comprehensive DOCX for technical deliverables, skeptical-by-default toward any agentic-tool-reported result until independently traceable to a file/artifact.
