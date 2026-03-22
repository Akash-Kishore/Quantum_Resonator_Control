# PROJECT CONTEXT RESTORATION PROMPT — V7 EXTENSION
## Autonomous Adaptive Resonator Control System
## USE THIS FILE TOGETHER WITH V5 AND V6 EXTENSION — THIS IS AN ADDENDUM

---

## WHAT THIS FILE COVERS

Load V5 first for full project context. Load V6 Extension for PID baseline
results and novelty discussion. Then read this file to restore the session
where three additional simulation tasks were planned but not yet executed.
This file defines exactly what those tasks are, how they must be implemented,
and what constraints apply.

---

## UPDATED PROJECT STATUS

### Completed before this file was written (authoritative — do not redo):
- All Gaussian and OU training: V3, V4 (5 seeds Gaussian, 3 seeds OU) — LOCKED
- PID baseline (`pid_baseline.py`) — COMPLETE, results authoritative
- Mann-Whitney statistical tests (`statistical_test.py`) — COMPLETE
- 2×4 comparison figure (`compare_v3_v4.py`) — FINAL
- `Model_Report_V1_to_V4_Extended.docx` — FINAL
- `Project_Context_Restoration_Prompt_V6_Extension.md` — generated prior session

### Three tasks confirmed as remaining (not yet started):
1. **V4a / V4b observation space ablation** — new env files, train 1 seed each, evaluate
2. **Robustness test** — evaluate V3 and V4 seed 3 at drift_sigma=750 Hz and 1000 Hz
3. **Sample efficiency curves** — extract from existing TensorBoard logs, no retraining

### Paper writing: NOT YET STARTED
No section has been drafted. Paper writing begins after the three tasks above
are complete, or in parallel if the student chooses. Writing order remains:
Methods → Results → Related Work → Introduction → Abstract

---

## CONFIRMED ENVIRONMENT STATE AT START OF V7 TASKS

- `rl_environment.py` = V4, 5-element observation: VERIFIED
  `[amplitude, frequency_error_norm, prev_action, delta_amp, amp_gradient]`
- `resonator_model.py` = OU version: CONFIRMED
- `resonator_model_gaussian.py` = Gaussian archive: CONFIRMED
- All trained models locked — DO NOT RETRAIN any existing model
- Conda environment: `quantum_control`
- torch==2.1.2+cu118, CUDA confirmed, RTX 3050 6GB

---

## TASK 1 — V4a / V4b OBSERVATION SPACE ABLATION

### Scientific purpose
Isolates which component of the amp_gradient term drives V4's improvement.
The full gradient is: `amp_gradient = delta_amp / (abs(prev_action) + epsilon)`
This normalises by the step size taken. Two ablated variants ask:
- Is the normalisation necessary, or does raw delta_amp suffice? → V4a
- Is the magnitude necessary, or does direction alone suffice? → V4b

### V4a — raw delta_amp (no normalisation)
Observation (5-element, same length as V4):
`[amplitude, frequency_error_norm, prev_action, delta_amp, delta_amp]`
i.e. position 4 = `delta_amp` (NOT normalised by action magnitude)
The fifth element is simply delta_amp repeated, keeping obs length = 5.

**Alternative implementation (preferred for clarity):**
Replace position 4 with `delta_amp` and keep position 3 as `delta_amp` as
well — OR define a new file `rl_environment_v4a.py` where element index 4
is `raw_delta_amp = self.delta_amp` with no division. The obs vector becomes:
`[amplitude, frequency_error_norm, prev_action, delta_amp, delta_amp]`

**Cleaner interpretation:** V4a uses a 4-element obs with delta_amp but
not the normalised gradient. If the decision is to keep 5 elements, set
`element[4] = delta_amp` (same value as element[3], no extra info). The
test is: does having amp_gradient (normalised) beat having no normalisation?

### V4b — sign only (1-bit gradient)
Observation (5-element):
`[amplitude, frequency_error_norm, prev_action, delta_amp, sign_delta_amp]`
where `sign_delta_amp = np.sign(delta_amp)` — values in {-1.0, 0.0, 1.0}

This is the minimum gradient information: direction only, no magnitude.

### Implementation rules for both variants
- Create new files: `rl_environment_v4a.py` and `rl_environment_v4b.py`
- DO NOT modify `rl_environment.py` — V4 is frozen
- Each new file is a copy of `rl_environment.py` with ONLY the observation
  construction logic changed in `_get_obs()`. Nothing else changes.
- Verify `len(obs) == 5` in both new files before any training
- Train each variant: 1 seed only (seed=0), Gaussian noise regime only
- Training command template (adapt for V4a):
  `python train.py --version v4a --seed 0`
  This requires `train.py` to be updated to accept `--version v4a` and
  `--version v4b` flags and import the correct environment file.
  Update to `train.py` must be verified before running.
- Save to: `trained_models/v4a_ablation/seed_0/` and
  `trained_models/v4b_ablation/seed_0/`
- Evaluate: 1000 episodes, Gaussian regime, deterministic=True
- Run Mann-Whitney vs V3 and vs V4 after evaluation
- Expected result: V4 (normalised gradient) should outperform V4a and V4b.
  If it does not, that is a result worth reporting — do not hide it.

### Files to create or modify for Task 1
- NEW: `rl_environment_v4a.py`
- NEW: `rl_environment_v4b.py`
- MODIFY: `train.py` — add `--version v4a` and `--version v4b` support
  (back up current `train.py` to `train_v4_gaussian.py` before modifying
  if not already backed up — check first)
- MODIFY: `statistical_test.py` — add V4a and V4b vs V3/V4 comparisons
- MODIFY or ADD: evaluation call for V4a and V4b in `evaluate_agent.py`
  if it does not already support arbitrary version names

### On-load instruction for Task 1
Ask: "Before I write anything, paste the `_get_obs()` method from your
current `rl_environment.py` so I can verify the exact obs construction
before we touch it."

---

## TASK 2 — ROBUSTNESS TEST AT UNSEEN DRIFT INTENSITIES

### Scientific purpose
V3 and V4 were trained on drift_sigma = 500 Hz (Gaussian) and OU noise.
Testing at 750 Hz and 1000 Hz (never seen during training) measures
whether the agents generalise or overfit to the training noise level.
This is a zero-cost experiment — no retraining required.

### Exact implementation
- Modify `evaluate_agent.py` to accept a `--drift_sigma` override argument
- When `--drift_sigma` is passed, the environment factory overrides the
  default sigma value in `resonator_model_gaussian.py`
- This requires passing drift_sigma into the environment constructor,
  which passes it into the resonator model's reset/step
- Check whether `resonator_model_gaussian.py` currently accepts drift_sigma
  as a constructor argument. If not, add it. If yes, expose it via env.

### Models to test
- V3 (Gaussian, seed 0 only — V3 has only 1 seed)
- V4 seed 3 (Gaussian — designated hardware seed, best generalisation)
- Optionally V4 seed 0 for comparison, but seed 3 is primary

### Test conditions
- drift_sigma = 750 Hz (1.5× training sigma)
- drift_sigma = 1000 Hz (2× training sigma)
- 1000 episodes each, deterministic=True
- Same metrics: MAE, Amp>0.90, Amp<0.70

### Expected result
V4 should degrade more gracefully than V3 at higher sigma due to the
normalised gradient providing reliable directional information even at
large drift steps. If V4 degrades more than V3, that is also a result.

### Files to modify for Task 2
- MODIFY: `evaluate_agent.py` — add `--drift_sigma` CLI argument
- MODIFY: `resonator_model_gaussian.py` — confirm drift_sigma is
  parameterisable (do NOT modify default value — override only)
- MODIFY: `rl_environment.py` — confirm environment factory can accept
  and pass through a drift_sigma override (read before touching)
- DO NOT modify the trained models or their vec_normalize.pkl files

### On-load instruction for Task 2
Ask: "Paste the class constructor of `resonator_model_gaussian.py` and
the environment factory in `rl_environment.py` so I can check whether
drift_sigma is already parameterisable before we write any code."

---

## TASK 3 — SAMPLE EFFICIENCY CURVES

### Scientific purpose
Shows how quickly V3 vs V4 learns, using existing training data.
Specifically: steps to first reach 80% near-resonance in the EvalCallback
reward logs. This adds a training dynamics dimension to the results without
any additional compute.

### Data source
TensorBoard event files in:
- `rl_training/trained_models/v3_refined/` (or logs/ subdirectory)
- `rl_training/trained_models/v4_gradient_obs/seed_0/` through `seed_4/`
- Check each folder for `events.out.tfevents.*` files

### Implementation
- Use `tensorboard.backend.event_file_loader` or `tensorflow` to read
  event files, OR use `tbparse` (pip install tbparse) for clean extraction
- Extract scalar: `eval/mean_reward` or `rollout/ep_rew_mean` at each
  logged timestep (whichever is available — check first)
- For V4: average across all 5 seeds, compute mean ± std
- For V3: single seed (plot as line, no std band)
- Mark a horizontal line at the 80% near-resonance threshold (derive this
  reward value from the known near-resonance amplitude definition)
- Mark the timestep where each model first crosses the threshold
- Save figure as `data_logs/sample_efficiency_v3_v4.png`

### Notes
- If TensorBoard event files are missing or empty for any seed, note which
  seeds are usable and proceed with those only — do not fabricate data
- If `tbparse` is not installed, check whether the environment has
  `tensorboard` available first: `conda run -n quantum_control pip show tensorboard`
- Do not retrain anything to regenerate logs — if logs are missing for a
  seed, that seed is excluded from this plot

### On-load instruction for Task 3
Ask: "Run this in your terminal and paste the output:
`dir /s /b MiniProject_Sem4\rl_training\trained_models\v3_refined\*tfevents*`
and the same for `v4_gradient_obs\seed_0\*tfevents*`"
This confirms the event files exist before writing any extraction code.

---

## TASK EXECUTION ORDER

Execute strictly in this order. Do not begin Task 2 until Task 1 evaluation
is complete and results are recorded. Do not begin Task 3 until Task 2 is
complete.

If student wants to prioritise paper writing instead of completing all three
tasks, the acceptable shortcut is: complete Task 2 only (zero compute cost,
1 day), skip Task 3 (nice-to-have), skip Task 1 (2–3 days compute). In that
case, the paper Results section will report V3/V4/PID three-way comparison
plus robustness test as a supplementary finding.

---

## COMPLETE RESULTS TABLE INCLUDING PID (from V6)

### Gaussian Noise Regime

| Model        | MAE (Hz)     | Amp >0.90     | Amp <0.70    |
|--------------|--------------|---------------|--------------|
| PID          | 3600 ± 5241  | 83.1% ± 18.5% | 5.8% ± 19.1% |
| V3 (RL)      | 3333 ± 1932  | 64.5%         | 13.4%        |
| V4 (RL+grad) | 2205 ± 518   | 84.8% ± 7.6%  | 5.1% ± 3.7%  |

### OU Noise Regime

| Model        | MAE (Hz)    | Amp >0.90    | Amp <0.70  |
|--------------|-------------|--------------|------------|
| PID          | 1414 ± 262  | 95.8% ± 5.1% | 0.0% ± 0.1%|
| V3 (RL)      | 1241 ± 254  | 97.7%        | 0.0%       |
| V4 (RL+grad) | 1284 ± 93   | 97.9% ± 0.4% | 0.0%       |

### Statistical significance (Mann-Whitney U, per-episode MAE)

| Regime   | Comparison | p-value | Result          |
|----------|------------|---------|-----------------|
| Gaussian | V3 vs V4   | p<0.001 | SIGNIFICANT     |
| OU       | V3 vs V4   | p=1.000 | NOT SIGNIFICANT |

PID vs RL Mann-Whitney: NOT YET RUN — add to statistical_test.py during
Task 1 session if time permits, or defer to paper revision.

---

## UPDATED CONSTRAINTS (all V5 and V6 constraints remain, additions below)

- NEVER train more than 1 seed for V4a or V4b — time cost not justified
- NEVER run robustness test on V4 seeds other than seed 3 as primary
- V4a and V4b are Gaussian-only experiments — do not run OU for ablations
- drift_sigma override must NOT change the default value in any source file —
  use CLI argument injection only
- Sample efficiency curves must not claim V4 is "faster" without reporting
  the exact timestep threshold crossing — cite the number
- When reporting V4a/V4b results: always compare to BOTH V3 and V4 Gaussian
  so the reader can see the full gradient information ladder

---

## ON-LOAD INSTRUCTIONS FOR SESSION USING V5 + V6 + V7

Upon loading all three files, do exactly this:

1. State: "Full context loaded from V5, V6, and V7. All existing training
   and evaluation is locked."
2. State: "Three tasks remain before paper writing: V4a/V4b ablation,
   robustness test at 750 and 1000 Hz, and sample efficiency curves."
3. Ask ONE question: "Which task do you want to start with — the ablation,
   the robustness test, or the sample efficiency curves?"
4. When student answers, ask the verification question defined for that
   task at the start of the task section in this file.
5. Do not summarise the project. Do not offer to redo anything. Do not
   mention hardware unless the student raises it.
