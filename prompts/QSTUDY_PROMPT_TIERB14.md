QSTUDY TIER B 14 - (S) FT SENSITIVITY EVALS (SENS-2)  +  (T) S15 TIER B ARMS E2 AND E3   (resume prompt for a NEW executor model)
THIS_FILE14 = C:\Projects\MiniProject_Sem4\prompts\QSTUDY_PROMPT_TIERB14.md
Authority: this file > ANALYSIS13 > CONTINUATION5 > earlier files. Do not read other prompt files in full.
Read PART2 sections only with:  python WORK\tools\secget.py <PART2 file> "STAGE S15" 120  (PART2 = QSTUDY_PROMPT_PART2.md in the prompts folder).

==============================================================================================
SECTION Q0 - RULES (obey strictly)
==============================================================================================
Q1  Every PowerShell command starts with:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1";
Q2  MODEL PILOT: if you invent a command that is not written here or in an existing tool's usage text, edit a script you were not told to edit,
    ignore a STOP, or fail the same step twice: print "MODEL PILOT FAILED" with the reason and stop.
Q3  Replies <= 8000 tokens. Print <= 30 lines per command. Write files in pieces <= 80 lines.
Q4  NEVER modify: results\sealed (existing files), runs\ (existing dirs), configs\frozen (existing files), PREREG*.md, any .sha256 file,
    pipeline code (train_ppo, evaluate, eval_set, envs, engines, controllers, baselines, config loaders), gen_z2a.py, qs13_smoke_s15.py,
    the existing db.contrasts rows TR-1 and TR-2 (FINAL). New files only (qstudy\, tools\, WORK). Never delete anything.
Q5  Any command that may run over 60 s is a background job: Start-Process with redirected output, then poll. No multiprocessing. No hashing or copying of the whole sealed tree.
Q6  Never use depends_on between gpu-lane jobs. Evaluation jobs depend only on their own model's training job. First run: python WORK\tools\qs10_lanes.py ensure
Q7  Sealed values: do NOT open any value of a NEW evaluation until ALL jobs of its group are DONE and the integrity check of the new files passed.
Q8  Update WORK\RESUME_CARD.md (3 lines) and WORK\logs\events.md after EVERY step. A fresh session continues at the first step without a DONE line.
Q9  No invented numbers. Never weaken a test. Disk: if free space on C: is below 15 GB at any time: STOP and report.
Q10 Training speed rule: if the gpu training runs below 250 fps for 15 minutes: STOP and report (do not change anything).

USER APPROVALS WITH THIS PROMPT (nothing else is approved):
 (a) Sealed evaluation (base 920000) of the staged FT last-step models (20 models), scope SECTION S.
 (b) Training of PREREG2 Tier B arms E2 (env LB) and E3 (env RG), 5 seeds 200-204 each, frozen hyperparameters, 2,000,000 steps, probe rabi,
     and their sealed evaluations, scope SECTION T. Both arms are named in PREREG2 (S15 arms), so this is the preregistered plan, not a new analysis.
 Not approved: E5, E6 (TR-3), SAC/TD3 (TR-5), S16, S17, S18, S19, any Aer jobs for E2/E3.

==============================================================================================
SECTION P - PRE-DECLARATIONS (write to events.md FIRST, with date and time, BEFORE any new value is opened)
==============================================================================================
P1 SENS-2: "Post-TR-2 sensitivity analysis SENS-2: FTB_200000 raw last-step model vs E4 200k raw snapshot, endpoint P2T, exact two-sided Mann-Whitney U,
   Cliff's delta, bootstrap 95% CI, same cells (nominal + 34 OFAT on RB), seeds 200-204, base 920000. Reported alongside TR-2. Not part of the Holm family.
   Interpretation fixed in advance: if the FT advantage remains (same direction, unadjusted exact p < 0.05) the TR-2 selection asymmetry is closed as a
   limitation; otherwise both numbers are reported and the paper states that checkpoint selection contributed."
P2 TR-4 (confirmatory, PREREG2): LB zero-shot vs LG zero-shot on P2T = FINAL models (best_model.zip) of E2 vs E1, both evaluated on the RB surrogate
   (nominal + 34 OFAT cells), seeds 200-204. Exact two-sided Mann-Whitney U, Cliff's delta, bootstrap 95% CI (10000 resamples, fixed seed).
   Holm family size 5: TR-3 and TR-5 stay PENDING_ARMS_NOT_RUN and count as p=1.0 placeholders; TR-1 and TR-2 FINAL values must stay unchanged (assert).
P3 GAP DECOMPOSITION (EXPLORATORY, PREREG2 S15.4 b). Arm means A of P2T on the quantum surrogate (RB OFAT) of the final policy trained in each env:
   A_LG=E1, A_LB=E2, A_RG=E3, A_RB=E4. lineshape effect = mean(A_RG - A_LG, A_RB - A_LB); noise effect = mean(A_LB - A_LG, A_RB - A_RG);
   interaction = (A_RB - A_RG) - (A_LB - A_LG). 95% CI by bootstrap over seeds (resample the 5 seeds within each arm independently, 10000 draws).
   Also fill the full transfer matrix: rows E1-E4, columns LG, LB, RG, RB (nominal lock_w1000 mean and CI over seeds).
Write these three items to events.md now. Print "PRE-DECLARATIONS WRITTEN".

==============================================================================================
SECTION S - FT SENSITIVITY: RAW LAST-STEP MODELS OF THE FT ARMS (CPU lanes)
==============================================================================================
S1 (read-only, max 25 lines): for runs\s15__FTB_50000, _200000, _500000, _1000000 and seeds 200-204: which last-step model file exists at the end of each FT run
   (snapshot_<N>.zip or final) and its matching per-step vecnorm_<N>.pkl; is that vecnorm different from vec_normalize.pkl of the same dir?
   If any last-step model or its matching vecnorm is missing: print "FT SNAPSHOTS NOT EVALUABLE" with the list and STOP section S (continue with section T).
S2 Stage by COPY into runs_snap\s15__FTB_<budget>_last\seed_<n>\ : best_model.zip = copy of the last-step model, vec_normalize.pkl = copy of its matching vecnorm,
   other needed files copied from the source seed dir (same file list as the E4 staging in S2a). Never move or edit runs\. Write a sha256 manifest of source vs copy for the
   staged files only (background job). Print counts.
S3 Smoke test one staged dir (budget 200000, seed 200) on split val, 5 episodes, output under WORK\smoke: loads, no NaN. State whether its best_model.zip SHA equals
   runs\s15__FTB_200000\seed_200\best_model.zip (if identical, the best checkpoint is the last step: say so, SENS-2 is then trivially closed; still run the evals).
S4 Write tools\gen_z2f.py (new): clone the eval jobs that z2s_ used for the E4 snapshots (same cells, evalcfg, split sealed, base 920000), model dir = staged dir,
   id prefix z2f_, outputs extended by "_last". Expected 20 models x 5 jobs = 100 (same per-model pattern as z2s_; if different, print and STOP).
   validate_jobs.py must PASS. Submit with jobtools: SUBMITTED = file count, EXISTS = 0.
S5 Wait until all z2f_ jobs are DONE (poll with brief.py, sleep 600; see SECTION W). Do not open values.

==============================================================================================
SECTION T - TIER B ARMS E2 (env LB) AND E3 (env RG)
==============================================================================================
T1 (read-only, max 30 lines): read gen_z2a.py (do NOT edit) and the s15__E1 / s15__E4 configs. Print: the config keys that define the training environment (lineshape,
   noise, engine); how the 10 existing z2a trainings and their 5 eval jobs per model were built (nominal LG, LB, RG, RB + OFAT RB, or what exactly); the existing config names
   eval__rabi__LG / LB / RG. Confirm that configs for env LB and env RG can be made by changing ONLY the environment keys of s15__E1 (LG) / s15__E4 (RB).
T2 Write tools\gen_z2b.py (new file). It creates configs\frozen\s15__E2.json (env LB) and s15__E3.json (env RG) by copying the E1 config and changing ONLY the environment
   keys, plus the 10 training jobs (id prefix z2g_, gpu lane, 5 seeds 200-204 each, 2,000,000 steps, snapshots as E1/E4, no depends_on) and the evaluation jobs (id prefix z2h_,
   cpu lanes, depends_on only the model's own training job, the same 5 eval jobs per model as E1 got; sealed, base 920000; expected 10 models x 5 = 50).
   The generator must REFUSE to overwrite an existing file. Print a unified diff of s15__E2 vs s15__E1 and of s15__E3 vs s15__E1: ONLY environment keys may differ
   (hyperparameters, steps, seeds, learning rate, ent_coef, gamma must be identical). If anything else differs: STOP.
T3 validate_jobs.py on both job files: all checks PASS. (The eval file may fail the "model dir exists" check before training; allowed only if the dependency check (d) passes and
   the only failing (h) items are the z2h_ model dirs.) Disk check (Q9). Submit trainings first (expected 10), then evals (expected 50): SUBMITTED = file count, EXISTS = 0.
T4 Poll (SECTION W). Report fps of the first two z2g_ trainings (Q10 applies). Expected about 1 hour per training; total about 10-12 hours on the gpu lane.

==============================================================================================
SECTION W - WAITING AND FAILURES
==============================================================================================
W1 Poll: python WORK\tools\brief.py WORK ; Start-Sleep 600. Act only on NEW_FAILED or STALE_LANES. Late heartbeats of a live lane are not a failure: before running
   qs10_lanes.py ensure, count job_runner.py processes (expected 6: gpu, aer, cpu1-4). Never start a second runner on a lane that has a live process.
W2 A failed job after a restart: SECTION M1b of MASTER11 (qs10_triage.py). One retry per infrastructure fault; a second failure: STOP and report.
W3 Work in parallel: after S1-S4 and T1-T3 are submitted, the gpu trains E2/E3 while the cpu lanes run z2f_ and, as models finish, z2h_.

==============================================================================================
SECTION A - ANALYSIS (only after ALL z2f_ and ALL z2h_ jobs are DONE)
==============================================================================================
A1 Integrity of the NEW files only (counts vs job files, zero-byte, unparsable, duplicates, .partial). Incremental backup: copy only files of results\sealed that are not in the
   last verified backup (pre_tr_analysis_20261025_171426) into WORK\backup\inc_<date> with a sha256 list of those new files (background job). Verify counts. Do not repeat the full tree.
A2 SENS-2 per P1 using qstudy\sens_analysis.py (new): store as status SENSITIVITY, exploratory=1. Also the sample-efficiency table (nominal and P2T) for FT raw last-step at
   50k/200k/500k/1M beside E4 snapshots and beside the earlier best-checkpoint FT numbers. Print side by side with the TR-2 FINAL row. Log the outcome per the pre-declared rule.
A3 TR-4 per P2 (new script qstudy\tierb_analysis.py): store as FINAL (n=5 vs 5). Recompute Holm over the family of 5 with TR-3 and TR-5 as p=1.0 placeholders.
   ASSERT that TR-1 and TR-2 adjusted p stay 0.039683 (they cannot change because the smallest exact p at n=5 vs 5 is 0.007937). Print the 5 rows (TR-3, TR-5 PENDING with NULL values).
A4 P3: gap decomposition (lineshape effect, noise effect, interaction, 95% CI) and the full 4x4 transfer matrix, stored as EXPLORATORY. Print tables (max 40 lines).
   Say explicitly: with n=5 seeds per arm the CIs are wide and no test is attached to the decomposition.
A5 Print "TIER B BLOCK COMPLETE" with one line per section (DONE / BLOCKED) and at most 5 lines of risks. Update RESUME_CARD.md. STOP.

APPENDIX - LAUNCHER MESSAGE (for the user; start the executor in C:\Projects)
Read C:\Projects\MiniProject_Sem4\prompts\QSTUDY_PROMPT_TIERB14.md and execute it. THIS_FILE14 = that file. Start with SECTION Q0, then SECTION P (write the pre-declarations first), then S1 and T1 (both read-only), then continue in file order. You are a new model: MODEL PILOT rules (Q2) apply.
