QSTUDY CONTINUATION 4 - FREEZE RESOLUTION   (executor: GLM 5.3)
THIS_FILE4 = path given in the launcher message. Earlier files in the same folder: QSTUDY_PROMPT.md (Part 1), QSTUDY_PROMPT_PART2.md (Part 2), QSTUDY_PROMPT_CONTINUATION3.md (lean mode). They remain authoritative.
TOKEN RULES: obey Section L0 (K1-K9) of CONTINUATION3: read it with  python WORK\tools\secget.py <CONTINUATION3 file> "SECTION L0"  . Do NOT read the chat exports or Part 1/Part 2 in full; use secget. Print at most 40 lines per command.

==============================================================================================
SECTION M0 - BOOT (lean, 4 steps)
==============================================================================================
1. cwd must end in MiniProject_Sem4; derive PARENT, NAME, WORK, STUDY; read WORK\00_paths.txt; if WORK is missing STOP "WORK NOT FOUND".
2. Read WORK\RESUME_CARD.md. Then OVERWRITE its "Next actions" and "Controller approvals received" sections with the patch of Appendix J (verbatim).
3. python WORK\tools\state.py WORK show | tail -15 ; python WORK\tools\brief.py WORK ; relaunch stale lanes/watchdog if any (Part 1 Stage 0.7).
4. state.py set BOOT4 DONE. Continue with SECTION M1.

==============================================================================================
SECTION M1 - CONTROLLER DECISION ON THE FREEZE
==============================================================================================
The review pack (STUDY\docs\freeze_review_pack.md) was reviewed. Decision: CONDITIONAL APPROVAL. The freeze is NOT approved as drafted. Do M2 (fixes), then M3 (acceptance checks). If ALL checks pass, proceed WITHOUT asking me: write the addendum and hashes (M4) and start the training queue (M5). If any check fails after two fix cycles, or the selection changes (M3 rule S), STOP and print exactly: FREEZE-BLOCKED - AWAITING CONTROLLER, followed by the failing checks (max 40 lines).
Write every fix and finding as a line in WORK\logs\events.md. Do not write any hash, addendum or sealed result before M4.

==============================================================================================
SECTION M2 - FIXES (all pre-freeze; code and exploratory validation only; no sealed split)
==============================================================================================
F1 BAYES PARAMETER ISOLATION (critical). In controllers.py and wherever evaluate.py builds the filter:
   - ctrl:bayes:nominal may use ONLY the frozen nominal assumption N0 = {drift: random walk theta 0, sigma 500 Hz, tau 50e-6, t2 200e-6 exponential, e01 = e10 = 0.02, omega 5000, dither_delta 2500} plus OBSERVABLES (counts, shots used this step, fc, probe type). It must NOT read any other attribute of the evaluation cell's cfg (sigma, t2, readout error, contrast power/override, drift kind, delay, DAC bits).
   - ctrl:bayes:oracle uses the cell's TRUE values: sigma, t2/contrast, e01, e10, tau, omega; theta 0.05 if drift is ou; for onef/telegraph/sinusoid/regime drifts use a random walk whose sigma equals the empirical per-step std of that drift (measure it once per cell from 2000 simulated steps with the cell's cfg).
   - evaluate.py must write "assumed_params" (the dict actually used by the filter) into every result JSON for ctrl:bayes:*.
   - Test (tests\test_bayes_isolation.py): for each probe in {rabi, dither, ramsey_iq} and cells ofat_drift_sigma_1000, ofat_t2_50, ofat_readout_0.2: assumed_params of nominal differ from oracle; nominal's assumed_params equal N0 in ALL cells; 100-episode results of nominal and oracle are NOT identical to 4 decimals in the sigma_1000 and readout_0.2 cells for dither and ramsey_iq.
F2 HYBRID FILTER FEATURES MUST USE N0. BatchEnv filter features (cfg filter_features=True) must use N0 (same dict as above) in training AND evaluation, never the episode's true sigma/t2/readout (the old default filter_sigma=None meant "true": remove that behaviour; allow an explicit cfg filter_assumed="oracle" only for labelled diagnostics).
   - Audit (script audit_filter_params.py): on one sample episode per config family print the parameters the filter actually used in training and in a sigma_1000 eval cell. Families: rabi/dither/ramsey_iq filter, dither/ramsey_iq full.
   - If the audit shows the NON-DR filter configs (trained at nominal where true = N0) are unaffected: keep them, but RE-EVALUATE their validation cells with the fixed code (they were evaluated in stress cells with true parameters). If any DR-trained filter config (the "full" family) used per-episode true parameters in training: mark ALL exploratory "full" results INVALID_LEAK (keep the files, exclude them from rankings), and define C6 = rl__dither__full to be trained in the confirmatory phase with the corrected filter (no exploratory retraining, no selection from the invalid results).
F3 SELECTION RECOMPUTATION. Recompute every val score in configs\frozen\selection.json from the database using ONLY valid runs (exclude every *_INVALID* directory and every result tagged INVALID). Print a table of old vs new val_mean/val_sd per config and the mismatches (known: rl__dither__plain 0.8149/0.0508 in selection.json versus 0.8322/0.0169 in T1: find and explain the stale input). The recomputed values must equal the T1 table of the review pack within 1e-6 for every config (after F1/F2 re-evaluations, recompute T1 as well).
F4 EVALUATION CONFIG AUDIT (class of bug A10/GRU-cfg). Extend evaluate.py to write the EFFECTIVE configuration (after all overrides) and the cond_id into every result JSON, and write STUDY\tools\audit_eval_cfg.py which, for every result file, checks that (a) the recorded cfg equals the definition of its cond_id in stress.py, (b) for RL models the recorded probe/stack/memory type equals the model's training config, (c) no two cells of the same method have identical cfg when their cond_ids differ. Run it on ALL existing validation results; every mismatch makes that file INVALID and it is re-run. Add a pre-flight check (before running) and post-flight check (after running) to evaluate.py: an inconsistent cfg aborts the job with exit code 4. Sealed evaluations may be re-run ONLY after an abort by this check (infrastructure fault), never because of their results; log each case.
F5 ORACLE REFERENCE ROWS. Evaluate ctrl:oracle (measured, not analytic) on the 4 val cells and all three probes (200 episodes, val seeds) and add them to the T2 table with a "ceiling" column. Check I5 of Part 1: at nominal the oracle lock_w1000 must be 0.954 +- 0.02, and no method may exceed the oracle of the same cell by more than 0.02 (otherwise record an ANOMALY).
F6 TUNING REPLICATE. The tuning table has a top-2 margin (0.8298 vs 0.8145) inside seed noise. Run 3 replicate tuning seeds (90, 91, 92) for the top-2 configs (lr 3e-4, gamma 0.99, ent_coef 0.003 and 0.0), 1M steps, probe dither plain, gpu lane, no depends_on. Choose the hyperparameters by the 3-seed mean val score; if the difference is smaller than the larger of the two sds, choose ent_coef 0.0 (the value used in the exploratory runs). Record the table in docs\tuning.md.
F7 RE-RUN THE AFFECTED BASELINES AND HYBRIDS (cpu lanes, val seeds, 200 episodes, 4 val cells): ctrl:bayes:nominal and ctrl:bayes:oracle for the 3 probes; RL filter configs (rabi, dither, ramsey_iq; seeds 0-4); then recompute T1/T2.
F8 REGENERATE THE REVIEW PACK as STUDY\docs\freeze_review_pack_v2.md (tables T1, T2 with oracle rows, T3 with the replicate, T4, T5, T6 with the new checks) and print only its path and size.

==============================================================================================
SECTION M3 - ACCEPTANCE CHECKS (all mechanical; print one compact table, max 30 lines)
==============================================================================================
A1 test_bayes_isolation passes; nominal/oracle differ as required (F1).
A2 audit_filter_params shows the filter parameters equal N0 in all stress cells for hybrids (F2); the status of the "full" family is recorded (VALID or INVALID_LEAK).
A3 recomputed selection.json val scores equal the regenerated T1 within 1e-6 for every config (F3).
A4 audit_eval_cfg reports 0 mismatches over all result files after re-runs (F4).
A5 oracle rows exist and satisfy the ceiling rule (F5).
A6 the full unit-test suite is green (job on cpu lane).
RULE S (selection stability): after F1-F7 recompute the draft selection with the same rules. If C1, C2, C3, C4 or C5 changes, or the C2 margin to the runner-up becomes smaller than 1 sd of the runner-up, STOP with FREEZE-BLOCKED and print the new tables. C6 is not selection-relevant (no contrast uses it).
Cycle limit: at most two fix cycles per failing check, then FREEZE-BLOCKED.

==============================================================================================
SECTION M4 - WRITE THE FREEZE (only if M3 passed)
==============================================================================================
1. Write configs\frozen\selection.json (status FROZEN) with: the selection C1-C6; the hyperparameters from F6; the nine confirmatory configs of Appendix K; baselines and their definitions.
2. Append the FREEZE ADDENDUM of Appendix K (verbatim, filling the bracketed fields from the database) to STUDY\PREREG.md, then write PREREG.sha256 (sha256 of PREREG.md) and insert it into the database. From now on PREREG.md and configs\frozen\ are read-only for you.
3. Write STUDY\PREREG2.md = PART2 "APPENDIX E" verbatim, then apply the PREREG2 AMENDMENTS of Appendix K in place (they are part of the frozen text), then PREREG2.sha256 and insert into the database.
4. Print: FREEZE-HASHED | PREREG sha256 prefix 12 chars | PREREG2 sha256 prefix 12 chars | the nine confirmatory configs. (This is an informational message, not a stop.)

==============================================================================================
SECTION M5 - START THE CONFIRMATORY AND PART-2 TRAINING QUEUE (no gpu depends_on, ever)
==============================================================================================
Job id prefixes control the order (the runner takes jobs alphabetically): z1_ (Part 1 Stage 12), z2_ (S15), z3_ (S16), z4_ (S17). Submit ALL training jobs up front with one JSON file per prefix, then the evaluation jobs per model (each depends only on its own training job).
z1_ Part 1 Stage 12: the nine confirmatory configs x seeds 101-110 (90 runs) with the frozen hyperparameters. Sealed evaluations (nominal 1000 episodes; all OFAT cells 300 episodes; grid cells 100 episodes; Aer subset for seeds 101-103) per Part 1 Stage 12 using seed base 900000; plus all baselines on the same cells. Read PART1 "STAGE 12" once.
z2_ S15, z3_ S16, z4_ S17: as in PART2 (read each stage section with secget when you reach it) with these changes: use the amended PREREG2; the C1-C6 envelope/platform/Aer seeds are 101-103; the training steps/seeds of Part 2 are unchanged.
During waiting follow K5 (brief.py + sleep 270). Update RESUME_CARD at milestones. Do not stop.
Remaining coding (S17.2 CUSUM, S17.5 fallback, S17.3 calibration, S19 complexity.py) continues in the gaps as in CONTINUATION3 W1-W2; missing exploratory documents (W3) are written after the ingest; evaluate sealed results only after their frozen hashes exist (they do after M4).

==============================================================================================
APPENDIX J - RESUME CARD PATCH (overwrite these two sections of WORK\RESUME_CARD.md)
==============================================================================================
## Next actions (in order)
1. CONTINUATION4: fixes F1-F8, acceptance A1-A6 + rule S; if all pass: write freeze (M4) and submit the queue (M5). Else FREEZE-BLOCKED - AWAITING CONTROLLER.
2. Then continue the lean routine: wait with brief.py; S17.2/S17.5/S17.3 code; S19 complexity.py; docs; sealed evaluations; S18; S20; Part 1 Stage 13-14.
## Controller approvals received
- Freeze gate reached; review pack reviewed by the controller; decision: CONDITIONAL APPROVAL subject to CONTINUATION4 Section M2-M3 (confirmatory seeds become 101-110; nine confirmatory configs; Bayes/filter parameter isolation fixes; tuning replicate). No unconditional approval yet.

==============================================================================================
APPENDIX K - TEXTS TO WRITE AT M4
==============================================================================================
=== FREEZE ADDENDUM (append to PREREG.md; fill bracketed values from the database) ===
## FREEZE ADDENDUM (written [date]; supersedes any conflicting text above)
1. Hyperparameters (tuning rule with 3-seed replicate, docs\tuning.md): learning_rate [lr], ent_coef [ent], gamma [gamma], n_envs/n_steps/batch/epochs as in train_ppo defaults, 2,000,000 steps.
2. Confirmatory seeds are 101-110 (seed 100 was used for tuning; every mention of 100-109 above means 101-110).
3. Confirmatory configs, 10 seeds each: rl__rabi__plain, rl__rabi__gru8, rl__dither__plain, rl__dither__filter, rl__dither__DR, rl__dither__DRcurr, rl__dither__full, rl__ramsey_iq__plain, rl__ramsey_iq__gru8. Mapping: C1=rl__rabi__plain (legacy analog), C2=[C2], C3=[C3], C4=[C4], C5=[C5], C6=rl__dither__full (probe of C2; trained only in the confirmatory phase with the corrected filter features because exploratory "full" results were [VALID|INVALID_LEAK]). Contrast sides: C-1 dither plain vs rabi plain; C-2 ramsey plain vs dither plain; C-3 dither plain vs ctrl:pid(dither); C-4 dither plain vs ctrl:bayes:nominal(dither); C-5 dither filter vs dither plain; C-6 rabi gru8 vs rabi plain; C-7 dither DRcurr vs dither plain; C-8 dither DRcurr vs dither DR.
4. Baseline definitions: ctrl:bayes:nominal assumes N0 (random walk, sigma 500 Hz, tau 50 us, T2 200 us exponential, readout 0.02/0.02, omega 5000, dither delta 2500) and sees only observables; ctrl:bayes:oracle uses the true cell parameters (moment-matched random walk for non-Gaussian drifts); hybrid filter features use N0 in training and evaluation; ctrl:oracle is a measured reference (previous true offset known); ctrl:pid gains are the frozen per-probe tuned gains; ctrl:hill is rabi only.
5. Ties and margins recorded before freezing: C6 margin to the runner-up was below one sd (no contrast uses C6); C-6 uses rabi gru8 although its margin to rabi K4 (0.020) was below one sd; memory variants did not improve over plain in exploration; fixed-from-start domain randomization underperformed plain in exploration. These are reported, not hidden.
6. Protocol deviations before freezing (details in events.md and docs\03_session_handover.md): anomalies A1-A11, the readout-OFAT invalidation (A10), the curriculum bug (A8), the filter/Bayes parameter-isolation fix (M2 F1-F2), the tuning replicate (F6), the seed shift (item 2).
7. Re-run policy: a sealed evaluation may be re-run only if the pre-flight/post-flight configuration audit (audit_eval_cfg) aborts it (infrastructure fault); never because of its results; every case is logged.
8. Compute estimate and order: z1_ confirmatory first, then S15-S17 trainings; evaluations follow each model.
=== PREREG2 AMENDMENTS (apply inside PREREG2.md before hashing) ===
A. Everywhere seeds "100-102" appear for C1-C6 (S16 zero-shot platform evaluation, S17.6 envelope, Part 1 12.3 Aer subset) read 101-103.
B. Fine-tune arms: FT-A uses the frozen hyperparameters; FT-B uses learning_rate = frozen/10 and frozen observation-normalisation statistics (no updates) during fine-tuning. Both arms are run for every budget. TR-2 (primary) uses FT-B; FT-A is reported as secondary. Reason (pre-freeze exploratory evidence): Aer fine-tuning with the frozen learning rate lowered lock for 2 of 3 seeds.
C. Bayes baselines in S16/S17 follow the Part-1 addendum item 4 (nominal assumes N0, never the platform parameters; oracle uses the platform/cell parameters).
D. Everything else in PREREG2 is unchanged.