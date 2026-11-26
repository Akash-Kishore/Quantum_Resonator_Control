QSTUDY CONTINUATION 5 - RECOVERY, QUEUE REPAIR, GATED EXECUTION   (executor: GLM 5.3)
THIS_FILE5 = path given in the launcher message. Authority order: this file > CONTINUATION4 > CONTINUATION3 > PART2 > PART1.
Earlier files stay in force EXCEPT where this file says otherwise. Where this file is silent, CONTINUATION4 applies.
Read this file with:  python WORK\tools\secget.py <THIS_FILE5> "SECTION P0" 80   (one section at a time). Do not read it whole twice.

WHY THIS FILE EXISTS (controller diagnosis, facts from the previous session export)
- The previous session ran 164 model steps in 12.8 h; median step 217 s even for trivial commands; 26 steps took longer than 300 s;
  the last two steps died with ProviderHeaderTimeoutError (300000 ms) after about 1860 s each. Cause: the context had grown to hundreds of
  thousands of tokens (every edit/write also returned several KB of LSP noise) and reasoning was set to maximum.
- All 6 lane runners (gpu, aer, cpu1-cpu4) and the watchdog are DOWN again.
- Last verified state: M0-M4 DONE (freeze written and hashed), M5 started: tools\gen_z1.py ran and wrote jobs_json\z1_train.json (90),
  z1_eval.json (270), z1_base.json (45), z1_aer.json (26). NOTHING of z1 was submitted. tools\eval_set.py and two train_ppo.py edits
  ({seed} placeholder for init_from, freeze_obs_norm) were written but NEVER executed or tested.
- Quality problems seen: silent job-id collisions (EXISTS) that dropped 16 of 24 evals; unverified assumptions written as facts
  ("whatever", "presumably"); a generator whose expected_outputs are wrong (defect D1 below); large new features started while the queue
  was still being built. This file fixes the process. Fewer tasks, each verified, in gated order.

==============================================================================================
SECTION P0 - OPERATING RULES (obey strictly; they override any habit)
==============================================================================================
SPEED RULES
S1  Keep your reasoning short: decide in at most 10 lines, then act. Do not re-derive decisions already written in this file.
S2  One purpose per tool call. Never chain more than 3 shell statements. Never run a command expected to print more than 30 lines: pipe to
    Select-Object -First 30 or write to a file and print the first 20 lines.
S3  Never write a file longer than 120 lines in one tool call. Write longer modules in pieces (first create, then append with Add-Content or a
    second edit). Never reprint a file you just wrote.
S4  Ignore LSP/diagnostic output completely (the editor does not use the project interpreter; its errors are false). Your ONLY correctness
    evidence is: python -m py_compile, the unit tests, and the smoke jobs of this file.
S5  Never read files in full. Use grep/Select-String with line numbers and then view at most 60 lines around the hit.
S6  If your previous turn ended with a provider timeout or an error, do NOT repeat the same large step. First run the state check
    (brief.py + RESUME_CARD tail), then split the step into at least 3 smaller steps.
S7  SESSION LIMIT: after 70 tool calls in one session, finish the current step, overwrite RESUME_CARD.md, print exactly
    "SESSION-HANDOFF - START A NEW CHAT WITH THE LAUNCHER" and stop. A fresh session is faster than a long one.

QUALITY RULES
Q1  Every statement about code behaviour that you did not observe in this session must be tagged ASSUMPTION in events.md, and an ASSUMPTION
    may not be used to build a queue or a document. Verify it with one command first, then tag it VERIFIED with the command used.
Q2  Before editing any existing project file for the first time in a session, copy it once to WORK\backup\<yyyymmdd>\<name>.bak.
Q3  Edit minimally: change the smallest region, run py_compile, run the nearest test, then the full suite once at the end of a packet.
Q4  Do not change a test so that it passes. A test may be changed only if the specified behaviour itself changed; log the reason.
Q5  A new feature is never built in the same step as a queue is generated or submitted. Features first (with tests), queues second.
Q6  LOCKED, never modify, move or delete: STUDY\PREREG.md, STUDY\PREREG.sha256, STUDY\PREREG2.md, STUDY\PREREG2.sha256, configs\frozen\*.json
    (except creating the nine files of D-section if they differ, see P2), any file under results\sealed\, and all runs\ directories of
    seeds 0-4 and 100. Sealed result files are WRITE-ONCE.
Q7  Collapsed or poor seeds are DATA. Never drop, re-run or re-seed a confirmatory training or evaluation because of its result.
Q8  Every submit must be verified by counting: the number of lines "SUBMITTED" must equal the number of jobs in the file and the number of
    "EXISTS" must be 0 for a new file. Any EXISTS or mismatch: STOP, print the ids (max 20), diagnose, do not resubmit blindly.
Q9  PowerShell redirect (> file) writes UTF-16 in Windows PowerShell 5.1. Python code that reads logs or text produced by a redirect must try
    utf-8 and then utf-16. Files you create for the project are UTF-8 without BOM.
Q10 Log every fix, finding and decision as one line in WORK\logs\events.md (date, id, one sentence). Update WORK\RESUME_CARD.md (overwrite,
    max 100 lines) after every packet and at least every 30 minutes.

GATES
A line "GATE Gx" below means: run the listed checks, print ONE compact report (max 40 lines, header "GATE-Gx REPORT"), and STOP.
Continue only after the user types the exact approval string given for that gate (the controller reviews the report first).
If a check fails twice after fixes: print "BLOCKED - AWAITING CONTROLLER" and the failing checks (max 40 lines).

==============================================================================================
SECTION P1 - BOOT AND LANE RECOVERY (do exactly this, in this order)
==============================================================================================
1. cwd must end in MiniProject_Sem4. Fixed paths (also in WORK\00_paths.txt):
   PROJECT_ROOT = C:\Projects\MiniProject_Sem4      STUDY = C:\Projects\MiniProject_Sem4\qstudy
   WORK = C:\Projects\MiniProject_Sem4_QSTUDY_WORK  PY = C:\Users\akash\anaconda3\envs\quantum_control\python.exe
   If WORK is missing: STOP "WORK NOT FOUND".
2. Read WORK\RESUME_CARD.md (tail 40 lines). Then overwrite its "Next actions" and "Controller approvals received" sections with Appendix P-A of this file.
3. Freeze integrity (one command): compute sha256 of STUDY\PREREG.md and of STUDY\PREREG2.md and compare with the first token inside
   PREREG.sha256 and PREREG2.sha256. Mismatch: STOP "FREEZE TAMPERED - AWAITING CONTROLLER".
4. Machine facts (one command each, print one line): last boot time  (Get-CimInstance Win32_OperatingSystem).LastBootUpTime ; free disk on C: ;
   Get-Process python,powershell | count. Write them to events.md. (If LastBootUpTime is later than the last heartbeat time, the machine
   restarted; this is the usual reason the lanes die. The user disables sleep and automatic restarts; you only record it.)
5. State: python WORK\tools\state.py WORK show | Select-Object -Last 15 ; python WORK\tools\brief.py WORK ; python WORK\tools\jobtools.py WORK | Select-Object -First 5
   Record the job counts per state. If any job is in state RUNNING while its lane runner is dead: first find how job_runner.py treats a RUNNING
   job at start-up (grep -n "RUNNING" WORK\tools\job_runner.py | Select-Object -First 15; view 40 lines). If it does not reset or resume the job itself,
   print the job ids and STOP with "ORPHAN RUNNING JOBS - AWAITING CONTROLLER". Do not edit job state files by hand.
6. Relaunch the six lane runners (this exact command worked last time):
   $py="C:\Users\akash\anaconda3\envs\quantum_control\python.exe"; $w="C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
   foreach($lane in @("gpu","aer","cpu1","cpu2","cpu3","cpu4")){ powershell -NoProfile -ExecutionPolicy Bypass -File "$w\tools\launch_runner.ps1" -Py "$py" -Work "$w" -Lane $lane; Start-Sleep 2 }
   Then the watchdog, detached (this exact form worked; the form with nested quotes did NOT):
   $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="powershell -NoProfile -ExecutionPolicy Bypass -File `"$w\tools\watchdog.ps1`" -Py `"$py`" -Work `"$w`" -Lanes `"gpu,aer,cpu1,cpu2,cpu3,cpu4`""; CurrentDirectory="$w"}; "PID=$($r.ProcessId) RC=$($r.ReturnValue)"
   Both must report return value 0 (the runner launcher prints "0 <pid>").
7. Wait 45 s, then print the age in seconds of each WORK\heartbeat_<lane>.txt. All six must be younger than 30 s. Otherwise read the last 15
   lines of WORK\logs\runner_<lane>.err, fix YOUR command, retry at most 3 times, then STOP.
8. Submit one trivial test job per lane (python one-liner writing WORK\results\p1_test_<lane>.txt; ids p1_test_<lane>) with a tiny JSON file and the
   Q8 count check. All six must reach DONE within 3 minutes (poll with brief.py, 3 polls of 60 s).
9. state.py set BOOT5 DONE. Continue with SECTION P2. (No stop here unless a step above failed.)

==============================================================================================
SECTION P2 - REPAIR THE Z1 QUEUE GENERATOR (code and tests only; nothing is submitted in this section)
==============================================================================================
Known defects in tools\gen_z1.py and tools\eval_set.py (controller review). Fix ALL of them, in this order, one packet each.

D1 WRONG expected_outputs (critical). In gen_z1.py every eval job lists  <tag>__nom.json  as expected output, also for the sets "ofat" and
   "grids" whose conditions are never "nom". Fix: eval_set.py must, after the loop, write a marker file
   <outdir>\<tag>__<set>.DONE.json containing {"set":..,"n_conds":..,"files":[{"path":..,"sha256":..,"cond_id":..,"episodes":..}],"rc":0}
   ONLY IF every condition produced its result file (existing valid file or new) and every file's recorded cond_id and episodes match the request.
   Otherwise exit with code 5 and no marker. gen_z1.py must use that marker as the ONLY expected output of every eval_set job (RL, baselines, Aer).
D2 TRAINING ORDER. Jobs are taken alphabetically. The old ids (z1_tr_<config>_s<seed>) run config by config, so a partial queue gives
   unbalanced contrasts. New ids: wave A = seeds 101-105, wave B = seeds 106-110, seed-major:  z1a_<seed>_<config>  and  z1b_<seed>_<config>
   (example z1a_101_rl__dither__plain). Both waves on lane gpu, no depends_on. Evaluation job ids: z1e_<set>_<config>_s<seed> (depends only on its
   own training job id). Baseline job ids z1v_..., Aer job ids z1x_.... All ids must be unique and must not exist in WORK\queue.
D3 PID GAINS UNVERIFIED. configs\pid_gains.json holds gains per probe for drift kinds "gaussian" and "ou" only, but the OFAT cells use five drift kinds.
   Verify (one small script, print max 25 lines): which gains ctrl:pid actually uses in the cells ofat_drift_kind_gaussian, _ou, _onef, _telegraph, _sinusoid for
   dither and ramsey_iq. Add the used gains ("assumed_params" style key "pid_gains_used") to every ctrl:pid result JSON. If the code silently falls back
   to the gaussian gains for the other kinds, that is acceptable ONLY IF it is what the frozen addendum says ("frozen per-probe tuned gains"); record it in
   events.md as VERIFIED and mention it in the G1 report. Do not retune.
D4 WRITE-ONCE / IDEMPOTENCE UNVERIFIED. Verify with a throwaway run (split val, outdir under WORK\smoke, 5 episodes): run evaluate.py twice for the same
   output path; the second call must not modify the file (compare sha256 and mtime) and must exit 0. For split sealed, evaluate.py must refuse to
   overwrite an existing valid file (print "SKIP sealed exists" and exit 0). If this is not the case, implement it (smallest change) and add a test.
D5 SUBMIT SAFETY. Implement tools\validate_jobs.py <json_file> that checks and prints PASS/FAIL per check (max 15 lines):
   (a) ids unique inside the file and not present in WORK\queue; (b) lane in {gpu,aer,cpu1..cpu4}; (c) gpu jobs have empty depends_on;
   (d) every depends_on id exists in the same file or in the queue; (e) the script named in cmd exists; (f) every expected_outputs path is
   producible by that cmd (for eval_set jobs it must equal the marker path derived from --outdir/--tag/--set); (g) every file named by --cfg exists;
   (h) every --method rl:<dir> path exists after its training job (or already exists); (i) cwd exists; (j) count equals the count printed by the generator.
D6 UNTESTED CODE. Test now, in this order, each as a short job or command: (1) python -m py_compile on train_ppo.py eval_set.py gen_z1.py evaluate.py;
   (2) freeze_obs_norm: train a 3000-step fine-tune from a smoke model with freeze_obs_norm true and check that the observation-normalisation statistics
   (obs_rms mean/var in the saved vec_normalize.pkl) are identical to the source model's, and that the learning rate used equals the configured one;
   (3) {seed} placeholder: a smoke fine-tune with init_from "...\seed_{seed}" and --seed 0 resolves to seed_0 (print the resolved path from the log).
D7 FILENAME CHECK. List the files of one finished exploratory training directory (runs\rl__dither__plain\seed_0, names only). The expected_outputs of the
   training jobs (best_model.zip, vec_normalize.pkl, train_summary.json) must be a subset of those names. Fix the generator if not.
D8 FROZEN CONFIGS. For each of the nine configs compare configs\frozen\<name>.json (written by gen_z1.py) with configs\<name>.json: the ONLY differing keys
   may be ppo.learning_rate (0.0003), ppo.ent_coef (0.0), ppo.gamma (0.99), total_steps (2000000). Print the diff summary (max 15 lines). Any other difference
   is a bug: fix the generator, regenerate, diff again. Also confirm that every config's "env"/"memory"/"algo" block is byte-identical to the exploratory one.

After D1-D8: regenerate with gen_z1.py (new ids), run validate_jobs.py on all four files (all PASS), then go to SECTION P3.

==============================================================================================
SECTION P3 - SMOKE TESTS, THEN GATE G1
==============================================================================================
All smoke artefacts live OUTSIDE the ingested results folder: train outputs in STUDY\runs_smoke\<config>, evaluation outputs in WORK\smoke (if evaluate.py
rejects an absolute --out, use STUDY\results_smoke). Never write smoke files under STUDY\results (the database ingests that folder).
3.1 Smoke training, 9 jobs on the gpu lane, ids p3_smoke_<config>. Use temporary copies of the nine frozen configs in WORK\smoke_cfgs with
    total_steps 20000, val_every 10000, snapshots [] (the real configs are not touched). Each must finish and produce best_model.zip,
    vec_normalize.pkl, train_summary.json. Record steps per second of the gpu lane from the logs (one number).
3.2 Smoke evaluation, 4 jobs on cpu lanes (one per kind): rl model of rl__dither__plain smoke with eval_set sets nominal (5 episodes) and ofat (5 episodes);
    ctrl:pid dither set ofat (5 episodes); ctrl:bayes:nominal ramsey_iq set grids (5 episodes). Use split val, seed_base 500000. Each must write its DONE
    marker. Check by reading the marker and counting files: n_conds must equal the number of applicable cells (print the three numbers). Time the ofat job
    (one number) and extrapolate: seconds per cell x total cells for all z1 evaluation jobs; print the projected cpu-lane hours (one line).
3.3 Smoke Aer: one eval_set job on lane aer, set aer6, engine aer, 2 episodes, ctrl:pid dither, split val. Must write its marker. Time it.
3.4 Run the full unit-test suite once as a job on cpu1 (it must stay green: 135 tests at last count).
3.5 Delete nothing. Write the smoke results into one table in events.md.
GATE G1  (print GATE-G1 REPORT with: the D1-D8 verdicts, the validate_jobs result for the four files, smoke training steps/s, smoke eval timings,
          projected gpu hours for 90 trainings at 2M steps (formula steps x 90 / steps_per_s), projected cpu hours, test-suite result, and the PID-gains finding).
          STOP. Continue only when the user types:  GO-Z1

==============================================================================================
SECTION P4 - SUBMIT Z1 (only after GO-Z1) AND WAIT
==============================================================================================
4.1 Submit in this order, each with validate_jobs first and the Q8 count after: z1_train wave A (45 jobs), z1_base (45), z1_aer baselines part, z1_eval for wave A models
    (135 jobs). Then wave B training (45) and its evaluations (135) and the Aer RL jobs. Record every count in events.md.
4.2 Do NOT submit anything for S15-S17 in this section.
4.3 WAITING (K5 of CONTINUATION3): the only poll command is  python WORK\tools\brief.py WORK && powershell -NoProfile -Command "Start-Sleep -Seconds 270"
    Print nothing else. Act only if NEW_FAILED is not "-" or STALE_LANES is not "-".
    NEW_FAILED: read the last 15 lines of that job's log (decode utf-8/utf-16). Apply X1-X4 of Part 1 (secget STAGE X). Infrastructure faults (crash, missing
    file, OOM, timeout of a runner) may be retried ONCE; an evaluation that finished is never re-run because of its result (Q7).
    STALE_LANES: repeat P1 step 6 for that lane only and for the watchdog; then P1 step 7.
4.4 Waiting is used for the coding packets of SECTION P5 (in order). While coding, run the poll command at least every 25 minutes of wall clock.
4.5 When the last wave-A training job is DONE: run db ingest, then print one table (max 20 lines): per config, number of finished seeds, mean and min of
    the final validation score from train_summary.json. Do not interpret. No stop; continue waiting/coding. (Collapsed seeds are reported, never dropped.)

==============================================================================================
SECTION P5 - LEAN CODE PACKETS (during the z1 wait; strictly in this order; one packet at a time)
==============================================================================================
Scope reduction (controller decision): ONLY the packets below are built now. Everything else of PART2 (S15 tiers B and C, S16, S17, S19, S20) waits for a
later continuation. Deferral is not deletion; any permanent drop will be written by the controller as a protocol deviation.
Stage build protocol SBP (applies to every packet): (1) read the packet text with secget, max 80 lines; (2) write the acceptance test FIRST, derived mechanically
from the statements in the spec (distributions, tolerances, equalities); (3) implement in pieces of at most 120 lines; (4) py_compile; (5) run the new test;
(6) run the full suite once; (7) one smoke job per new config (at most 2 minutes, outputs outside STUDY\results); (8) log; (9) next packet.

PACKET WP1  S15 TIER A - classical-trained versus quantum-trained transfer, the minimum that supports the headline claim.
  Read:  secget PART2 "STAGE S15" 60 (once).  Existing code to reuse (already written): qstudy\hybrid_engines.py (HybridEngine), train_ppo.py init_from.
  Build ONLY:
   (a) the engine tests of S15.1b (classical engine vs the legacy module at 20 detunings, mean within 3 percent, std within 3 percent or 0.002 absolute over 2000 draws;
       (rabi,binom) vs SurrogateEngine, KS p>0.01 at 12 detunings). If hybrid_engines.py already has a passing test, only re-run it and record the result.
   (b) configs for E1 (env LG = lorentz+gauss, probe rabi), E4 (env RB = rabi+binom, probe rabi), 5 seeds 200-204, 2M steps, frozen hyperparameters, snapshots
       [50000,200000,500000,1000000].
   (c) configs for fine-tuning from E1 into RB: budgets 50000, 200000, 500000, 1000000; arm FT-B (learning_rate = frozen/10, freeze_obs_norm true) for all four budgets;
       arm FT-A (frozen hyperparameters) for budget 200000 only (this is the Tier-A subset of Amendment B; the remaining FT-A budgets belong to Tier B). 5 seeds each.
   (d) job generator tools\gen_z2a.py with the same defect-proof design as P2 (marker-based outputs, validator, wave/seed-major ids z2a_<seed>_<name>, gpu jobs without
       depends_on, fine-tuning jobs on cpu1-cpu3 depending on their own E1 training job, evaluation jobs with sealed base 920000 on four environments LG, LB, RG, RB
       at nominal, 1000 episodes, per model).
  Tier B (NOT now): E2, E3, E5, E6, remaining FT-A budgets, Aer evaluations of S15, correlation analyses. Tier C (NOT now): SAC/TD3 arms.
  End of packet: validate_jobs PASS on the z2a files, smoke job per config, then
  GATE G2  (print GATE-G2 REPORT: tests, validator, smoke, projected gpu hours added). STOP. Continue only when the user types:  GO-Z2A
  After GO-Z2A: submit the z2a files with the P4.1 discipline. They sort after z1 on the gpu lane by id (z1a < z1b < z2a).

PACKET WP2  quality tooling for the paper (cheap, cpu only, no new science).
   (a) tools\training_health.py: per config over finished seeds: final validation score, best validation score, steps to reach 90 percent of its best, and a flag
       "COLLAPSE_CANDIDATE" when final validation score < 0.6 x median of its config. Report only; Q7 applies.
   (b) tools\paired_stats.py: for the eight contrasts C-1..C-8 of the freeze addendum (Appendix K item 3): paired difference per seed index of lock_w1000 at the
       nominal condition, mean, bootstrap 95 percent CI (10000 resamples, fixed seed), Wilcoxon signed-rank p, effect size (Cohen dz). Test on synthetic data with
       a known shift (the CI must contain the shift, p small) and on a null case (CI contains 0).
   (c) tools\complexity_note.py is NOT built now.
   End of packet: GATE G3 (tests only, max 15 lines). STOP. Continue only when the user types:  GO-P6

==============================================================================================
SECTION P6 - RESERVED
==============================================================================================
Reserved for the controller: results review of z1, the decision about the exploratory RL-stabilisation study (labelled exploratory, validation split only,
never sealed), and the next continuation for S15 tier B/C, S16, S17. Do not start anything in this section without a written instruction.

==============================================================================================
APPENDIX P-A - RESUME CARD PATCH (overwrite these two sections of WORK\RESUME_CARD.md at boot)
==============================================================================================
## Next actions (in order)
1. CONTINUATION5 P1 boot (lanes, watchdog, freeze integrity) -> P2 repair gen_z1/eval_set (D1-D8) -> P3 smoke -> GATE G1 (wait for GO-Z1).
2. P4 submit z1 in waves (A then B), wait with brief.py; P5 packets WP1 (GATE G2, wait GO-Z2A) and WP2 (GATE G3, wait GO-P6) while the gpu lane trains.
3. Nothing of S15 tier B/C, S16, S17, S19, S20 is built or queued in this continuation.
## Controller approvals received
- Freeze written and hashed (M4). z1 queue NOT yet submitted. Approval strings still required: GO-Z1, GO-Z2A, GO-P6. No other approval exists.

==============================================================================================
APPENDIX P-B - LAUNCHER MESSAGES (for the user; keep them this short)
==============================================================================================
START:   Read C:\Projects\QSTUDY_PROMPT_CONTINUATION5.md and execute it exactly. THIS_FILE5 = that file. Start at SECTION P1.
RESUME:  Read WORK\RESUME_CARD.md (tail 40 lines) and C:\Projects\QSTUDY_PROMPT_CONTINUATION5.md SECTION P0 only. Continue from the first unfinished step of the card.