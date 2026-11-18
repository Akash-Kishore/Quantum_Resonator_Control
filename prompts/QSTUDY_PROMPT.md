QSTUDY MASTER PROMPT  (executor: GLM 5.3)       THIS_FILE = the path given in the launcher message
======================================================================================================
SECTION 0 - ROLE AND MISSION
======================================================================================================
You are an EXECUTION ENGINE. I (the controller) have already done the thinking: designs, formulas,
thresholds, tests, file names and order are fixed below. Follow them literally. Do not redesign, do not
"improve", do not skip. If something is ambiguous or fails, apply Sections 1-3. If no rule applies, mark
the step BLOCKED(reason + evidence) and continue with other work. Never stop and never ask me questions.

PROJECT: a simulated-quantum frequency-tracking study (RL vs PID vs Bayesian filtering) that must become a
rigorous, publishable case study (journal: IEEE Access). The existing project (in your working directory)
has a classical resonator RL pipeline and a weak legacy Qiskit pipeline. You will:
  PART A  Understand the whole existing project (Stage 1) and PROVE it with a comprehension gate.
  PART B  Build, test, run and document SEVEN improvement approaches in a NEW folder qstudy\ :
          1 Two-point (dither) probing            5 Fast analytic surrogate engine + Qiskit Aer engine
          2 Ramsey with two quadrature readouts   6 Temporal memory (frame stacking, windowed GRU)
          3 Bayesian-filter baseline + hybrid RL  7 Realistic noise: T2, readout error, delay, DAC/ADC bits
          4 Train directly in the quantum env with domain randomization + curriculum
          EXECUTION ORDER (dependencies): foundation, 5, 1, 2, 7, 3, 6, 4, then tuning/combos, then freeze,
          confirmatory runs, sealed evaluation, stress tests, analysis, database, figures, paper package.
  PART C  Produce a database and documentation so that the paper plus its data package contain all
          information about PID, drift (Gaussian and others), RL and quantum behaviour.
INTEGRITY RULES (non-negotiable; they are what makes the work publishable):
 I1 Metrics, thresholds, seeds, contrasts and the experiment matrix are PRE-REGISTERED in Stage 2 and
    frozen (hash) before any sealed evaluation. You may add analyses, but they are labelled EXPLORATORY.
 I2 Sealed evaluation seeds are used exactly once, after the freeze. evaluate.py refuses sealed runs
    otherwise. Hyperparameters and model selection use tuning/validation seeds only.
 I3 Report every pre-registered result whether good or bad. Never drop seeds, runs, or conditions. Never
    re-run a sealed evaluation to get a better number. If a result is bad, document it truthfully.
 I4 Every number in any document must come from a result file or the database (generated tables), never
    typed from memory. Never invent a citation or reference; write [CITATION NEEDED] instead.
 I5 State the ceiling: with one-step drift sigma=500 Hz, a perfect oracle has time-in-lock(1000 Hz)=0.954.
    No method may be reported as exceeding the oracle ceiling at nominal conditions (if one does, it is a bug:
    flag it as an ANOMALY and investigate).

======================================================================================================
SECTION 1 - ABSOLUTE RULES
======================================================================================================
R1  PROJECT_ROOT = your working directory when you start. All new work goes in PROJECT_ROOT\qstudy\ and in
    WORK (below). NEVER modify, rename, move or delete any pre-existing file or folder anywhere. The final
    manifest diff must show: ADDED only under qstudy\, CHANGED = none, REMOVED = none.
R2  Git on the original: read-only commands only (status, log, show, diff, ls-files, rev-parse). Never add,
    commit, checkout, restore, reset, clean, stash, pull, fetch, push, merge, rebase, gc, fsck.
R3  No network. No pip/conda install, no package changes. If an optional package is missing, use the stated
    fallback and document it.
R4  Existing scripts are NEVER run unless this prompt says so (they overwrite checkpoints). Do not run any
    script starting with run_ in the project root. Existing code may be IMPORTED/READ by legacy_diag.py only.
R5  NEVER invent. Unknown = UNKNOWN. Missing = MISSING. Every claim in notes/docs cites path:line or a result
    file. No "probably", "likely", "seems".
R6  Never claim a step DONE without evidence you saw (output, file, test pass). Evidence goes in WORK\evidence\.
R7  Log every command (with timestamp) to WORK\logs\commands.log. Log every incident to WORK\logs\events.md.
R8  Keep each tool call and each of your replies small. Print at most 80 lines from any command; write long
    output to a file and print the first 30 lines. Write helper scripts in pieces of at most ~120 lines (the
    Appendix files and big modules like envs.py may be written in several appends).
R9  Never use Python bytecode caching (PYTHONDONTWRITEBYTECODE=1 everywhere). Never write outside qstudy\,
    WORK, and the system temp folder.
R10 After each stage print: STAGE <id> DONE | ok=<n> | blocked=<n> | notes=<short>.

======================================================================================================
SECTION 2 - TIMEOUT-PROOF EXECUTION  (read twice; earlier agents died here)
======================================================================================================
Your tool calls are killed after a few minutes. Therefore:
T1  Any command YOU run directly must finish in under 60 seconds. Everything longer (hashing, copying, tests,
    training, evaluation, Aer, reading hundreds of files in bulk, building the database) runs as a JOB.
T2  Jobs run in detached runner processes started through WMI (they survive your tool call being killed).
    Lanes: gpu (training), aer (Qiskit Aer work), cpu1..cpuK (everything else), K = max(1, min(4, CORES//4)).
    A watchdog (also detached) relaunches any runner whose heartbeat is older than 240 s.
T3  A job is DONE only if its command exited 0 AND all its expected_outputs exist. Jobs have no fixed
    time limit; they are killed only on stall (no new output in the log or the job's watch files for
    stall_minutes) or max_wall_hours. Stalls and transient errors (CUDA OOM, PermissionError, WinError 32,
    Access denied, MemoryError) are retried automatically; code errors are not.
T4  Every long script MUST print a progress line at least every 60 s (flush) and write results incrementally,
    and MUST be resumable (skip outputs that already exist). Use the job field "watch" for files that grow.
T5  Polling: run  python WORK\tools\jobtools.py WORK  (short), check heartbeat ages, then
    powershell -NoProfile -Command "Start-Sleep -Seconds 90"  and repeat. Never sleep longer than 120 s in one
    command. While jobs run, do the next independent writing/reading work instead of idling. Print one status
    line per poll; every 10th poll print the full job table (jobtools.py WORK -v | head -60).
T6  If a heartbeat file is older than 240 s while jobs are pending/running, relaunch the lane runner with
    launch_runner.ps1 and the watchdog with watchdog.ps1 (a second runner exits by itself: single instance).
T7  If YOUR OWN tool call times out or errors: do NOT repeat it. Check whether the work is still running
    (tasklist, job table), read the log, write the event to events.md, resubmit as a job if it was long.
T8  RESUME PROTOCOL. State is kept with:  python WORK\tools\state.py WORK set <STEP_ID> <STATE> <note>
    (STATE: RUNNING, DONE, DONE_WITH_NOTES, BLOCKED, SKIPPED). At the start of EVERY session or whenever unsure:
    (1) re-read THIS_FILE, (2) python WORK\tools\state.py WORK show, (3) python WORK\tools\jobtools.py WORK,
    (4) continue at the first unfinished step. Never redo a DONE step. If WORK already exists you are RESUMING.
T9  Job submission: write a JSON list of job dicts to a file and run
    python WORK\tools\jobtools.py WORK submit <file.json>
    Job dict keys: id, lane, cwd, cmd (ONE string run by cmd.exe), depends_on[], expected_outputs[], env{},
    max_retries (default 2), stall_minutes (default 60), max_wall_hours (default 24), watch[].
    Job ids are unique and descriptive. Resubmitting an existing id does nothing ("EXISTS").
T10 Training env counts: NENVS = max(2, min(8, CORES//2)). Evaluation/cpu jobs set OMP_NUM_THREADS=1,
    MKL_NUM_THREADS=1. Aer jobs set max_parallel_threads = max(1, CORES//4).

======================================================================================================
SECTION 3 - EXCEPTION PLAYBOOK (apply to ANY failure)
======================================================================================================
X1  Transient infrastructure error: the runner retries; if it still fails record FAILED with the log tail and
    continue with independent work.
X2  Python exception in code YOU wrote: read the traceback, fix the bug in your own code, re-run the unit test
    that covers it (max 3 fix cycles per bug). After 3 cycles mark BLOCKED with the exact error and continue.
X3  Python exception in PRE-EXISTING project code: never edit it. Record the cause with file:line, mark the
    dependent step BLOCKED(reason) and continue.
X4  A test fails: do not weaken the test or its threshold. Fix the code. If the threshold itself seems wrong,
    record both numbers in events.md and mark BLOCKED(test) for the controller; continue with other work.
X5  Disk free < 15 GB: stop launching new jobs, finish running ones, write DISK_LOW, shrink outputs (per-step
    traces for the first 50 episodes only), continue.
X6  GPU unavailable or CUDA errors persist: training runs on CPU (device="cpu") and are slower; never stop.
X7  A result looks impossible (NaN, zero episodes, identical arrays across different methods, lock above the
    oracle ceiling): record an ANOMALY row in the database and in events.md, do not hide it, investigate.
X8  Two sources disagree: report both with citations; never pick silently.
X9  Unsure whether an action could modify a pre-existing file: do not do it; mark BLOCKED(needs controller).
X10 Context getting long: write state (T8), keep going; after a restart obey T8.
The only reasons to stop before Stage 14 are the explicit STOP lines in Stage 0.

======================================================================================================
STAGE 0 - BOOTSTRAP
======================================================================================================
0.1 Print your working directory = PROJECT_ROOT and list its top level (including hidden). If it contains
    no .py file within 4 levels: STOP "WRONG FOLDER - AWAITING INSTRUCTIONS".
0.2 Derive (never type): PARENT = parent folder of PROJECT_ROOT; NAME = its folder name;
    WORK = PARENT\<NAME>_QSTUDY_WORK ; STUDY = PROJECT_ROOT\qstudy.
    If WORK exists you are RESUMING (obey T8 and jump to the first unfinished step). Otherwise create WORK with
    subfolders: tools, logs, queue, results, evidence, understanding, jobs_json. Also look (do not assume) for
    sibling folders named <NAME>_FIXRUN, <NAME>_AUDIT3, <NAME>_BACKUP_* : if present record them as FIXRUN, AUDIT3,
    BACKUP (leads/handover only). Create STUDY only if it does not exist.
0.3 Interpreter: read the line "PROJECT_PY" or the python path from FIXRUN\00_environment.md if it exists; else
    find the conda env named in any environment*.yml under PROJECT_ROOT (conda env list, 30 s timeout).
    Verify: "<PY>" -c "import stable_baselines3,gymnasium,numpy,scipy,torch,qiskit,qiskit_aer,matplotlib;print('OK')".
    If it fails: STOP "INTERPRETER NOT FOUND - AWAITING INSTRUCTIONS" with the error.
0.4 Environment record WORK\evidence\00_environment.md: date, OS, CORES (os.cpu_count()), RAM, free disk (must be
    >= 40 GB else STOP "DISK TOO SMALL"), python/numpy/scipy/torch/SB3/gymnasium/qiskit/qiskit_aer/matplotlib
    versions, CUDA availability and GPU name, and presence (yes/no, version) of: sb3_contrib, pandas, pyarrow,
    duckdb, pylatexenc, pypdf, seaborn. Also run powercfg /query SCHEME_CURRENT SUB_SLEEP (read-only) and report
    whether sleep/hibernate timeouts are enabled (report only; do not change).
0.5 Write WORK\00_paths.txt as NAME=VALUE lines: PROJECT_ROOT, PARENT, NAME, WORK, STUDY, PROJECT_PY, THIS_FILE,
    CORES, NENVS, K_CPU. Every later script reads this file; never hardcode a path.
0.6 Write the Appendix A files into WORK\tools\ VERBATIM. Run python -m py_compile on the .py files.
0.7 Launch runners (one command each; ReturnValue must be 0):
    powershell -NoProfile -ExecutionPolicy Bypass -File "<WORK>\tools\launch_runner.ps1" -Py "<PY>" -Work "<WORK>" -Lane gpu
    ... same for -Lane aer, cpu1, cpu2, ... cpuK.
    Launch the watchdog detached:
    powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -File \"<WORK>\tools\watchdog.ps1\" -Py \"<PY>\" -Work \"<WORK>\" -Lanes \"gpu,aer,cpu1,...,cpuK\"'; CurrentDirectory='<WORK>'}"
    Wait 40 s; confirm every heartbeat_<lane>.txt exists and is fresh. Submit a trivial test job per lane (a Python
    one-liner writing a file into WORK\results) and confirm all reach DONE within 3 minutes. Else read
    WORK\logs\runner_<lane>.err, fix YOUR tool usage, retry (max 3).
0.8 Safety manifest (job on cpu1): python WORK\tools\manifest.py build "<PROJECT_ROOT>" "<WORK>\evidence\manifest_BEFORE.csv"
    and save "git status --short" (30 s timeout) to WORK\evidence\git_status_BEFORE.txt.
0.9 state.py set S0 DONE.

======================================================================================================
STAGE 1 - UNDERSTAND THE EXISTING PROJECT (nothing in Part B starts before the gate in 1.8 passes)
======================================================================================================
Notes go to WORK\understanding\ . FIXRUN/AUDIT3 reports are LEADS: re-verify every fact you use.
1.1 INVENTORY (job): list every file under PROJECT_ROOT (skip .git, __pycache__, env folders) with: relative path,
    size, mtime, extension, category (code, doc, data, checkpoint, figure, circuit, firmware, config, other).
    Write understanding\inventory.csv and docs\00_project_map.md (2-level tree + one line per folder, from contents).
1.2 CODE: read EVERY .py file COMPLETELY (chunks of at most 200 lines using the file-read tool). For each write
    understanding\code\<relative path with \ replaced by __>.md with exactly these fields, all non-empty:
      PURPOSE (2-4 sentences) | PUBLIC API (classes/functions with signatures) | INPUTS (files, args, env vars) |
      OUTPUTS (files written) | KEY CONSTANTS (name=value, with line) | ALGORITHM (numbered steps) |
      DEPENDENCIES (project modules + third-party) | RISKS (hardcoded paths, overwrite hazards, bugs seen) |
      LINE EVIDENCE (at least 3 entries "path:line - what it shows").
    Then run the mechanical gate (write understanding\check_code_notes.py, run as a job): every .py has a notes
    file; all fields present; at least 3 LINE EVIDENCE entries whose cited line numbers exist. Fix and re-run until it prints
    CODE NOTES COMPLETE.
1.3 DOCUMENTS: extract and read ALL .docx (unzip, read word/document.xml text), .md, .txt, and .pdf (pypdf if installed, else
    mark PDF_NOT_READ with name). For each write understanding\docs\<name>.md: SUMMARY (per section one line),
    CLAIMS (every sentence with a number, a result, a date or a status like done/verified/locked, copied verbatim <=200
    chars with line or paragraph index), label them UNVERIFIED, and CONTRADICTIONS (where two documents disagree).
1.4 IMAGES: for EVERY .png/.jpg/.svg: if your tools can view images, view it and write understanding\images\<name>.md:
    WHAT IT SHOWS (axes, labels, legend, visible numbers), PRODUCER (script that writes this filename, or NO_PRODUCER),
    ANOMALIES. If you cannot view images, write NOT_VIEWABLE plus the metadata (dimensions via PIL) and the producer.
    Never describe an image you did not view.
1.5 DATA AND CHECKPOINTS: for every .json/.csv/.npz and for every trained_models folder: structure (keys, columns, shapes),
    first 3 records, and which script produced it. Re-verify the FIXRUN env_map if it exists.
1.6 CIRCUITS AND FIRMWARE: read every LTspice/.asc/.cir/.ino/.cpp file; record components, values, directives, pin maps.
1.7 SYNTHESIS: write docs\01_project_understanding.md with sections: Purpose and history; Architecture (legacy
    environments, simulators, trainers, evaluators); Legacy observation/reward/action definitions (with path:line);
    Legacy Qiskit measurement model (exact formula and parameters); What FIXRUN verified (numbers + source);
    Known defects and quarantines; Glossary.
1.8 COMPREHENSION GATE. Write docs\02_comprehension_answers.md answering ALL of the following; each answer is one block
    "A<n>: <text> [path:line | keyword]" with at least one citation whose keyword is literally present within +-2 lines of the cited
    line. Then run (as a job) your script understanding\cite_check.py that verifies every citation mechanically. The gate passes only
    when it prints ALL CITATIONS VERIFIED. Questions:
  Q1 ENV_MODULE_MAP and EXPECTED_OBS_LEN entries in train.py.      Q2 Model output-folder pattern (including the ablation suffix rule).
  Q3 PPO hyperparameters, number of parallel envs, device and total timesteps in train.py.   Q4 How VecNormalize is made, saved and loaded.
  Q5 Legacy V4 env: action, observation vector (order and formulas), reward, termination, episode length.  Q6 Legacy V3 env: same, plus differences from V4.
  Q7 The two resonator_model modules: drift equations and parameters, amplitude/measurement model.   Q8 evaluate_agent.py: exact MAE, near-resonance and
  low-amplitude definitions, episodes, seeding.  Q9 Legacy Qiskit env v2: omega_rabi, action_scale, shots, measurement formula, obs, reward.
  Q10 Legacy Qiskit env v3: same, plus differences from v2.   Q11 phase1_baseline.py: what it evaluates, episodes, output file.
  Q12 phase2_finetune_v5.py: init checkpoint, steps, learning rate, env, output folders.   Q13 phase3_evaluate_v5.py: what it loads and writes.
  Q14 pid_baseline.py: grid, episodes, what the PID acts on.   Q15 verification_tools\eval_checkpoints.py (if present): regime-injection method, metrics.
  Q16 Count of trained_models folders per family/seeds and the evidence for each family's training drift.   Q17 Qiskit-trained checkpoint folders and what each is.
  Q18 FIXRUN numbers (V4 5-seed Gaussian/OU, PID both regimes, Qiskit zero-shot x3, V5 fine-tuned, shot sweep) with source files.
  Q19 Contents of _quarantine_corrupted and why.   Q20 Three documents (path:line) that cite the unreproducible 4370 Hz / 22.1% baseline.
  Q21 LTspice/firmware facts: resonator f0, Q, component values, pin map.   Q22 Which scripts overwrite checkpoints and must never be run.
  Q23 Names of the legacy env attributes that expose true frequency, control frequency and amplitude (needed for diagnostics).
  Q24 Every PNG in the project with its producer script.
1.9 state.py set S1 DONE only after the gate passes. Print the 12-line summary of docs\01_project_understanding.md.

======================================================================================================
STAGE 2 - PRE-REGISTRATION, SKELETON, PHYSICS TESTS
======================================================================================================
2.1 Create STUDY subfolders: tests, configs, configs\frozen, runs, results, results\eval, results\val, results\tables, db, docs,
    docs\paper, figures, tools, legacy. Write STUDY\README.md (what this folder is, how to run tests, folder map).
2.2 Write STUDY\qphysics.py = Appendix B file 1 VERBATIM. Write STUDY\tests\test_aer_physics.py = Appendix B file 3 VERBATIM and
    STUDY\tests\ref_tracker.py = Appendix B file 2 VERBATIM (copy exactly). Write STUDY\tests\test_physics.py (unittest) with these tests:
    (a) rabi_p(0)==1 and symmetric; (b) ramsey_p I/Q identities and sign: at delta=+1000 Hz the Q quadrature is >0.5 and I quadrature <1;
    (c) 2000 simulated Ramsey I/Q draws at delta=1500 Hz, 512 shots each, T2=200us, e=0.02: mean of delta_hat=phase/(2*pi*tau) within 60 Hz of 1500, std within
    +-30% of 1/sqrt(fisher_ramsey_iq) (use readout-corrected phase: divide (2p-1) by (1-2e) first); (d) Drift: ou std over 5000 steps within 10% of 1601 Hz,
    gaussian lag-1 autocorrelation > 0.99, onef lag-1 between 0.80 and 0.92, all kinds respect |x| <= 25000; (e) GridBayes random walk: predict 50 steps from std 300
    gives std within 3% of sqrt(300^2+50*500^2); (f) quantize(x,8) has step 195.3 Hz.
    Run: python -m unittest discover STUDY\tests -v  as a job (cpu1). Run test_aer_physics.py as a job (aer lane; it must print AER VALIDATION PASS).
    Run STUDY\tests\ref_tracker.py as a job (cpu2) and store the output in evidence\ref_tracker_output.txt; it must reproduce Appendix B file 2's REF lines.
2.3 Write STUDY\PREREG.md = Appendix D text VERBATIM (it is the pre-registration). Do NOT create PREREG.sha256 yet (that happens in Stage 11).
2.4 state.py set S2 DONE when all tests pass (X4 applies on failure).

======================================================================================================
STAGE 3 - FOUNDATION (shared code; extended by later stages; every test from earlier stages must keep passing)
======================================================================================================
Write these modules in STUDY\. Interfaces are fixed; extend, never rename.
3.1 envs.py - class BatchEnv(cfg, B, seed_base, engine=None). ONE physics implementation used for training (B=1 via the Gym wrapper),
    evaluation (B up to 1000) and everything else.
  cfg defaults (nominal): probe="rabi" (also "dither","ramsey_iq","rabi_legacy" later), engine="surrogate", drift="gaussian",
    drift_sigma=500.0, shots=1024 (TOTAL per control step; dither and ramsey split it in two halves), omega=5000.0, dither_delta=2500.0,
    tau=50e-6, t2=200e-6, e01=0.02, e10=0.02, action_scale=3000.0, action_delay=0, dac_bits=12, obs_bits=None, horizon=200, burn_in=20,
    init_offset=5000.0, reward_width=4000.0, action_penalty=0.01, filter_features=False, filter_sigma=None, filter_theta=None,
    filter_window=5000.0, stack=1, dr=None.
  State per episode b: x (true offset of the frequency from F_NOM, Hz), fc (control offset), rngs: drift RNG = default_rng([seed_base, ep, 1]),
    measurement RNG = default_rng([seed_base, ep, 2]) (so every method sees identical drift realisations: paired design).
  reset(): x0 = U(-init_offset, +init_offset) from the drift RNG; fc=0; one measurement at fc (no drift step); return obs.
  step(action[B]) order: (1) a=clip(action,-1,1); action delay d (per episode) via a history buffer; (2) fc = clip(quantize(fc + a_eff*action_scale, dac_bits), -CLIP, CLIP);
    (3) x = drift.step(); (4) measure at detuning x - fc using the probe's rule; (5) update filter features if enabled; (6) e = x - fc (ERROR AT PROBE TIME);
    reward = 1 - min(1, |e|/reward_width) - action_penalty*|a|; (7) no early termination; done at t == horizon.
  Observation (float32, all entries in about [-1,1]):  base = [fc/25000, prev_action]; probe features: rabi -> [2*phat-1, 2*(phat - phat_prev)];
    dither -> [2*phat_plus-1, 2*phat_minus-1, 2*(phat_plus - phat_minus)]; ramsey_iq -> [2*pI-1, 2*pQ-1, phase/pi] with phase=ramsey_phase(pI,pQ);
    filter features (if enabled): [(mu-fc)/5000, sigma_f/5000]. phat = counts/shots_used. If obs_bits is not None: phat is rounded to 2^obs_bits levels.
    With stack=K>1 the observation is the concatenation of the last K frames [oldest ... newest] (frames before reset are copies of the first frame).
  info (arrays): e, fc, x (x is for METRICS ONLY and must never enter obs), counts, shots_used. public_state() returns what a controller may see
    (counts, shots, fc, probe) and NEVER x.
  Metric helper metrics(e_traces[T,B]) -> per-episode arrays: lock_w500/1000/2000/3333/5000 = mean over t>=burn_in of (|e|<=W); rms; med_abs; p95_abs; max_abs;
    lock_lost = 1 if some run of >=20 consecutive steps has |e|>10000 after burn-in; mean_reward.
  Gym wrapper QuantumCalibGym(gymnasium.Env): Box obs float32 (-10,10), Box action (-1,1,(1,)), wraps BatchEnv with B=1, seeds from reset(seed).
3.2 engines.py - class SurrogateEngine: counts(kind, delta[B], shots[B], t2[B], e01[B], e10[B], rngs, omega, tau) with kind in {"rabi","ramsey_I","ramsey_Q",
    "rabi_legacy"}; probabilities from qphysics (readout error applied last), counts via per-episode RNG binomial. (AerEngine is Stage 4.)
3.3 controllers.py - classes with act(obs_or_public_state) -> action[B]: NoControl, Oracle (uses x_prev: fc <- previous true x; ceiling reference; needs BatchEnv internal access, used
    only for reference rows and flagged ORACLE in the database), HillClimb(step_hz, thr) for the single rabi probe (perturb-and-observe: keep direction; reverse when
    phat drops by more than thr; action = dir*step_hz/action_scale). (PID, Bayes are added in Stages 6, 8.)
3.4 evaluate.py - CLI: --method {ctrl:noctrl | ctrl:oracle | ctrl:hill | ctrl:pid | ctrl:bayes[:nominal|:oracle] | rl:<model_dir>} --cfg <json overrides>
    --cond <cond_id or json> --split {tune,val,sealed} --episodes N --seed_base S --out <json> [--traces N]. Seed bases: tune 100000, val 500000, sealed 900000 (the
    caller passes the exact base; sealed refuses to run unless STUDY\PREREG.sha256 exists and equals sha256(PREREG.md); exit code 3 otherwise). Output JSON keys: status, method,
    model_dir, cfg, cond_id, split, episodes, seed_base, lock_w500..w5000 (mean, sd over episodes), rms, med_abs, p95_abs, lock_lost_rate, mean_reward, git_head, versions,
    started, finished, cmd, npz_path, npz_sha256. The NPZ stores per-episode arrays (all metrics) and, for the first --traces episodes, the traces e, fc, x. RL inference is batched
    over episodes: build observations for all B episodes, normalise with the saved VecNormalize statistics, model.predict(obs, deterministic=True).
    Progress line every 60 s. Skip work whose output exists.
3.5 stress.py - defines the pre-registered conditions (Appendix D lists them) as a dict cond_id -> cfg overrides: nominal "nom", OFAT cells "ofat_<factor>_<value>", grids
    "g2_shotsT2_<i>_<j>" and "g2_sigmaRO_<i>_<j>". Function list_conds(set_name) for sets: "val" (nom, ofat_drift_sigma_1000, ofat_shots_128, ofat_t2_50), "ofat", "grids".
3.6 stats.py, db.py, figures.py, docs_check.py: created as stubs now (full specs in Stages 5 and 13); db.py already creates the SQLite schema of Appendix D and has
    "ingest <results_dir>" that is idempotent (skips rows whose eval_id exists).
3.7 Tests (STUDY\tests\test_env.py, unittest): (a) determinism: two BatchEnv runs with the same seed_base and actions give identical e traces; (b) paired design: changing the probe
    does not change x traces; (c) no leakage: obs never changes when only x_true noise-free parts change except via counts (check by reading obs construction code path: unit test
    asserts info["x"] is not used in obs by perturbing x after the measurement and confirming obs is unchanged); (d) NoControl lock_w1000 at nominal with drift gaussian < 0.35;
    (e) Oracle lock_w1000 within 0.954 +- 0.02 (1000 episodes) and lock_w2000 > 0.995; (f) HillClimb on rabi nominal (sweep step_hz in {500,1000,1500,2000}, thr in {0,0.02,0.05}, tune seeds)
    achieves best lock_w1000 > NoControl; (g) metrics() unit-checked on a synthetic trace; (h) Gym wrapper passes gymnasium.utils.env_checker.check_env.
    Run the whole suite as a job; all must pass.
3.8 state.py set S3 DONE.

======================================================================================================
STAGE 4 - APPROACH 5: ANALYTIC SURROGATE ENGINE + QISKIT AER ENGINE
======================================================================================================
4.1 Add to engines.py: class AerEngine(noise config) with the SAME counts() interface. Rules: one AerSimulator, circuits built exactly as validated in Appendix B file 3
    (Rabi: UnitaryGate(rabi_unitary(delta)); Ramsey I: rx(pi/2), id, rz(phi), rx(pi/2); Ramsey Q: rx(pi/2), id, rz(phi), ry(pi/2); phi = 2*pi*delta*tau). Noise model (built once per
    cfg): thermal_relaxation_error(T1=100*T2, T2, tau) attached to 'id' on qubit 0; ReadoutError([[1-e01,e01],[e10,1-e10]]). Batch execution: collect the B circuits of a step, run one
    backend.run(list_of_circuits, shots=shots, seed_simulator=<drawn from the episode RNG>) and read counts. Noise parameters must be uniform across the batch (domain randomization is
    surrogate-only); shots uniform across the batch. Aer rabi pulses are treated noise-free apart from readout (documented limitation).
4.2 STUDY\validate_engines.py (run as jobs; results to results\validation\ and database table validation):
    V1 expected-probability check: for kinds {rabi, ramsey_I, ramsey_Q}, 13 detunings from -12000 to 12000 Hz step 2000, noise sets {none; T2=200us,e=0.02/0.02; T2=50us,e01=0.05,e10=0.10}:
       Aer with 200000 shots vs surrogate probability: pass if |p_aer - p_sur| <= 4*sqrt(p(1-p)/200000)+0.002 for 100% of cells.
    V2 distribution check: 12 selected cells at shots 64 and 1024, 300 repetitions each: two-sample KS test between Aer counts and surrogate counts; report all p-values; pass if
       at least 90% of cells have p > 0.01.
    V3 closed-loop check: Bayes tracker not available yet at this stage, so use HillClimb on probe rabi and a scripted open-loop schedule; compare lock_w1000 on 100 episodes
       per engine with the same seed_base: |difference| <= 0.04. (Re-run V3 with the Bayes tracker in Stage 8 and record both.)
    V4 speed: steps/s of both engines (report only).
4.3 Experiments: none beyond validation. 4.4 Run the unit-test suite again (all earlier tests must still pass).
4.5 DOC: write docs\approach_5_engines.md using the template in Appendix C. Results tables come from the database (db.py ingest, then stats.py tables). Run docs_check.py; fix until PASS.
4.6 state.py set S4 DONE.

======================================================================================================
STAGE 5 - STEP 0: DIAGNOSE THE LEGACY QUANTUM RESULT (no new training; informs the paper)
======================================================================================================
5.1 Using Q9, Q10, Q12, Q13, Q23 from Stage 1: write STUDY\legacy_diag.py. It adds PROJECT_ROOT to sys.path, imports the legacy Qiskit env v3 and loads the fine-tuned policy
    checkpoint (folder v4_qft_v5 under data_logs\qiskit_aer_simulation_2, with its VecNormalize file) exactly as phase3_evaluate_v5.py does, then runs 200 deterministic episodes
    (seed base 777000 + episode index) recording per step: amplitude/probability, true frequency error and control frequency (attribute names from Q23, cite them). Output
    results\legacy\legacy_diag.npz + json: per-episode MAE, near fraction, lock-loss flag (>=20 consecutive steps with |error| > 10000 Hz), fraction of episodes with MAE > 5000 Hz.
    It must write NOTHING into pre-existing folders. Run as a job on the aer lane (stall_minutes 120, watch the npz/progress file). If the legacy env or checkpoint cannot be loaded
    (X3), record BLOCKED(reason) and still do 5.2.
5.2 Lineshape replica test: implement rabi_legacy(delta) = exactly the legacy env's measurement probability (cite the code lines). Compare against the legacy env's own measurement at 20
    detunings with 100000 shots (or its analytic expression if the env has one): max abs difference <= 0.01 else BLOCKED(mismatch with numbers). Then compute with qphysics: the
    Fisher-information floor of the legacy single probe vs the exact-Rabi single probe vs two-point dither vs Ramsey I/Q (at shots 1024, e=0.02) as a function of detuning.
5.3 Analysis outputs: histogram of per-episode MAE, survival curve of lock-loss, fraction of lost episodes, representative traces; write docs\step0_legacy_diagnosis.md (Appendix C template,
    adapted) stating plainly what limits the legacy result (use only what the data show).
5.4 state.py set S5 DONE (DONE_WITH_NOTES if 5.1 was blocked).

======================================================================================================
STAGE 6 - APPROACH 1: TWO-POINT (DITHER) PROBING
======================================================================================================
6.1 Add probe="dither" to BatchEnv: two Rabi probes at detunings (x - fc - dither_delta) and (x - fc + dither_delta), shots//2 each; obs as in Stage 3.1. Sign rule (tested): if the true
    frequency lies above fc (e>0) then p_plus > p_minus on average.
6.2 Add to controllers.py: PIDController(probe): the error signal is the observation entry 2*(p_plus - p_minus) (dither), the entry phase/pi (ramsey_iq, Stage 7); a = Kp*err + Ki*sum(err) +
    Kd*(err - err_prev), anti-windup clip of the integral to +-1; tuned on TUNE seeds (seed base 100000, 200 episodes) by exhaustive grid Kp in {0.3,0.5,0.7,0.9,1.0,1.5}, Ki in {0,0.02,0.05},
    Kd in {0,0.1}; the winner per (probe, drift kind) is saved to configs\pid_gains.json with its tune score. Also add the Bayes tracker later (Stage 8).
6.3 Tests (test_dither.py): sign rule with 20000 simulated draws (probability that sign(p_plus-p_minus)==sign(e) > 0.5 for |e|=1000 and > 0.9 for |e|=2500 at 1024 shots);
    dither_delta sweep {1000,1500,2000,2500,3500,5000} for the PID controller on TUNE seeds: record the table (the default dither_delta stays 2500 unless the sweep says otherwise by more
    than 0.03 lock_w1000; if it does, record the finding in events.md and keep 2500 for the pre-registered runs anyway: pre-registration wins); PID dither lock_w1000 at nominal > HillClimb on rabi.
6.5 Experiments (all on val seeds; jobs):  (i) controllers: NoControl, HillClimb(rabi), PID(dither) at nominal and on the "val" set; (ii) RL training: configs rl__rabi__plain and
    rl__dither__plain, seeds 0-4 each (Stage 6.6 specifies training), (iii) evaluate each trained model on val seeds, nominal + val set, 200 episodes.
6.6 TRAINING RECIPE (train_ppo.py, written now; used by all later stages). CLI: --config <json> --seed <int> --out <dir>. Config keys: env cfg overrides, ppo hyperparameters, total_steps,
    val_every. Default ppo: n_envs=NENVS (SubprocVecEnv; torch.set_num_threads(1) in workers), n_steps=256, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, learning_rate=3e-4
    (constant), ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, policy net_arch pi=[128,128] vf=[128,128] tanh, VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10), total_steps=2000000,
    device cpu (cuda for GRU if available). Validation callback every 100000 steps: evaluate the CURRENT policy on val seeds (seed base 500000, the "val" condition set, 100 episodes per cell) with
    BatchEnv; score = mean lock_w1000 over the 4 cells; keep best_model.zip and vec_normalize.pkl at the best score. Print "PROGRESS steps=... fps=... val=..." every 30 s (flush) and write the
    same lines to <out>\progress.log. Save train_summary.json (config hash, seed, steps, wall seconds, best_val, val_history, versions, cmd). Training seeds: 0-4 exploratory, 100-109 confirmatory
    (never 5000+ or 900000+ for anything). Config file naming: configs\<family>__<probe>__<variant>.json ; model dir: runs\<config_name>\seed_<k>.
    Job: lane gpu, cwd STUDY, cmd: "<PY>" train_ppo.py --config configs\<name>.json --seed <k> --out runs\<name>\seed_<k>, expected_outputs best_model.zip, vec_normalize.pkl, train_summary.json,
    stall_minutes 45, max_wall_hours 24, watch [<out>\progress.log]. Chain training jobs with depends_on so only one gpu job runs at a time.
    Smoke test first: a 20000-step run of rl__dither__plain must finish and produce all three files (job, then check).
6.7 db ingest; DOC: docs\approach_1_dither.md (Appendix C), results tables generated from the database. docs_check PASS. state.py set S6 DONE.

======================================================================================================
STAGE 7 - APPROACH 2: RAMSEY WITH TWO QUADRATURE READOUTS
======================================================================================================
7.1 Add probe="ramsey_iq": two Ramsey circuits per step (I and Q), shots//2 each, via SurrogateEngine kinds ramsey_I/ramsey_Q and AerEngine; obs as in Stage 3.1.
7.2 Tests (test_ramsey.py): sign convention against Aer (already in test_aer_physics); recovered detuning bias < 60 Hz and std within +-30% of the Fisher floor over 2000 draws at delta in
    {-6000,-3000,0,3000,6000}; aliasing demonstration: for tau=100us the estimate wraps beyond +-5 kHz (record); PIDController(ramsey_iq) lock_w1000 at nominal > 0.85 on tune seeds.
7.3 Experiments: tau sweep for PID and (after Stage 8) Bayes: tau in {25,50,100} us; RL config rl__ramsey_iq__plain seeds 0-4; evaluate all on val seeds, nominal and val set.
7.4 db ingest; DOC docs\approach_2_ramsey.md; docs_check PASS; state.py set S7 DONE.

======================================================================================================
STAGE 8 - APPROACH 7: REALISTIC NOISE AND HARDWARE QUANTISATION (done before 3, 6, 4 so they inherit it)
======================================================================================================
8.1 Add to BatchEnv: action delay (0-3 steps), DAC quantisation (dac_bits), observation quantisation (obs_bits), per-episode arrays for shots/e01/e10/t2/sigma (needed by Stage 12); T2 acts
    through the Ramsey contrast. Add to AerEngine: depolarizing gate error option (probability 0 by default; the surrogate applies the same factor (1-4p/3)^(#gates) to the Ramsey contrast;
    validate the equivalence at p=0.002).
8.2 Tests (test_noise.py): delay=2 makes NoControl unchanged and Oracle worse than Oracle at delay 0 (record); dac_bits=6 gives measurable lock_w1000 loss for PID(ramsey_iq) vs 12 bits (record);
    T2 -> infinite raises the Fisher floor accuracy as predicted (floor ratio within 10%); readout error 0.2 symmetric: phase estimate unbiased.
8.3 Experiments (baselines only, on val seeds, 200 episodes): PID(rabi-HillClimb, dither, ramsey_iq) over all OFAT noise factors of Appendix D (shots, readout, t2, delay, dac_bits).
    Output: tables of lock_w1000 vs each factor and probe.
8.4 db ingest; DOC docs\approach_7_noise.md; docs_check PASS; state.py set S8 DONE.

======================================================================================================
STAGE 9 - APPROACH 3: BAYESIAN-FILTER BASELINE AND HYBRID RL
======================================================================================================
9.1 Add to controllers.py: BayesTracker(cfg, assumed): vectorised GridBayes (qphysics) with grid half_range 30000, n=601, window=filter_window; per step: predict, build the log-likelihood from the
    probe model with the ASSUMED parameters (assumed = nominal config for ":nominal", the episode's true parameters for ":oracle"), update, estimate = mean_std(window) mode, action =
    clip((mu - fc)/action_scale, -1, 1). The drift model inside the filter: random walk (theta=0) with sigma=filter_sigma or the cfg sigma; theta=0.05 if drift is "ou" and assumed=oracle.
    Probe models: rabi (single), dither (two probes), ramsey_iq (two quadratures), all with readout error and T2 from the assumed parameters. Counts come from public_state().
9.2 Add filter features to BatchEnv (cfg filter_features=True): the same filter runs inside the env after each measurement and exposes [(mu-fc)/5000, sigma_f/5000].
9.3 Tests (test_bayes.py): REFERENCE REPRODUCTION. With cfg identical to Appendix B file 2 (shots 1024, e=0.02, T2=200us, tau=50us, sigma 500, init offset 5000, 200 episodes, seed_base so that results are
    comparable) BayesTracker must reproduce the REF values within tolerance: lock1k within +-0.04 absolute and rms within +-15% for all six rows (gaussian/ou x rabi/dither/ramsey). Also: a dither or ramsey
    tracker never exceeds the oracle ceiling (0.954 +- 0.02 at lock_w1000); the filter-feature obs has finite values for 10000 steps; re-run validation V3 of Stage 4 with the Bayes tracker on both engines
    (|difference| <= 0.04) and store it.
9.4 Experiments (val seeds): Bayes ":nominal" and ":oracle" for the 3 probes at nominal and the "val" set; RL hybrid configs rl__rabi__filter, rl__dither__filter, rl__ramsey_iq__filter seeds 0-4.
9.5 db ingest; DOC docs\approach_3_bayes_hybrid.md (include the mismatch analysis: filter assumes nominal sigma/drift/T2 while the stress cell differs); docs_check PASS; state.py set S9 DONE.

======================================================================================================
STAGE 10 - APPROACH 6: TEMPORAL MEMORY
======================================================================================================
10.1 Frame stacking is the BatchEnv option stack=K (Stage 3.1). Write STUDY\policies.py: class WindowGRUExtractor(stable_baselines3.common.torch_layers.BaseFeaturesExtractor): input observation
    shape (K*D,) (layout [oldest..newest], D = per-frame dimension passed in as argument), reshape to (batch, K, D), one nn.GRU(input_size=D, hidden_size=64, batch_first=True), features = last hidden state
    (features_dim=64). train_ppo.py option "memory": {"type":"stack","K":k} (MLP on the stacked obs) or {"type":"gru","K":8}. If sb3_contrib is installed also support {"type":"recurrent"} with RecurrentPPO and an LSTM
    policy trained on stack=1 episodes; if not installed record SKIPPED(sb3_contrib not installed) in the document.
10.2 Tests (test_memory.py): shapes: for K in {4,8,16} the observation length equals K*D and the newest frame is the last D entries; the extractor forward pass on a random batch returns (batch,64) and
    gradients are finite; a 20000-step smoke training for stack K=4 and for gru K=8 finishes and writes all files (jobs). THIS EXTRACTOR AND ITS TRAINING PATH HAVE NOT BEEN RUN BY THE CONTROLLER: if a test
    fails, fix your own code (X2).
10.3 Experiments (exploratory, seeds 0-4): probes {rabi, ramsey_iq} x variants {K4, K8, K16, GRU8}: configs rl__<probe>__K4, ...__K8, ...__K16, ...__gru8 (40 runs). K=1 is the plain config from Stage 6/7.
    Evaluate on val seeds, nominal + val set.
10.4 db ingest; DOC docs\approach_6_memory.md; docs_check PASS; state.py set S10 DONE.

======================================================================================================
STAGE 11 - APPROACH 4: DOMAIN RANDOMIZATION, CURRICULUM, AER FINE-TUNE; then TUNING, COMBINATIONS, SELECTION, FREEZE
======================================================================================================
11.1 DR in BatchEnv: cfg dr={"scale":s} resamples per episode (surrogate only): drift kind uniformly from {gaussian, ou, onef, telegraph, sinusoid}; sigma log-uniform 250-1500 Hz; shots log-uniform 64-4096 (integer);
     e01 U(0,0.05), e10 U(0,0.08); t2 log-uniform 40-400 us; action_delay in {0,1,2}; dac_bits in {8,10,12}. Interpolation toward nominal with scale s in [0,1]: value = nominal + s*(sample - nominal) for numeric
     parameters; categorical parameters take the sampled value with probability s else nominal. Curriculum: train_ppo.py callback calls env_method("set_curriculum", s) every 10000 steps with
     s = min(1, steps/(0.5*total_steps)). Config families: rl__<probe>__DR (s fixed 1.0 from the start) and rl__<probe>__DRcurr.
11.2 Tests (test_dr.py): sampled parameter ranges respect the bounds (10000 samples); s=0 reproduces nominal exactly; curriculum callback is called (a 20000-step smoke training).
11.3 Experiments (exploratory, seeds 0-4): probes {rabi, dither, ramsey_iq} x {DR, DRcurr} = 6 configs (30 runs). Evaluate on val seeds nominal + val set.
11.4 Aer fine-tune (4c): take rl__ramsey_iq__DRcurr seeds 0,1,2 (their best_model.zip) and fine-tune 300000 steps on the Aer engine (nominal noise, NENVS=4 workers, lane aer, stall_minutes 120, watch progress
     log); config rl__ramsey_iq__DRcurr_aerft. Evaluate zero-shot (surrogate-trained) vs fine-tuned on Aer val seeds (100 episodes, nominal): report both.
11.5 HYPERPARAMETER TUNING (tune seed 100, one seed per config, 1000000 steps, probe dither plain): grid learning_rate {3e-4,1e-4} x ent_coef {0.0,0.003} x gamma {0.99,0.995} = 8 runs; select by the val score in
     train_summary.json. The winning hyperparameters are then used by ALL later (confirmatory) trainings; if they differ from the default, retrain nothing from the exploratory phase (the exploratory results keep their
     labels). Record the table in docs\tuning.md.
11.6 COMBINATION RUNS (exploratory, seeds 0-4): "full system" = best probe by validation + filter features + best memory variant + best DR variant, for the best two probes: configs rl__<probe>__full (10 runs).
11.7 db ingest; DOC docs\approach_4_dr_curriculum.md; docs_check PASS.
11.8 SELECTION AND FREEZE. Using ONLY validation scores (never sealed): write configs\frozen\selection.json listing the CONFIRMATORY SET:
     C1 legacy-analog = rl__rabi__plain ; C2 = best plain RL config over the 3 probes ; C3 = best hybrid (filter) config ; C4 = best memory config ; C5 = best DR/DRcurr config ; C6 = best "full" config.
     (If two names coincide, keep both rows and note it.) Copy the config json files and configs\pid_gains.json into configs\frozen\. Baselines frozen: ctrl:pid (tuned), ctrl:hill, ctrl:bayes:nominal, ctrl:bayes:oracle,
     ctrl:noctrl, ctrl:oracle. Append a "FREEZE ADDENDUM" to PREREG.md (the selected names, the hyperparameters from 11.5, the date), then compute sha256 of PREREG.md and write it to STUDY\PREREG.sha256 and into the database.
     From this moment PREREG.md and configs\frozen\ are read-only for you.
11.9 state.py set S11 DONE.

======================================================================================================
STAGE 12 - CONFIRMATORY TRAINING, SEALED EVALUATION, STRESS TESTS, AER
======================================================================================================
12.1 Train every config in the confirmatory set with seeds 100-109 (10 seeds each) with the frozen hyperparameters (gpu lane, chained). Model dirs runs\<name>\seed_<k>.
12.2 Sealed evaluation (seed base 900000; paired across all methods): for every confirmatory model and every baseline:
     (a) nominal condition, 1000 episodes; (b) ALL "ofat" cells of Appendix D, 300 episodes each; (c) ALL "grids" cells, 100 episodes each. Jobs round-robin over cpu1..cpuK; one job per (method, cond set); resumable.
12.3 Aer evaluation (aer lane, sealed seed base 900000, 100 episodes): nominal plus 6 cells (shots 128, shots 16384, t2 50us, readout 0.10, delay 1, dac_bits 8) for: all C1-C6 models of seeds 100-102 (3 seeds), the baselines
     pid/bayes:nominal/bayes:oracle/hill, noctrl. Report next to the surrogate results for the same seeds and cells (paired).
12.4 db ingest everything. state.py set S12 DONE.

======================================================================================================
STAGE 13 - ANALYSIS, DATABASE, FIGURES, PAPER PACKAGE
======================================================================================================
13.1 stats.py: implement Appendix D's statistical plan exactly: per-seed scores; endpoints P1 (nominal lock_w1000) and P2 (mean lock_w1000 over all OFAT cells = stress-macro); the 8 contrasts C-1..C-8 on P2
     with Mann-Whitney U (scipy, method exact, two-sided), Cliff's delta, bootstrap 95% CI of the mean difference (10000 resamples of seeds, seeded), Holm correction over the 8; baseline contrasts as paired
     differences (seed score minus baseline score) with bootstrap CI and the count of seeds above baseline; secondary metrics tables with mean, sd, 95% CI; everything not in the 8 contrasts is labelled EXPLORATORY.
     Also: Cramer-Rao (static Fisher) floor vs achieved rms error per probe and shot budget; Allan deviation of the residual error series (averaging times 1,2,4,8,16,32 steps, from stored traces); PSD (Welch) of residual error.
     Outputs to results\tables\*.csv, ingested into the database table contrasts and others.
13.2 db.py: all tables of Appendix D filled; add "export_csv" (writes db\export\*.csv.gz), "describe" (writes docs\DB_README.md with schema, row counts and 12 example SQL queries each executed and its result shown) and
     db\qstudy.sqlite integrity_check = ok. Write STUDY\reproduce.py: runs a 5-minute smoke subset (physics tests, one 20000-step training, one evaluation, a database query) and prints REPRODUCE OK.
13.3 figures.py: generate figures F01-F28 (Appendix C list) from the database/NPZ only; each as PNG (300 dpi) and PDF; consistent colours (one colour per method family); axis labels with units; confidence bands
     where applicable; figures\INDEX.md with caption stubs containing n, seeds, conditions. Qiskit circuit drawings: try draw("mpl"); on failure save the text drawing and a matplotlib schematic. Bloch-sphere figure: try
     qiskit.visualization.plot_bloch_vector; on failure use your own matplotlib 3D sphere.
13.4 Update each docs\approach_*.md with a CONFIRMATORY RESULTS section (generated tables + figure references) and a final "Interpretation" that states what is supported, what is not, and every failed expectation.
13.5 PAPER PACKAGE in docs\paper\: methods.md (full methods incl. equations, drift catalog, probes, filter, RL setup), experimental_setup.md (matrix, seeds, protocol, pre-registration summary with the hash),
     results_tables.md (generated), figure_list.md (with captions), discussion_notes.md (only evidence-backed observations; each with the table/figure that supports it), limitations.md (simulation-only, noise-free Rabi
     pulse, privileged training reward, single qubit, surrogate-vs-Aer gap, drift models are synthetic, no hardware), threats_to_validity.md, reproducibility.md (commands, versions, hashes, seeds), data_descriptor.md
     (what the database contains, how to query it), claims_register.csv (every claim: statement, supporting table/figure id, status SUPPORTED/NOT_SUPPORTED/EXPLORATORY).
13.6 state.py set S13 DONE.

======================================================================================================
STAGE 14 - FINAL VERIFICATION AND REPORT
======================================================================================================
14.1 Run the entire unit-test suite (job). Run reproduce.py (job). Run docs_check.py on every doc. Run the database integrity check.
14.2 Manifest proof: build manifest_AFTER of PROJECT_ROOT (job); python manifest.py diff BEFORE AFTER: ADDED must be only paths under qstudy\ ; CHANGED and REMOVED must be empty. If not: STOP making changes, record
     "ORIGINAL ANOMALY" and the list, and report (restore from BACKUP only for files you did not create: and only by copying, never by deleting qstudy).
14.3 Write STUDY\FINAL_QSTUDY_REPORT.md with sections: 1 Executive summary (12 lines: counts of runs/models/evaluations/figures/docs, headline results for P1 and P2 with CIs, all 8 contrasts with Holm p and effect sizes,
     the oracle ceiling, what the data do and do not support); 2 Step ledger (state.py show output with evidence files); 3 Foundation and validation (Stage 3-4 outcomes); 4 Legacy diagnosis; 5 One subsection per approach
     (1..7) with result tables and verdict (SUPPORTED / NOT SUPPORTED / MIXED + reason); 6 Tuning and selection; 7 Confirmatory and stress results (tables + figure index); 8 Surrogate vs Aer; 9 Anomalies and BLOCKED items
     (with evidence); 10 Everything not done and why; 11 File and database inventory with sizes; 12 Versions and hashes; 13 What I (controller) must decide next.
14.4 Print in chat: section 1, the contrast table, the BLOCKED list, the figure index headings, state.py show. Create WORK\STOP to stop runners and the watchdog.
14.5 End with exactly: QSTUDY COMPLETE - ORIGINAL PROJECT UNCHANGED - AWAITING REVIEW.

======================================================================================================
APPENDIX A - TOOL FILES (write each VERBATIM into WORK\tools\)
======================================================================================================
=== FILE: state.py ===
import datetime, json, os, sys

def path(w): return os.path.join(w, "STATE.json")

def load(w):
    try:
        with open(path(w), encoding="utf-8") as f: return json.load(f)
    except Exception:
        return {"steps": {}}

def save(w, s):
    with open(path(w) + ".tmp", "w", encoding="utf-8") as f: json.dump(s, f, indent=1)
    os.replace(path(w) + ".tmp", path(w))

if __name__ == "__main__":
    w, cmd = sys.argv[1], sys.argv[2]
    s = load(w)
    if cmd == "set":
        s["steps"][sys.argv[3]] = {"state": sys.argv[4], "note": " ".join(sys.argv[5:]), "time": datetime.datetime.now().isoformat(timespec="seconds")}
        save(w, s)
    elif cmd == "show":
        for k, v in s["steps"].items(): print(k, v["state"], v["time"], v["note"][:100])
    elif cmd == "get":
        print(s["steps"].get(sys.argv[3], {}).get("state", "NONE"))
=== END FILE ===

=== FILE: manifest.py ===
import csv, hashlib, os, sys
SKIP = {".git", "__pycache__", "node_modules"}

def walk(root):
    for d, ds, fs in os.walk(root):
        ds[:] = [x for x in ds if x not in SKIP and not os.path.exists(os.path.join(d, x, "pyvenv.cfg")) and not os.path.exists(os.path.join(d, x, "conda-meta"))]
        for f in fs: yield os.path.join(d, f)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def build(root, out):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["rel", "size", "mtime", "sha256"])
        for p in walk(root):
            try:
                s = os.stat(p); w.writerow([os.path.relpath(p, root), s.st_size, int(s.st_mtime), sha(p)])
            except Exception as e:
                w.writerow([os.path.relpath(p, root), "", "", "ERR:%s" % type(e).__name__])

def load(p):
    with open(p, encoding="utf-8") as f: return {r["rel"]: r for r in csv.DictReader(f)}

if __name__ == "__main__":
    if sys.argv[1] == "build": build(sys.argv[2], sys.argv[3])
    else:
        a, b = load(sys.argv[2]), load(sys.argv[3])
        print("ADDED", sorted(set(b) - set(a))); print("REMOVED", sorted(set(a) - set(b)))
        print("CHANGED", sorted(k for k in a if k in b and a[k]["sha256"] != b[k]["sha256"]))
=== END FILE ===

=== FILE: jobtools.py ===
import glob, json, os, sys

def submit(work, jid, lane, cwd, cmd, depends_on=(), expected_outputs=(), env=None, max_retries=2, stall_minutes=60, max_wall_hours=24, watch=()):
    p = os.path.join(work, "queue", jid + ".job.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p): return "EXISTS"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dict(id=jid, lane=lane, cwd=cwd, cmd=cmd, depends_on=list(depends_on), expected_outputs=list(expected_outputs), env=env or {}, max_retries=max_retries, stall_minutes=stall_minutes, max_wall_hours=max_wall_hours, watch=list(watch)), f, indent=1)
    return "SUBMITTED"

if __name__ == "__main__":
    w = sys.argv[1]
    if len(sys.argv) > 3 and sys.argv[2] == "submit":
        for j in json.load(open(sys.argv[3], encoding="utf-8")):
            j = dict(j); jid = j.pop("id")
            print(jid, submit(w, jid, **j))
        sys.exit(0)
    n = {}
    for p in sorted(glob.glob(os.path.join(w, "queue", "*.job.json"))):
        with open(p, encoding="utf-8") as f: j = json.load(f)
        sp = p.replace(".job.json", ".status.json")
        s = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {"state": "PENDING"}
        n[s["state"]] = n.get(s["state"], 0) + 1
        if "-v" in sys.argv or s["state"] in ("RUNNING", "FAILED", "BLOCKED"):
            print(j["id"], j["lane"], s["state"], s.get("attempts", 0), (s.get("tail") or s.get("note") or "")[-80:].replace("\n", " "))
    print("COUNTS", n)
=== END FILE ===

=== FILE: job_runner.py ===
import datetime, glob, json, os, re, subprocess, sys, time, traceback
WORK, LANE = sys.argv[1], sys.argv[2]
Q, L = os.path.join(WORK, "queue"), os.path.join(WORK, "logs")
os.makedirs(Q, exist_ok=True); os.makedirs(L, exist_ok=True)
TRANSIENT = re.compile(r"CUDA out of memory|PermissionError|WinError 32|WinError 5|Access is denied|Resource temporarily|Connection reset|MemoryError|cudaError|DefaultCPUAllocator", re.I)

def now(): return datetime.datetime.now().isoformat(timespec="seconds")

def rd(p):
    for _ in range(6):
        try:
            with open(p, encoding="utf-8") as f: return json.load(f)
        except Exception: time.sleep(0.5)
    return None

def wr(p, d):
    with open(p + ".tmp", "w", encoding="utf-8") as f: json.dump(d, f, indent=1)
    os.replace(p + ".tmp", p)

def status(jid): return rd(os.path.join(Q, jid + ".status.json")) or {"state": "PENDING", "attempts": 0}

def setstat(jid, **kw):
    s = status(jid); s.update(kw); s["updated"] = now(); wr(os.path.join(Q, jid + ".status.json"), s)

def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % int(pid), "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return False
    for line in out.splitlines():
        parts = line.strip().strip('"').split('","')
        if len(parts) > 1 and parts[1] == str(int(pid)): return True
    return False

def beat(msg=""):
    with open(os.path.join(WORK, "heartbeat_%s.txt" % LANE), "w") as f: f.write(now() + " " + msg)

def nap(sec):
    end = time.time() + sec
    while time.time() < end:
        beat("backoff"); time.sleep(max(0, min(10, end - time.time())))

def kill(pid): subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)

def all_jobs():
    out = {}
    for p in sorted(glob.glob(os.path.join(Q, "*.job.json"))):
        j = rd(p)
        if j: out[j["id"]] = j
    return out

def finish(jid, j, rc, reason, log, att):
    ok = os.path.join(Q, jid + ".ok")
    tail = open(log, errors="replace").read()[-3000:] if log and os.path.exists(log) else ""
    good = os.path.exists(ok) and all(os.path.exists(x) for x in j.get("expected_outputs", []))
    if good:
        setstat(jid, state="DONE", rc=rc, finished=now()); return
    retry = (reason or "").startswith("STALL") or bool(TRANSIENT.search(tail))
    if retry and att <= j.get("max_retries", 2):
        setstat(jid, state="PENDING", rc=rc, note="retry %d after %s" % (att, reason or "transient error"), tail=tail[-500:])
        nap(min(600, 60 * att))
    else:
        setstat(jid, state="FAILED", rc=rc, finished=now(), reason=reason, tail=tail[-1500:])

def watch(jid, j, pid, log, att, p=None):
    t0, reason = time.time(), None
    stall, maxw = j.get("stall_minutes", 60) * 60, j.get("max_wall_hours", 24) * 3600
    wf = j.get("watch", [])
    while (p.poll() is None) if p else alive(pid):
        time.sleep(10); beat(jid)
        ts = [os.path.getmtime(x) for x in [log] + wf if x and os.path.exists(x)]
        age = time.time() - (max(ts) if ts else t0)
        if age > stall: reason = "STALL no output for %d min" % (age // 60)
        elif time.time() - t0 > maxw: reason = "MAXWALL %s h" % j.get("max_wall_hours", 24)
        if reason:
            kill(pid); time.sleep(5); break
    finish(jid, j, (p.poll() if p else None), reason, log, att)

def run_job(j):
    jid = j["id"]; att = status(jid).get("attempts", 0) + 1
    log = os.path.join(L, "%s.attempt%d.log" % (jid, att)); ok = os.path.join(Q, jid + ".ok")
    if os.path.exists(ok): os.remove(ok)
    env = dict(os.environ); env.update(j.get("env", {})); env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    with open(log, "wb") as lf:
        p = subprocess.Popen(j["cmd"] + ' && echo OK> "%s"' % ok, shell=True, cwd=j["cwd"], env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        setstat(jid, state="RUNNING", pid=p.pid, attempts=att, log=log, started=now())
        watch(jid, j, p.pid, log, att, p)

def single_instance():
    pf = os.path.join(WORK, "runner_%s.pid" % LANE)
    try:
        old = int(open(pf).read().strip())
    except Exception:
        old = None
    if old and old != os.getpid() and alive(old):
        sys.exit(0)
    with open(pf, "w") as f: f.write(str(os.getpid()))

def main():
    for jid, j in all_jobs().items():
        s = status(jid)
        if j["lane"] == LANE and s["state"] == "RUNNING":
            if alive(s.get("pid", -1)): watch(jid, j, s["pid"], s.get("log", ""), s.get("attempts", 1))
            else: finish(jid, j, None, None, s.get("log", ""), s.get("attempts", 1))
    while not os.path.exists(os.path.join(WORK, "STOP")):
        beat("idle"); ran = False
        for jid, j in all_jobs().items():
            if j["lane"] != LANE or status(jid)["state"] != "PENDING": continue
            dep = [status(d)["state"] for d in j.get("depends_on", [])]
            if any(d in ("FAILED", "BLOCKED") for d in dep):
                setstat(jid, state="BLOCKED", note="dependency failed"); continue
            if all(d == "DONE" for d in dep):
                run_job(j); ran = True; break
        if not ran: time.sleep(15)

single_instance()
while True:
    try:
        main(); break
    except SystemExit:
        raise
    except Exception:
        with open(os.path.join(L, "runner_%s.err" % LANE), "a") as f: f.write(now() + "\n" + traceback.format_exc() + "\n")
        time.sleep(30)
=== END FILE ===

=== FILE: launch_runner.ps1 ===
param($Py, $Work, $Lane)
$cmd = '"' + $Py + '" "' + $Work + '\tools\job_runner.py" "' + $Work + '" ' + $Lane
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd; CurrentDirectory=$Work}
"$($r.ReturnValue) $($r.ProcessId)"
=== END FILE ===

=== FILE: watchdog.ps1 ===
param($Py, $Work, $Lanes)
$ls = $Lanes -split ','
while (-not (Test-Path "$Work\STOP")) {
  foreach ($l in $ls) {
    $hb = "$Work\heartbeat_$l.txt"
    $stale = $true
    if (Test-Path $hb) { $stale = (((Get-Date) - (Get-Item $hb).LastWriteTime).TotalSeconds -gt 240) }
    if ($stale) {
      $cmd = '"' + $Py + '" "' + $Work + '\tools\job_runner.py" "' + $Work + '" ' + $l
      Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd; CurrentDirectory=$Work} | Out-Null
      Start-Sleep -Seconds 20
    }
  }
  Start-Sleep -Seconds 60
}
=== END FILE ===

======================================================================================================
APPENDIX B - VERIFIED REFERENCE CODE (copy VERBATIM; verified against Qiskit Aer 0.17 by the controller)
======================================================================================================
=== FILE 1: STUDY\qphysics.py ===
"""Reference physics for the quantum resonator-tracking study (verified against Qiskit Aer)."""
import numpy as np

F_NOM = 500000.0          # Hz, nominal qubit/resonator frequency
CLIP = 25000.0            # Hz, drift is clipped to F_NOM +/- CLIP

# ---------------------------------------------------------------- probes (ideal expected P(|1>))
def rabi_p(delta, omega=5000.0):
    """Detuned Rabi pi-pulse of duration t=1/(2*omega). delta, omega in Hz. Peak (P=1) at delta=0."""
    d = np.asarray(delta, dtype=float)
    w = np.sqrt(omega ** 2 + d ** 2)
    return (omega ** 2 / w ** 2) * np.sin(np.pi * w / (2.0 * omega)) ** 2

def rabi_unitary(delta, omega=5000.0):
    """2x2 unitary of the same pulse (for Aer). H=(wr/2)sx+(dl/2)sz, wr=2*pi*omega, dl=2*pi*delta, t=1/(2*omega)."""
    wr, dl = 2 * np.pi * omega, 2 * np.pi * float(delta)
    w = np.hypot(wr, dl)
    t = 1.0 / (2.0 * omega)
    c, s = np.cos(w * t / 2), np.sin(w * t / 2)
    sx = np.array([[0, 1], [1, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    return c * np.eye(2) - 1j * s * (wr * sx + dl * sz) / w

def ramsey_contrast(tau, t2):
    """exp(-tau/T2); t2 may be None/inf (no dephasing) or an array (per-episode)."""
    t2 = np.inf if t2 is None else np.asarray(t2, dtype=float)
    return np.exp(-tau / t2)

def ramsey_p(delta, tau=50e-6, t2=200e-6, quad="I"):
    """Ramsey with Rx(pi/2)-wait(tau)-[Rx|Ry](pi/2). I: (1+C cos phi)/2 ; Q: (1+C sin phi)/2 ; phi=2*pi*delta*tau."""
    c = ramsey_contrast(tau, t2)
    phi = 2 * np.pi * np.asarray(delta, dtype=float) * tau
    return 0.5 * (1.0 + c * (np.cos(phi) if quad == "I" else np.sin(phi)))

def readout(p1, e01=0.0, e10=0.0):
    """e01 = P(read 1 | state 0), e10 = P(read 0 | state 1)."""
    return e01 + (1.0 - e01 - e10) * np.asarray(p1, dtype=float)

def sample_counts(rng, p, shots):
    return rng.binomial(np.asarray(shots, dtype=np.int64), np.clip(p, 0.0, 1.0))

def ramsey_phase(pI_hat, pQ_hat):
    """Signed phase estimate in (-pi, pi]; detuning estimate = phase/(2*pi*tau), unambiguous for |delta|<1/(2*tau)."""
    return np.arctan2(2 * np.asarray(pQ_hat) - 1, 2 * np.asarray(pI_hat) - 1)

# ---------------------------------------------------------------- Fisher information / CRB (static estimate floor)
def fisher_ramsey_iq(delta, shots_each, tau=50e-6, t2=200e-6, e01=0.0, e10=0.0):
    c = ramsey_contrast(tau, t2)
    phi = 2 * np.pi * np.asarray(delta, float) * tau
    k = (1 - e01 - e10) * 0.5 * c * 2 * np.pi * tau
    pI = readout(ramsey_p(delta, tau, t2, "I"), e01, e10)
    pQ = readout(ramsey_p(delta, tau, t2, "Q"), e01, e10)
    dI, dQ = -k * np.sin(phi), k * np.cos(phi)
    eps = 1e-9
    return shots_each * (dI ** 2 / np.clip(pI * (1 - pI), eps, None) + dQ ** 2 / np.clip(pQ * (1 - pQ), eps, None))

def fisher_rabi(delta, shots, omega=5000.0, e01=0.0, e10=0.0, h=1.0):
    p = lambda d: readout(rabi_p(d, omega), e01, e10)
    dp = (p(np.asarray(delta, float) + h) - p(np.asarray(delta, float) - h)) / (2 * h)
    pp = p(delta)
    return shots * dp ** 2 / np.clip(pp * (1 - pp), 1e-9, None)

# ---------------------------------------------------------------- drift generators (per control step)
class Drift:
    """Offset (Hz) of the true frequency from F_NOM. kinds: gaussian, ou, onef, telegraph, sinusoid."""
    THETA_ONEF = np.array([0.5, 0.2, 0.08, 0.03, 0.012, 0.005])

    def __init__(self, kind, rng, sigma=500.0, theta=0.05, amp=3000.0):
        self.kind, self.rng, self.sigma, self.theta, self.amp = kind, rng, float(sigma), float(theta), float(amp)

    def reset(self, init_offset):
        r = self.rng
        self.t = 0
        self.x = float(init_offset)
        if self.kind == "onef":
            var_ou = self.sigma ** 2 / (2 * self.theta - self.theta ** 2)
            v = var_ou / len(self.THETA_ONEF)
            self.s = np.sqrt(v * (2 * self.THETA_ONEF - self.THETA_ONEF ** 2))
            self.comp = r.normal(0.0, np.sqrt(v), len(self.THETA_ONEF))
            self.comp += (self.x - self.comp.sum()) / len(self.THETA_ONEF)
        elif self.kind == "telegraph":
            self.state, self.y = 1.0 if r.random() < 0.5 else -1.0, 0.0
            self.x = self.amp * self.state * 0.0 + float(init_offset)
            self.base = float(init_offset)
        elif self.kind == "sinusoid":
            self.period, self.ph = r.uniform(60, 120), r.uniform(0, 2 * np.pi)
            self.y, self.base = 0.0, float(init_offset)
        return self.x

    def step(self):
        r, k = self.rng, self.kind
        if k == "gaussian":
            self.x += self.sigma * r.normal()
        elif k == "ou":
            self.x += -self.theta * self.x + self.sigma * r.normal()
        elif k == "onef":
            self.comp += -self.THETA_ONEF * self.comp + self.s * r.normal(size=self.comp.shape)
            self.x = float(self.comp.sum())
        elif k == "telegraph":
            if r.random() < 0.02:
                self.state = -self.state
            self.y += -0.2 * self.y + 0.5 * self.sigma * r.normal()
            self.x = self.base + self.amp * self.state * 0.5 + self.y
        elif k == "sinusoid":
            self.t += 1
            self.y += -0.2 * self.y + 0.5 * self.sigma * r.normal()
            self.x = self.base + self.amp * np.sin(2 * np.pi * self.t / self.period + self.ph) + self.y
        else:
            raise ValueError(k)
        self.x = float(np.clip(self.x, -CLIP, CLIP))
        return self.x

def quantize(x, bits, half_range=CLIP):
    if bits is None:
        return x
    lsb = 2.0 * half_range / (2 ** int(bits))
    return np.round(np.asarray(x, float) / lsb) * lsb

# ---------------------------------------------------------------- grid Bayes filter (vectorised over a batch)
class GridBayes:
    """Posterior over true-frequency offset on a grid. drift model: x' = x - theta*x + sigma*N (theta=0 -> random walk)."""
    def __init__(self, half_range=30000.0, n=601, sigma=500.0, theta=0.0):
        self.g = np.linspace(-half_range, half_range, n)
        d = self.g[:, None] - ((1 - theta) * self.g)[None, :]          # row i: new x_i ; col j: old x_j
        T = np.exp(-0.5 * (d / sigma) ** 2)
        self.T = T / T.sum(axis=0, keepdims=True)                        # columns sum to 1
        self.n = n

    def init(self, batch, prior_std=8000.0):
        p = np.exp(-0.5 * (self.g / prior_std) ** 2)
        return np.tile(p / p.sum(), (batch, 1))

    def predict(self, post):
        out = post @ self.T.T
        return out / out.sum(axis=1, keepdims=True)

    def update(self, post, loglik):
        l = loglik - loglik.max(axis=1, keepdims=True)
        out = post * np.exp(l)
        s = out.sum(axis=1, keepdims=True)
        return out / np.where(s > 0, s, 1.0)

    def mean_std(self, post, window=5000.0):
        """Mean/std of the dominant posterior mode: restrict to |g - g_MAP| <= window (Hz) and renormalise.
        Needed because Ramsey/Rabi likelihoods are periodic/symmetric and the plain posterior mean averages alias peaks."""
        if window is None:
            m = post @ self.g
            v = post @ (self.g ** 2) - m ** 2
            return m, np.sqrt(np.clip(v, 0, None))
        gm = self.g[np.argmax(post, axis=1)]
        mask = (np.abs(self.g[None, :] - gm[:, None]) <= window)
        w = post * mask
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-300, None)
        m = w @ self.g
        v = w @ (self.g ** 2) - m ** 2
        return m, np.sqrt(np.clip(v, 0, None))

def binom_loglik(k, n, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return k * np.log(p) + (n - k) * np.log(1 - p)
=== END FILE ===

=== FILE 2: STUDY\tests\ref_tracker.py ===
"""Reference closed-loop check: grid-Bayes tracker on a batch of episodes. Prints lock fractions measured AT PROBE TIME."""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("tests", 1)[0].rstrip("\\/"))
from qphysics import *

def run_ref(kind, probe, B=200, T=200, shots=1024, e=0.02, t2=200e-6, tau=50e-6, sigma=500.0, seed=0, amax=3000.0, deltad=2500.0):
    rng = np.random.default_rng(seed)
    drifts = [Drift(kind, np.random.default_rng(seed * 1000 + i), sigma) for i in range(B)]
    x = np.array([d.reset(o) for d, o in zip(drifts, rng.uniform(-5000, 5000, B))])
    fc = np.zeros(B)
    gb = GridBayes(sigma=sigma, theta=0.05 if kind == "ou" else 0.0)
    post = gb.init(B)
    pe = np.zeros((T, B))
    for t in range(T):
        x = np.array([d.step() for d in drifts])
        pe[t] = x - fc                                   # error at probe time
        post = gb.predict(post)
        Dg = gb.g[None, :] - fc[:, None]
        if probe == "ramsey":
            n = shots // 2
            kI = sample_counts(rng, readout(ramsey_p(x - fc, tau, t2, "I"), e, e), n)
            kQ = sample_counts(rng, readout(ramsey_p(x - fc, tau, t2, "Q"), e, e), n)
            ll = binom_loglik(kI[:, None], n, readout(ramsey_p(Dg, tau, t2, "I"), e, e)) + binom_loglik(kQ[:, None], n, readout(ramsey_p(Dg, tau, t2, "Q"), e, e))
        elif probe == "dither":
            n = shots // 2
            kp = sample_counts(rng, readout(rabi_p(x - fc - deltad), e, e), n)
            km = sample_counts(rng, readout(rabi_p(x - fc + deltad), e, e), n)
            ll = binom_loglik(kp[:, None], n, readout(rabi_p(Dg - deltad), e, e)) + binom_loglik(km[:, None], n, readout(rabi_p(Dg + deltad), e, e))
        else:
            n = shots
            k = sample_counts(rng, readout(rabi_p(x - fc), e, e), n)
            ll = binom_loglik(k[:, None], n, readout(rabi_p(Dg), e, e))
        post = gb.update(post, ll)
        m, _ = gb.mean_std(post)
        fc = fc + np.clip(m - fc, -amax, amax)
    a = np.abs(pe[20:])
    return {"lock1k": float((a <= 1000).mean()), "lock2k": float((a <= 2000).mean()), "lock3333": float((a <= 3333).mean()), "rms": float(np.sqrt((a ** 2).mean()))}

if __name__ == "__main__":
    for kind in ("gaussian", "ou"):
        for probe in ("rabi", "dither", "ramsey"):
            r = run_ref(kind, probe)
            print("REF", kind, probe, " ".join("%s=%.3f" % (k, v) if k != "rms" else "rms=%.0f" % v for k, v in r.items()))
    rng = np.random.default_rng(1); d = rng.normal(0, 500.0, 200000)
    print("REF oracle_ceiling lock1k=%.3f lock2k=%.4f (one-step drift sigma=500 Hz)" % ((np.abs(d) <= 1000).mean(), (np.abs(d) <= 2000).mean()))

# EXPECTED OUTPUT (controller's run):
# REF gaussian rabi lock1k=0.571 lock2k=0.879 lock3333=0.982 rms=1388
# REF gaussian dither lock1k=0.951 lock2k=1.000 lock3333=1.000 rms=506
# REF gaussian ramsey lock1k=0.941 lock2k=1.000 lock3333=1.000 rms=528
# REF ou rabi lock1k=0.660 lock2k=0.937 lock3333=0.994 rms=1103
# REF ou dither lock1k=0.949 lock2k=1.000 lock3333=1.000 rms=512
# REF ou ramsey lock1k=0.941 lock2k=1.000 lock3333=1.000 rms=530
# REF oracle_ceiling lock1k=0.954 lock2k=1.0000 (one-step drift sigma=500 Hz)
=== END FILE ===

=== FILE 3: STUDY\tests\test_aer_physics.py ===
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("tests", 1)[0].rstrip("\\/"))
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, thermal_relaxation_error
from qphysics import *

SH = 200000
def run(qc, nm=None):
    s = AerSimulator(noise_model=nm) if nm else AerSimulator()
    qc.measure_all()
    r = s.run(qc, shots=SH, seed_simulator=7).result().get_counts()
    return r.get("1", 0) / SH

for d in [-9000, -3000, -800, 0, 500, 2500, 7000]:
    qc = QuantumCircuit(1); qc.append(UnitaryGate(rabi_unitary(d)), [0])
    p_aer, p_an = run(qc), float(rabi_p(d))
    print("rabi d=%6d aer=%.4f analytic=%.4f diff=%+.4f" % (d, p_aer, p_an, p_aer - p_an))
    assert abs(p_aer - p_an) < 0.004
tau, t2 = 50e-6, 200e-6
nm = NoiseModel()
nm.add_quantum_error(thermal_relaxation_error(t2 * 100, t2, tau), ["id"], [0])
e01, e10 = 0.03, 0.05
nm.add_readout_error(ReadoutError([[1 - e01, e01], [e10, 1 - e10]]), [0])
for d in [-9000, -4000, -1000, 0, 1500, 6000]:
    phi = 2 * np.pi * d * tau
    for quad in ["I", "Q"]:
        qc = QuantumCircuit(1); qc.rx(np.pi / 2, 0); qc.id(0); qc.rz(phi, 0)
        qc.rx(np.pi / 2, 0) if quad == "I" else qc.ry(np.pi / 2, 0)
        p_aer = run(qc, nm)
        p_an = float(readout(ramsey_p(d, tau, t2, quad), e01, e10))
        print("ramsey %s d=%6d aer=%.4f analytic=%.4f diff=%+.4f" % (quad, d, p_aer, p_an, p_aer - p_an))
        assert abs(p_aer - p_an) < 0.006
print("AER VALIDATION PASS")
=== END FILE ===

======================================================================================================
APPENDIX C - DOCUMENTATION TEMPLATE, DOC CHECK, FIGURE LIST
======================================================================================================
Every docs\approach_<n>_<name>.md has EXACTLY these numbered headings:
 1 Purpose and hypothesis | 2 Physics and mathematics (equations) | 3 Implementation (files, classes, functions with path:line) | 4 Parameters and defaults (table) |
 5 Validation tests (table: test, expected, observed, PASS/FAIL, evidence file) | 6 Experiments (design, seeds, conditions, episodes, split) |
 7 Results (tables generated by stats.py from the database, with n, mean, sd, 95% CI; figure references) | 8 Interpretation (supported / not supported / mixed, each with the table that shows it) |
 9 Limitations and threats | 10 Reproduction commands (exact) | 11 Raw data paths with sha256 and database tables.
docs_check.py (write it, run as a job) FAILS a document if: a heading is missing, section 5 or 7 has no table, the text contains TODO, TBD, XXX, "to be filled", a referenced file path does not exist, or a numeric
table cell in section 7 cannot be found in the database or a results file (check by tolerance 1e-6 against the generated CSV the table came from).
FIGURES (figures\figNN_<name>.png and .pdf): F01 circuits (Rabi pulse, Ramsey I, Ramsey Q) | F02 line shapes P(delta): legacy Rabi vs exact Rabi vs Ramsey I/Q with the +-dither points marked |
F03 Fisher/CRB floor vs detuning per probe | F04 Bloch-sphere paths of the Ramsey sequence for delta=0,+1000,+4000 Hz | F05 I/Q phasor clouds at several detunings (64 vs 1024 shots) |
F06 drift catalog: traces and PSD for the 5 drift kinds | F07 surrogate-vs-Aer validation: parity plot, residual histogram, closed-loop lock difference | F08 legacy diagnosis: per-episode MAE histogram, lock-loss survival, traces |
F09 probe comparison at nominal: lock_w1000 bars with CIs (Rabi/Dither/Ramsey x PID/Bayes/RL) | F10 tracking traces true vs control per probe | F11 error CDFs |e| (log x) per method | F12 lock-loss survival curves |
F13 Bayes posterior heatmap over time with true and control overlays | F14 Bayes nominal-assumption vs oracle-parameter under the sigma sweep | F15 memory ablation: lock vs K and GRU per probe | F16 training curves (val score vs steps, mean +- sd over seeds) |
F17 curriculum schedule and val score | F18 OFAT stress panel (all factors) for C1-C6 and baselines with CI bands | F19 2D heatmaps (shots x T2, sigma x readout) for three key methods plus difference maps |
F20 quantisation and delay panel | F21 Aer vs surrogate paired bars | F22 Aer fine-tune before/after | F23 Allan deviation of the residual error | F24 PSD of the residual error | F25 CRB vs achieved rms (shot sweep) |
F26 forest plot of the 8 pre-registered contrasts (effect and CI, Holm p) | F27 seed variability (violin per config) | F28 compute cost (steps/s, wall time per family).

======================================================================================================
APPENDIX D - TEXT OF STUDY\PREREG.md (copy VERBATIM in Stage 2.3)
======================================================================================================
=== FILE: STUDY\PREREG.md ===
# Pre-registration (written before any RL result of the study existed)
Study: tracking a drifting qubit frequency with RL, PID and Bayesian filtering in a simulated quantum environment.
## Definitions
- Error e_t = true offset - control offset at probe time. Burn-in: first 20 of 200 steps are excluded. Lock fraction L(W) = mean over t>=20 of 1[|e_t|<=W].
- PRIMARY WIDTH W = 1000 Hz. Secondary: 500, 2000, 3333, 5000 Hz. Other metrics: rms, median |e|, 95th percentile |e|, lock-loss rate (>=20 consecutive steps with |e|>10000 Hz), mean reward.
- Oracle ceiling at nominal: L(1000)=0.954 (one-step drift sigma 500 Hz).
## Nominal condition
probe as named; drift gaussian sigma 500 Hz/step; shots 1024 total per step (split in two halves for two-circuit probes); Rabi frequency 5000 Hz; dither delta 2500 Hz; Ramsey tau 50 us; T2 200 us; readout error e01=e10=0.02; action scale
3000 Hz/step; action delay 0; DAC 12 bits; observation not quantised; horizon 200.
## Endpoints
P1 = L(1000) at the nominal condition (1000 sealed episodes). P2 = macro-average of L(1000) over all OFAT stress cells (300 sealed episodes each). Per-seed score = mean over episodes. Any other quantity is EXPLORATORY.
## Stress conditions
OFAT around nominal (one factor changed at a time): drift_kind {gaussian, ou, onef, telegraph, sinusoid}; drift_sigma {250, 500, 1000, 2000, 4000}; shots {64, 128, 256, 512, 1024, 4096, 16384}; readout error e01=e10 {0, 0.02, 0.05, 0.10, 0.20};
t2 (us) {25, 50, 100, 200, 800, inf}; action_delay {0,1,2,3}; dac_bits {6, 8, 10, 12, 16}; omega (Hz) {2500, 5000, 10000}; tau (us) {25, 50, 100}. Grids: shots x t2 (7 x 6) and drift_sigma x readout error (5 x 5), 100 episodes per cell.
Factors that do not apply to a probe (omega for Ramsey, tau for Rabi/dither) are skipped for that probe and excluded from its P2 average (documented).
## Seeds and splits
Training seeds: exploratory 0-4; confirmatory 100-109 (fresh). Tuning episodes seed base 100000; validation 500000; sealed 900000 (used once, after the freeze). Episode e uses drift RNG [base, e, 1] and measurement RNG [base, e, 2] for all methods (paired).
## Methods
RL: PPO (SB3), reward 1 - min(1,|e|/4000) - 0.01|a| (privileged training signal; the observation never contains the true frequency), net 2x128 tanh, 2M steps, hyperparameters from the frozen tuning table.
Baselines: NoControl; Oracle (reference ceiling); HillClimb (single Rabi probe); PID per probe (grid-tuned on tuning seeds); Bayes tracker with assumed-nominal parameters; Bayes tracker with oracle parameters.
## Confirmatory contrasts (all on P2; Holm correction over the 8)
C-1 RL dither vs RL rabi (plain, K=1). C-2 RL ramsey_iq vs RL dither (plain). C-3 best plain RL vs PID (same probe). C-4 best plain RL vs Bayes assumed-nominal (same probe). C-5 hybrid (filter features) vs plain (same probe as C-3).
C-6 best memory variant vs K=1 on probe rabi. C-7 best DR variant vs fixed-condition training (same probe). C-8 DRcurr vs DR (same probe as C-7). The selected configurations are named in the FREEZE ADDENDUM.
## Statistics
Seed-level Mann-Whitney U (exact, two-sided) + Cliff's delta + bootstrap 95% CI (10000 resamples of seeds, fixed seed) for RL-vs-RL contrasts; paired seed-score minus baseline-score differences with bootstrap CI and the count of seeds above the baseline for RL-vs-baseline
contrasts. Pooled-episode p-values are never used for inference. Everything outside the 8 contrasts is exploratory.
## Rules
Negative or null results are reported. No run, seed, or condition is dropped. Sealed evaluations are run once. Model selection uses validation seeds only.
=== END FILE ===
======================================================================================================
END OF PROMPT
======================================================================================================