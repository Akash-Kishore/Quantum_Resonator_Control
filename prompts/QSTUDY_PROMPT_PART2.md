QSTUDY PART 2 - CONTINUATION AND EXTENSION   (executor: GLM 5.3)
THIS_FILE2 = path given in the launcher message.   PART1 = QSTUDY_PROMPT.md in the same folder as THIS_FILE2.
==========================================================================================================
SECTION P0 - ROLE AND RELATIONSHIP TO PART 1
==========================================================================================================
You are an EXECUTION ENGINE continuing a long project. I (the controller) did the thinking: designs, formulas, thresholds, tests, order. Follow them literally; do not redesign; never ask me questions.
Part 1 (PART1) is still authoritative: its Sections 0-3 (integrity I1-I5, rules R1-R10, timeout-proof execution T1-T10, exception playbook X1-X10), Stages 0-14 and Appendices A-D all remain in force, EXCEPT where this file
overrides them. This file (a) tells you how to recover context after a restart, (b) fixes problems found so far, (c) continues Part 1 from Stage 6, (d) adds new Stages 15-20.
The only deliberate stops anywhere in the whole project are: the FREEZE-READY gate (O2) and the final completion message. Everything else: never stop, never wait.
PROJECT_ROOT = your working directory (it must end in MiniProject_Sem4). All new files go in PROJECT_ROOT\qstudy\ and in WORK. Never modify any pre-existing file (Part 1 R1).

==========================================================================================================
SECTION P1 - BOOT: RECOVER CONTEXT WITHOUT RE-READING THE PROJECT
==========================================================================================================
B0  Derive paths exactly as Part 1 Stage 0.2 (PARENT, NAME, WORK = PARENT\<NAME>_QSTUDY_WORK, STUDY = PROJECT_ROOT\qstudy). WORK must exist, else STOP "WORK NOT FOUND - AWAITING INSTRUCTIONS".
    Read WORK\00_paths.txt. Do NOT re-read the old project code/docs/images: use WORK\understanding\ and STUDY\docs\01_project_understanding.md and 02_comprehension_answers.md instead.
B1  Read PART1 completely and then this file completely (chunks of at most 250 lines; write nothing yet).
B2  Runtime state: run  python WORK\tools\state.py WORK show ; python WORK\tools\jobtools.py WORK ; list every heartbeat_*.txt age. If a lane heartbeat is older than 240 s relaunch that lane (launch_runner.ps1) and the watchdog (Part 1 Stage 0.7).
B3  Chat exports: folder = PARENT\GLM Chat Exports (verify it exists; if not, search PARENT for a folder whose name contains "Chat Export"). List its files (name, size, date). Read EVERY file completely, oldest first, in chunks of at most 250 lines.
    After each chunk append to WORK\understanding\chat_digest.md: (a) controller instructions/decisions (verbatim, max 200 chars each); (b) what the previous agent session did (steps, files, jobs, numbers with where reported); (c) errors/anomalies; (d) open items.
    Keep WORK\understanding\chat_digest_progress.txt (file, last line read) so you can resume if interrupted.
B4  Write STUDY\docs\03_session_handover.md with: timeline; per-stage state from STATE.json versus what the exports say (list every discrepancy); the job table summary (counts per state per lane; list FAILED jobs with cause); the table of exploratory trainings completed per config and seed;
    the list of controller addenda already applied (verify each against disk: e.g. no gpu job has depends_on on another gpu job; unchain.py exists; DR-support column exists) and which are NOT applied.
B5  Gate: answer in STUDY\docs\04_handover_answers.md, each with a citation (export file:line or disk path), and verify citations mechanically with understanding\cite_check.py (reuse it): H1 which Part-1 stages are DONE/RUNNING/NOT STARTED; H2 job counts per state; H3 which configs have all 5 exploratory seeds trained; H4 the current best_val per config family;
    H5 where selection.json draft would go; H6 which unit tests exist and their last result; H7 disk free space; H8 the number of results files per split (confirm none is sealed); H9 how many seeds of each config exist in runs\ ; H10 any anomalies recorded.
B6  state.py set BOOT DONE. Print a 10-line summary. Continue at the first unfinished step (obey T8).

==========================================================================================================
SECTION P2 - OVERRIDES AND CORRECTIONS (apply now)
==========================================================================================================
O1  NO CHAINING OF GPU JOBS. The gpu lane already runs one job at a time. Never use depends_on between gpu-lane jobs (a single failure would block the rest). Evaluation jobs may depend on their own training only.
    If WORK\tools\unchain.py does not exist, write it exactly as below, run it (python WORK\tools\unchain.py WORK), print its output.
=== FILE: unchain.py ===
import glob, json, os, sys, time

def rd(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def wr(p, d):
    for _ in range(8):
        try:
            with open(p + ".tmp", "w", encoding="utf-8") as f: json.dump(d, f, indent=1)
            os.replace(p + ".tmp", p); return True
        except PermissionError:
            time.sleep(1)
    return False

w = sys.argv[1]; q = os.path.join(w, "queue")
jobs = {}
for p in glob.glob(os.path.join(q, "*.job.json")):
    j = rd(p); jobs[j["id"]] = (p, j)
changed = unblocked = 0
for jid, (p, j) in jobs.items():
    sp = p.replace(".job.json", ".status.json")
    s = rd(sp) if os.path.exists(sp) else {"state": "PENDING"}
    if s["state"] not in ("PENDING", "BLOCKED") or j["lane"] != "gpu":
        continue
    old = j.get("depends_on", [])
    keep = [d for d in old if d in jobs and jobs[d][1]["lane"] != "gpu"]
    if keep != old:
        j["depends_on"] = keep
        if wr(p, j): changed += 1
    if s["state"] == "BLOCKED":
        s["state"] = "PENDING"; s["note"] = "unchained (was blocked by a failed predecessor)"
        if wr(sp, s): unblocked += 1
print("unchained jobs:", changed, " unblocked jobs:", unblocked)
=== END FILE ===
O2  FREEZE GATE (changes Part 1 Stage 11.8). Do everything in 11.8 up to writing configs\frozen\selection.json as a DRAFT plus the ranking tables (validation score per config, per seed, mean/sd; the tuning table). Do NOT write the FREEZE ADDENDUM and do NOT create PREREG.sha256.
    Before printing, verify and print: (a) no result file with split=sealed exists; (b) no run directory for seeds 100-109 exists except the 8 tuning runs. Print the tables and the draft, then print exactly: FREEZE-READY - AWAITING CONTROLLER APPROVAL, and stop.
    I will answer "FREEZE APPROVED" (optionally with changes). Only after that message: write the addendum, hash, and continue with Part 1 Stage 12. No seed 100-109 training and no sealed evaluation before that message.
    (If the controller's approval is already present in the chat exports, record the quote in the handover and continue.)
O3  DR SUPPORT SPLIT. conditions table gets a column dr_support (IN/OUT): a stress cell is IN if all its values lie inside the DR sampling ranges of Part 1 Stage 11.1 (all five drift kinds; sigma 250-1500; shots 64-4096; e01<=0.05 and e10<=0.08; t2 40-400 us or nominal; delay<=2; dac_bits>=8; omega/tau variations count as IN), else OUT.
    In Stage 13 report P2 separately for IN and OUT cells as EXPLORATORY columns beside the pre-registered P2. Do not change the pre-registered P2.
O4  DOC POINTS. docs\step0_legacy_diagnosis.md and methods.md must state precisely, with equations and plots from the replica test, how the legacy Qiskit rx-angle measurement curve differs from the exact detuned Rabi (peak, width, symmetry, slope at the peak) and what that implies for the legacy results.
    Never compare a validation score (mean over the 4 val cells) with a nominal-condition number; always label which is which.
O5  EVERY evaluation output from now on also stores (extend evaluate.py now, keep backward compatibility): CVaR95 of per-episode rms and of per-episode (1 - lock_w1000) (mean of the worst 5% episodes), the 95th percentile of per-episode rms, lock-loss probability, per-episode first-lock time (first step with 3 consecutive |e|<=W), and for up to K lock-lost episodes per cell (flag --store_failures K, default 20) the full traces (e, fc, x, actions, counts, shots, mode, posterior sigma if available) plus K matched non-lost episodes.
    Existing result files are not rewritten; new files carry the new fields; the database columns are nullable.
O6  Part 1 Stage 6.6 is amended: train_ppo.py must (a) log "RSS_MB=<n>" in every progress line (psutil if present, else ctypes GetProcessMemoryInfo), (b) accept "snapshots": [steps] to save the CURRENT policy and VecNormalize statistics at those total steps (files snapshot_<steps>.zip and vecnorm_<steps>.pkl) in addition to best_model, (c) accept "algo": "ppo"|"sac"|"td3" (SB3 defaults except net_arch [128,128], learning_starts 5000, batch_size 256, gradient_steps 1; no VecNormalize for SAC/TD3: normalise observations with a fixed scale-free wrapper that is already in [-1,1]),
    (d) accept "init_from": <model dir> to continue training (fine-tune) from a saved model and its VecNormalize statistics (freeze obs normalisation statistics updates: set training=True anyway but continue from the loaded running stats). Add unit tests; existing configs must still produce the same results.
O7  Documentation rule stays: at the end of every approach/study a docs file with the Part 1 Appendix C template (all 11 headings), generated tables only from the database, docs_check PASS. Never type a number from memory.

==========================================================================================================
SECTION P3 - ORDER OF WORK
==========================================================================================================
 1. Finish Part 1 Stages 6-10 and 11.1-11.7 exactly as specified (their trainings run in the background on the gpu lane; use cpu lanes for evaluations).
 2. While trainings run: build and test the Part-2 code (S15.1 shared extensions, S16 profile code, S17 environments, S18/S19 analysis scripts) on the cpu lanes. Smoke trainings of at most 20000 steps are allowed.
 3. At 11.8 stop at FREEZE-READY (O2). While waiting, nothing else is queued on gpu except the Part-2 smoke tests.  (If the exploratory queue has not drained, you reach 11.8 later: that is fine.)
 4. After FREEZE APPROVED: write PREREG2 (Appendix E verbatim) and its hash FIRST (S15.0), then Part 1 Stage 12 confirmatory trainings, then the Part-2 training jobs in the order S15, S16, S17 (submit them all up front on the gpu lane; the lane runs them one at a time; never chain).
 5. Evaluations (cpu lanes, aer lane) run as soon as their models exist (evaluation jobs depend on their own training job only).
 6. Stage 13 (analysis/figures/paper package) and Stage 14 (final verification) of Part 1 are executed AFTER S15-S19 and are extended by S20 below.
Seeds and splits for Part 2 (never reuse): transfer training seeds 200-204; platform specialists 300-302; generalist 310-312; leave-one-platform-out 320-321; adaptive 400-404; DR-restricted 410-414; reward-check 420-422. Tuning base 100000 (PID/Hill/filters per profile), validation base 500000, SEALED bases: transfer 920000, platform 950000, adaptive/change-point/envelope/generalisation 960000. Sealed refuses to run unless PREREG2.sha256 exists and matches (extend evaluate.py).

==========================================================================================================
STAGE S15 - CLASSICAL-TRAINED VERSUS QUANTUM-TRAINED TRANSFER STUDY (plus classical/quantum environment correlation)
==========================================================================================================
S15.0 Write STUDY\PREREG2.md = Appendix E verbatim and compute sha256 -> STUDY\PREREG2.sha256 (do this at the start of Part-2 training, after FREEZE APPROVED; before that only code and smoke tests).
S15.1 Shared extensions (code + tests; allowed before the freeze):
  a) engines: class HybridEngine with cfg lineshape in {"lorentz","rabi"} and noise in {"gauss","binom"}. Measurement value m = lineshape(delta) + noise.
     lorentz+gauss = the CLASSICAL engine: replicate exactly the amplitude model and noise of the classical GAUSSIAN resonator module as read in Stage 1 Q7 (cite file:line in the docstring): amplitude Lorentzian with Q=50 about F_NOM (half-width F_NOM/(2Q)), additive Gaussian noise with the legacy noise floor; clip to [0,1].
     rabi+binom = the quantum surrogate rabi probe incl. readout error and shots. lorentz+binom: p = Lorentzian lineshape, observation = binomial(shots,p)/shots with readout error. rabi+gauss: p = exact Rabi lineshape, observation = p + the legacy Gaussian noise. Supported probes: rabi and dither (two measurements at +-dither_delta). Observation layout identical to the quantum rabi/dither layouts (so one policy runs in all four environments).
  b) Test: ClassicalEngine versus the legacy module's own measure function at 20 detunings from -9000 to +9000 Hz: mean and std over 2000 draws agree within 3% (or 0.002 absolute for the std). Test: (rabi,binom) reproduces the SurrogateEngine rabi counts distribution (KS p>0.01 at 12 detunings).
  c) controllers for the hybrid engines: PIDController and HillClimb work unchanged on the obs; BayesTracker gets a hybrid likelihood: for gauss noise use Gaussian(a | lineshape(grid - fc), sigma_a^2) with the same sigma_a as the engine, for binom use the binomial likelihood as before.
S15.2 Training arms (probe rabi unless stated; PPO with the FROZEN hyperparameters, 2M steps; train env -> config name):
  E1 env LG (lorentz,gauss)=classical ; E2 env LB (lorentz,binom) ; E3 env RG (rabi,gauss) ; E4 env RB (rabi,binom)=quantum. 5 seeds (200-204) each, snapshots at 50000, 200000, 500000, 1000000 steps.  (20 runs)
  E5 probe dither: envs LG and RB, 5 seeds each.  (10 runs)
  E6 classical with randomisation: env LG with per-episode Q in U(30,80), noise sigma in U(0.01,0.05), drift sigma log-uniform 250-1500, drift kind in {gaussian, ou}; 5 seeds.  (5 runs)
  FT arms: initialise from E1 (LG) final policy and fine-tune on RB for 50000, 200000, 500000, 1000000 steps (init_from), 5 seeds; same from E6 for 200000 only.  (25 runs)
  ALG arms: algo sac and td3 on envs LG and RB, probe rabi, 500000 steps, 5 seeds each (20 runs) plus FT from LG-trained sac/td3 for 200000 steps on RB (10 runs).
  Quantum-from-scratch matched budgets come from the E4 snapshots (50k/200k/500k/1M). PPO at 500k snapshot is the reference for the SAC/TD3 comparison.
S15.3 Evaluation (sealed base 920000, paired): every policy above AND every baseline (noctrl, hill, pid, bayes:nominal, bayes:oracle, oracle) on (i) all four environments at nominal, 1000 episodes; (ii) the quantum surrogate OFAT stress cells (300 episodes) for P2; (iii) the Aer engine (100 episodes, nominal and the 6 Aer cells of Part 1 12.3) for zero-shot and FT arms with seeds 200-202.
  Also evaluate the legacy-transfer reference: the legacy pipeline results already measured (zero-shot 17.5 kHz / 7%, fine-tuned V5 ~34%) are quoted from the database with their provenance, not recomputed.
S15.4 Transfer analyses (generated by scripts, stored in the database):
  a) transfer matrix: rows = training env (E1-E4), columns = evaluation env, entries lock_w1000 mean and CI over seeds; transfer gap G = performance in training env minus performance in the quantum env, with CI.
  b) gap decomposition: effect of lineshape (LG->RG, LB->RB) and effect of noise (LG->LB, RG->RB) on quantum performance; interaction term.
  c) sample efficiency: lock (P1 and P2) versus total quantum-env steps for (i) quantum from scratch, (ii) classical pretrain + FT; classical pretraining steps (2M) are reported separately, plus the break-even analysis in wall-clock and in quantum-env steps (Aer cost is much higher than surrogate cost: use the measured steps/s).
  d) classical-quantum environment correlation:
     (i) policy level: over all evaluated policies and baselines, Spearman and Kendall correlation (with bootstrap 95% CI over policies) between lock in LG and lock in RB (nominal), and between P2 in the two env families; scatter with identity line.
     (ii) condition level: for each method, correlation across stress cells between classical and quantum lock.
     (iii) measurement level: at 37 detunings between -9000 and 9000 Hz: KS statistic and Wasserstein-1 distance between classical amplitude readings and quantum p-hat readings (shots 64, 1024), lineshape RMS difference and FWHM ratio.
     (iv) trajectory level: for the same policy and the same episode seeds, RMS difference of e traces and the fraction of steps in the same lock state (S18) between environments.
  e) failure and lock-state analysis of all arms by S18.
S15.5 Pros and cons document docs\transfer_pros_cons.md: a table with rows: sample efficiency; asymptotic performance; wall-clock cost; simulator cost; robustness to unseen noise; sensitivity to lineshape mismatch; sensitivity to noise-model mismatch; risk of negative transfer; dependence on fidelity of the classical model; reproducibility/variance across seeds; algorithm family differences (PPO, SAC, TD3). Each cell: the finding, the table/figure id, and the verdict SUPPORTED / NOT SUPPORTED / NOT TESTED. No claim without evidence.
S15.6 db ingest; DOC docs\study_15_transfer.md (Appendix C template); docs_check PASS; state.py set S15 DONE.

==========================================================================================================
STAGE S16 - QUBIT-PLATFORM STUDY (all algorithms, realistic noise structure, dimensionless scaling)
==========================================================================================================
Principle: the tracking task depends on dimensionless numbers (drift per step times Ramsey time, tau/T2, readout error, shot noise, drift spectrum) and on the noise STRUCTURE (telegraph, 1/f, line pickup, random walk, decay shape, readout asymmetry). Each platform is a physical profile mapped to simulation units. This is a parameterised regime study, NOT a device-accurate simulation of each platform; say so in every document.
S16.1 Write STUDY\qphysics_ext.py and STUDY\tests\test_platform_ext.py = Appendix F VERBATIM. Run the test as a job on the aer lane: it must print "defaults identical: OK", "AER Z-FLIP DEPHASING EQUIVALENCE PASS", "PROFILE MAPPING PASS".
S16.2 BatchEnv/engine hooks (extend; defaults unchanged; all earlier tests must still pass): cfg keys contrast_power (default 1.0), contrast_override (default None), drift_params (dict; default {}), profile (string, informational).
  - Every Ramsey probability in BatchEnv, SurrogateEngine, BayesTracker (assumed or oracle) and the filter features must go through ramsey_p_ext with these keys. A test asserts identical results to the old path at the defaults.
  - Drift: use DriftExt(kind, rng, sigma, **drift_params) for every drift kind (identical to Drift at defaults).
  - AerEngine: for contrast C (any shape) apply a Z-flip channel pauli_error([("Z",p),("I",1-p)]) with p = (1-C)/2 on the id gate instead of thermal relaxation whenever contrast_power != 1 or contrast_override is set (the validated equivalence of Appendix F); keep thermal relaxation otherwise. Readout error as before.
  - The drift in a platform cfg is already expressed per step (profile_to_cfg handles the physical step time T_step = shots x shot_time); with fixed shots per run nothing else is needed.
S16.3 Profiles (Appendix F, dict PROFILES) and mapping profile_to_cfg(name, shots, drift_mult). Record in the database table platform_profiles every field with its tag G (grounded in published order of magnitude), A (assumed modelling choice) or S (speculative); mark every numeric value [CITATION NEEDED]; the Majorana-class profile is HYPOTHETICAL and is excluded from every "realistic" summary row (shown in a separate clearly labelled table).
  The photonic profile uses the interferometer visibility as a fixed contrast and maps phase drift to simulation detuning through phase = 2*pi*delta*TAU_NOM.
S16.4 Tuning (tune base 100000): for every profile and every probe in {rabi, dither, ramsey_iq} tune PID (Part 1 grid) and HillClimb on that profile at its default shots and drift_mult=1 and at drift_mult=100; store per-profile gains.
S16.5 Methods evaluated on every profile (sealed base 950000, paired; 200 episodes per cell):
  noctrl; hill (rabi); pid (per probe, profile-tuned); bayes:nominal (assumes the NOMINAL parameters of Part 1 PREREG, i.e. mismatched); bayes:oracle (knows the profile parameters); oracle (reference); RL zero-shot: the Part-1 confirmatory models C1-C6 (seeds 100-102);
  RL specialists: for each profile, the best plain probe config trained on that profile with drift_mult sampled log-uniform in [1,1000] and default shots (seeds 300-302; 7 profiles x 3 = 21 runs); RL generalist trained on the mixture of ALL profiles (profile drawn per episode, drift_mult log-uniform 1-1000; seeds 310-312);
  leave-one-platform-out: for each profile p a generalist trained on the other six (seeds 320-321; 14 runs), evaluated on p. Config names rl__<probe>__plat_<profile>, rl__<probe>__gen_all, rl__<probe>__lopo_<profile>.
  Grid per profile and method: drift_mult in {1, 10, 100, 1000, 10000} x shots in {64, 256, 1024, 4096} (physical-time coupling on: step time = shots x shot_time, so more shots mean more drift per step). The probe set for baselines: all three probes; RL uses the probe of its config.
S16.6 Endpoints (PREREG2): critical multiplier m*(method,profile) = largest drift_mult in the grid with lock_w1000 >= 0.90 at the profile's default shots (0 if none); lock_w1000 at drift_mult 1 and 100. Descriptive: Kendall coefficient of concordance W of method rankings across profiles; per-profile ranking tables.
S16.7 Sensitivity analysis (exploratory): for the top 4 methods on each GROUNDED/ASSUMED profile, multiply each A-tagged parameter (t2, e01, e10, drift D or std, jump rate/amplitude, shot_time) by 1/3 and by 3 one at a time (surrogate, 100 episodes); report whether method ranking changes (Kendall tau between baseline ranking and perturbed ranking).
S16.8 Fidelity ladder: for every profile at default shots and drift_mult in {1,100}: evaluate the same methods on the surrogate and on Aer (100 episodes, seeds as above, aer lane) and report ranking agreement (Kendall tau) and absolute differences.
S16.9 db ingest; DOC docs\study_16_platforms.md; docs_check PASS; state.py set S16 DONE.

==========================================================================================================
STAGE S17 - ADAPTIVE SENSING, CHANGE-POINT DETECTION, CALIBRATION, RISK, FALLBACK, ENVELOPE, GENERALISATION
==========================================================================================================
S17.1 ADAPTIVE SENSING (hypothesis: choosing how to measure improves reliability per measurement spent).
  Environment probe "adaptive" in BatchEnv with physical_time=True and drift gaussian sigma_base 500 Hz at 1024 shots; step drift sigma = sigma_base*sqrt(N/1024) (more shots cost time = more drift).
  Action Box(3): a0 control; a1 mode: rabi if a1<-1/3, dither if -1/3<=a1<1/3, ramsey_iq if a1>=1/3; a2 shots: 64 if a2<-1/3, 256 if -1/3<=a2<1/3, 1024 if a2>=1/3. Hard bins, documented.
  Observation (fixed layout): [fc/25000, prev a0, onehot(prev mode) (3), log2(N_prev)/10, (mu-fc)/5000, sigma_f/5000, f1, f2, f3] where an always-on GridBayes filter (assumed nominal parameters, probe model = chosen mode and shots) gives mu, sigma_f, and f1..f3 are the raw features of the last measurement in a unified 3-slot layout (rabi: [2p-1,0,0]; dither: [2p+ -1, 2p- -1, 2(p+ - p-)]; ramsey: [2pI-1, 2pQ-1, phase/pi]).
  Reward: base reward - lambda*N/1024 with lambda in {0, 0.02, 0.05, 0.1, 0.2}. One policy per lambda, seeds 400-404 (25 runs). The result is a Pareto set (lock_w1000 versus mean shots per step), never a single weighted score.
  Baselines at matched budgets: for each fixed (mode, N) with N in {64,128,256,512,1024}: pid and bayes:nominal (and bayes:oracle) in the same physical-time environment; the fixed-policy set forms the "fixed" Pareto curve. Information rule (pre-registered): at each step choose the cheapest (mode,N) among the 9 candidates such that the predicted posterior std sqrt(1/(1/sigma_f^2 + FI)) <= 350 Hz, where FI is the Fisher information of that candidate at the posterior mean (fisher_* functions); if none qualifies choose the maximum-information candidate.
  Report: lock_w1000 versus mean shots (nominal and stress-macro), usage of modes/shots versus posterior uncertainty (histograms and time series), and the area under the Pareto curve over mean shots in [64,1024] (log scale).
S17.2 REGIME SWITCHING AND CHANGE-POINT DETECTION. Drift kind "regime": hidden Markov regimes slow (gaussian sigma 250) and fast (gaussian sigma 1000), switch probability 0.02 per step; plus jump events with probability 0.01 per step, size uniform 2000-8000 Hz, random sign, applied instantly to x. Held-out evaluation family: jump size 8000-12000 Hz and probability 0.02.
  Detector (pre-registered): CUSUM on the standardised innovation z_t = (observed count - predicted count)/sqrt(predicted variance) from the filter's posterior predictive; S_t = max(0, S_{t-1} + |z_t| - 0.5); alarm when S_t > 5; on alarm inflate the filter's drift sigma by 10 for 5 steps and reset S. Attach to bayes:nominal (with/without detector) and to the RL hybrid (filter features with detector inflation).
  Metrics per jump: detection within 10 steps, detection delay, false alarms per 100 steps (alarms with no jump in the previous 10 steps), recovery time (steps until 3 consecutive |e|<=W), maximum |e| after the jump, measurement cost during recovery. Report shot noise honestly: false-alarm rate must be reported next to detection rate.
S17.3 UNCERTAINTY CALIBRATION of the Bayes filter (observability): for 50% and 90% credible intervals of the windowed posterior: empirical coverage, mean width, log score and NLL over episodes in the nominal and 10 stress cells, per probe. Also state the Fisher-information floor (CRB) versus achieved rms for each probe and shot budget.
S17.4 RISK METRICS: already stored (O5): report CVaR95, p95, lock-loss probability, recovery-time distributions for every method in every study.
S17.5 FALLBACK CONTROLLER (exploratory, thresholds fixed here): wrapper for bayes and RL-hybrid: if posterior sigma_f > 700 Hz or the CUSUM alarm is active then action scale x0.5 and (adaptive env only) force mode ramsey_iq with 1024 shots; exit when sigma_f < 350 Hz for 3 consecutive steps. Compare on and off in the regime env and the nominal env; report failure severity (CVaR, p95) versus average performance (lock mean).
S17.6 DRIFT-SPEED x DELAY ENVELOPE: drift sinusoid with amplitude A in {500,1000,2000,4000} Hz, period P in {8,16,32,64,128} steps (random phase, small OU jitter as in the sinusoid kind), action delay in {0,1,2,3}, probe ramsey_iq and dither. Methods: C1-C6 seeds 100-102, pid, bayes:nominal, bayes:oracle. Find per method the boundary where lock_w1000 falls below 0.90; compare with the slew-rate bound 2*pi*A/P versus action_scale (3000 Hz per step) and the delay-induced error A*2*pi*d/P.
S17.7 GENERALISATION MATRIX: DR-restricted training (drift kinds gaussian and ou only, all other DR ranges as DR; best probe; seeds 410-414) versus the full-DR models of Part 1; evaluate on held-out dynamics (onef, telegraph, sinusoid), held-out parameters (the DR-OUT cells of O3), observation shifts (shots 16384 and 64 with readout 0.10) and combined shifts (OUT sigma AND OUT readout). Report each shift separately.
S17.8 REWARD-MISMATCH CHECK: best probe, 3 seeds each (420-422): reward R1 = base; R2 = threshold lock reward (+1 if |e|<=1000 Hz else 0, minus 0.01|a|). Evaluate both on ALL metrics (lock widths, rms, CVaR, lock-loss) and report cases where training reward and scientific metrics disagree.
S17.9 Sealed evaluation base 960000 for everything in S17; db ingest; DOC docs\study_17_adaptive_changepoint_envelope.md; docs_check PASS; state.py set S17 DONE.

==========================================================================================================
STAGE S18 - LOCK FAILURES AND LOCK-IN STATES (for every method, environment and platform)
==========================================================================================================
S18.1 LOCK-STATE MACHINE (per episode, applied to e over all steps 0-199; W = 1000 Hz or 0.05 cycles in platform units): state LOCKED if |e|<=W for at least 3 consecutive steps; state LOST if |e|>10000 Hz for at least 3 consecutive steps; otherwise TRANSIT. Output per episode: first-lock time, LOCKED fraction, number of lock-loss events (entries into LOST after having been LOCKED), re-lock time (steps from entering LOST to next LOCKED), sojourn-time lists, 3x3 transition counts.
  Aggregate per (method, environment variant, platform, condition): occupancy fractions, mean/median sojourn times, transition matrix, first-lock-time and re-lock-time distributions. Table lock_states in the database; apply to ALL sealed evaluations of Parts 1 and 2 (classical LG/LB/RG/RB, surrogate, Aer, all platforms) wherever traces exist (store traces for at least 50 episodes per cell; for all lock-lost episodes up to K).
S18.2 FAILURE TAXONOMY (rule-based; first matching rule wins; the onset step t0 = first step of a >=20-step run with |e|>10000 Hz after burn-in; inspect the 10 steps before t0):
  F1 INFORMATION DEFICIT: at t0 the Fisher information of the probe at the true detuning (shots as used) corresponds to a static std > 5000 Hz, OR the true detuning is outside the probe's unambiguous region (Ramsey: |delta| > 1/(2*tau); rabi: |delta| > 2*omega where the lineshape is below 0.02) -> label IF.
  F2 SUDDEN JUMP: a drift jump/flip with size > 3000 Hz occurred within the 5 steps before t0 -> label JUMP.
  F3 DELAY/TRACKING-RATE: |x(t0) - x(t0 - delay - 1)| > W while no jump -> label DELAY.
  F4 ESTIMATE ALIASING: Ramsey/dither estimate error > half the unambiguous range with adequate information (not F1) -> label ALIAS.
  F5 CONTROL INSTABILITY: action saturation (|a|>0.95) in more than 8 of the 10 steps before t0, or at least 5 sign changes of a in those 10 steps -> label CONTROL.
  F6 QUANTISATION: DAC LSB > W/2 -> label DAC.
  F7 MODEL MISMATCH: (filter-based methods only) the assumed drift sigma, T2, contrast or readout error differs from the truth by more than a factor 2 -> label MISMATCH.
  F8 otherwise -> label UNEXPLAINED. Never assign a cause by timing alone beyond these mechanical rules. Table failures(eval_id, episode, onset, label, evidence fields).
S18.3 Questions to answer in docs\study_18_failures_lockstates.md: (a) WHY did the legacy single-probe Qiskit policy fail (apply the taxonomy to the legacy_diag.npz episodes using the legacy lineshape: F1 uses the legacy Fisher floor from Stage 5.2); (b) failure-label distribution per method and condition in the quantum surrogate, per platform, and for classical-trained policies evaluated in the quantum env versus in their own env; (c) lock-state occupancy and transition structure per method, environment and platform; (d) which failure labels disappear with dither/Ramsey probing, with the filter, with memory, with DR.
S18.4 db ingest; DOC; docs_check PASS; state.py set S18 DONE.

==========================================================================================================
STAGE S19 - TIME AND SPACE COMPLEXITY
==========================================================================================================
S19.1 Theory table docs\complexity_theory.md: per-decision time and memory in Big-O with symbols (G grid points, B batch, P policy parameters, D observation size, K stack length, H GRU hidden size, S shots simulated, E number of environments) for: NoControl, HillClimb, PID, Bayes grid filter (dense transition matrix: O(G^2) time per predict, O(G^2) memory; with the windowed estimate O(G); mention the O(G log G) FFT-convolution alternative as not implemented), RL MLP policy (O(P)), stacked MLP (O(K*D*h)), windowed GRU (O(K*H^2)), training cost PPO O(steps*P*epochs), SAC/TD3 O(steps*P) with replay memory O(buffer*D), surrogate engine O(B), Aer engine O(B*S-circuit cost), classical engine O(B).
S19.2 Measurements (script complexity.py, jobs on a single dedicated cpu lane while no other job runs on that lane, OMP_NUM_THREADS=1): per-decision latency (50th/95th/99th percentile in microseconds over 5000 decisions after 200 warm-up) for PID, Hill, Bayes with G in {101,201,401,601,1201}, MLP policy forward pass (torch, single thread), stacked MLP K in {4,8,16}, GRU K in {4,8,16}; environment throughput (steps/s) for surrogate, classical, hybrid, Aer (B in {1,10,100,1000}); peak resident memory of each; parameter counts and model-file sizes; training wall-clock, steps/s and peak RSS per config family from the train summaries.
S19.3 Fit scaling exponents (log-log least squares with 95% CI) for latency versus G, versus K, versus B and compare with the theory; real-time feasibility table: ratio of measured decision latency to the physical step time T_step of every platform profile (S16) at shots 64/1024.
S19.4 db table complexity; DOC docs\study_19_complexity.md (tables + figures); docs_check PASS; state.py set S19 DONE.

==========================================================================================================
STAGE S20 - INTEGRATION, DATABASE, FIGURES, PAPER PACKAGE (extends Part 1 Stage 13)
==========================================================================================================
S20.1 Database: add the tables of Appendix G to db.py (idempotent), ingest everything of Parts 1 and 2, extend docs\DB_README.md with 12 more example queries (e.g. transfer matrix, m* per method and platform, failure labels per platform, Pareto points, complexity fits) each executed with its result shown. reproduce.py gains a Part-2 smoke subset (profile mapping test, one hybrid-engine evaluation, one lock-state computation) and must print REPRODUCE OK.
S20.2 Figures F29-F58 (figures\ as in Part 1; PNG 300 dpi + PDF; INDEX.md captions with n/seeds/conditions): F29 transfer matrix heatmap (train env x eval env); F30 sample-efficiency curves (classical pretrain+FT versus quantum scratch, versus quantum steps and versus wall-clock); F31 transfer-gap decomposition (lineshape versus noise); F32 classical-quantum correlation (policy level scatter, condition level); F33 observation-distribution comparison (KS/Wasserstein versus detuning); F34 lineshape comparison (Lorentz, exact Rabi, legacy rx-angle); F35 PPO/SAC/TD3 comparison; F36 platform parameter map (sigma_n versus tau/T2 with profile points and tags); F37 per-platform tracking envelope (lock versus drift multiplier per method); F38 critical multiplier m* heatmap (method x platform); F39 ranking concordance across platforms; F40 specialist versus generalist versus leave-one-out; F41 sensitivity of rankings to A-tagged parameters; F42 Pareto (lock versus mean shots) with fixed-policy curves and information rule; F43 mode/shot usage versus uncertainty; F44 change-point metrics (delay, false alarms, recovery); F45 CUSUM illustration; F46 calibration/coverage; F47 CVaR/tail-risk bars; F48 drift-speed x delay envelope heatmaps with slew bound; F49 generalisation matrix; F50 reward-mismatch check; F51 failure taxonomy stacked bars (per method, per platform); F52 lock-state occupancy and transition diagrams; F53 lock/re-lock duration distributions; F54 legacy failure analysis; F55 latency scaling (log-log) with theory; F56 memory/space; F57 training cost; F58 realism audit table (G/A/S tags and sensitivity).
S20.3 Update the paper package (docs\paper\): add sections: transfer study, platform study (with the explicit caveat paragraph: parameter regimes, not device-accurate; Majorana-class is hypothetical), adaptive sensing and change-point, failure/lock-state analysis, complexity; extend claims_register.csv (every claim: statement, supporting table/figure id, status SUPPORTED/NOT_SUPPORTED/EXPLORATORY); limitations.md must include: simulation-only; privileged training reward; noise-free Rabi pulse; single qubit; synthetic drift; platform parameters assumed/not validated; Aer is not hardware; classical engine replicates the project's classical resonator model only. Write docs\paper\related_work_matrix_template.csv with the columns: paper, physical system, parameters calibrated, measurement method and budget, drift model, controller/estimator, metrics and baselines, validation level, code/data available, limitations (EMPTY rows: the controller fills it; you never invent citations).
S20.4 Then run Part 1 Stage 13 remaining items and Stage 14 (final verification). Extend the FINAL report (Stage 14.3) with sections for S15-S19 following the same pattern (tables, verdicts SUPPORTED/NOT SUPPORTED/MIXED, anomalies, BLOCKED list). The final manifest diff must still show only additions under qstudy\.
S20.5 End with exactly: QSTUDY PART 2 COMPLETE - ORIGINAL PROJECT UNCHANGED - AWAITING REVIEW.

==========================================================================================================
APPENDIX E - TEXT OF STUDY\PREREG2.md (copy VERBATIM at S15.0)
==========================================================================================================
=== FILE: STUDY\PREREG2.md ===
# Pre-registration 2 (new studies S15-S19; written before any new-study training or sealed evaluation)
Inherits all definitions of PREREG.md (error at probe time, burn-in 20 of 200 steps, lock fraction L(W), primary W=1000 Hz, nominal condition, oracle ceiling 0.954, paired seeds, Holm per family).
## Platform-normalised units
All platform parameters are mapped to simulation units with tau_sim = 50 us; frequencies scale by k = tau_platform/50us. Lock width W = 1000 Hz in simulation units corresponds to 0.05 cycles of the Ramsey time (platform-independent).
## Lock states and failure taxonomy
As specified in S18 (state machine: LOCKED >=3 steps within W; LOST >=3 steps beyond 10000 Hz; failure onset = first >=20-step run beyond 10000 Hz; rule order F1-F8). Thresholds are fixed and not tuned.
## S15 Transfer study
Arms and seeds: environments LG, LB, RG, RB (rabi probe, 5 seeds 200-204); dither LG and RB; classical-randomised LG-DR; fine-tune budgets 50k/200k/500k/1M; PPO/SAC/TD3. Sealed base 920000.
Endpoints: P1T = lock_w1000 in the quantum surrogate nominal condition (1000 episodes); P2T = macro-average lock_w1000 over the OFAT cells applicable to the probe (300 episodes each). Transfer gap = P1 in the training environment minus P1 in the quantum environment.
Confirmatory contrasts (Holm over 5): TR-1 LG zero-shot vs RB-from-scratch (2M steps) on P2T. TR-2 LG+FT 200k vs RB 200k snapshot on P2T. TR-3 LG-DR zero-shot vs LG zero-shot on P2T. TR-4 LB zero-shot vs LG zero-shot on P2T. TR-5 SAC(500k) vs PPO(500k snapshot) on RB-trained P2T.
Statistics: seed-level Mann-Whitney U (exact, two-sided), Cliff's delta, bootstrap 95% CI of the mean difference (10000 resamples of seeds). Everything else is exploratory (including all correlation analyses).
## S16 Platform study
Profiles in qphysics_ext.PROFILES (tags G/A/S). Grid: drift multiplier {1,10,100,1000,10000} x shots {64,256,1024,4096}. Sealed base 950000, 200 episodes per cell.
Endpoints: m*(method,profile) at default shots; lock_w1000 at multiplier 1 and 100. Confirmatory contrasts over the 6 non-speculative profiles (Holm over 4): PL-1 m* of RL zero-shot (best plain config) vs PID (paired by profile, Wilcoxon signed-rank). PL-2 generalist vs per-profile specialists, non-inferiority margin 0.02 on lock_w1000 at multiplier 100 (bootstrap CI over profiles and seeds). PL-3 m* of bayes:nominal vs bayes:oracle (paired by profile, Wilcoxon). PL-4 leave-one-platform-out generalist vs zero-shot best plain RL on the held-out profile (paired by profile, Wilcoxon).
The Majorana-class profile is hypothetical and excluded from all confirmatory contrasts. Sensitivity and fidelity-ladder analyses are exploratory.
## S17 Adaptive sensing and robustness
Adaptive environment as specified (physical-time coupling, lambda in {0,0.02,0.05,0.1,0.2}, seeds 400-404). Sealed base 960000. Confirmatory contrasts (Holm over 3): AD-1 adaptive-RL Pareto area vs best fixed-mode bayes:nominal Pareto area (bootstrap CI over seeds); AD-2 adaptive-RL vs information-rule policy (same area measure); CP-1 recovery time after jumps, bayes with CUSUM vs without (paired by episode, Wilcoxon, 1000 episodes with at least one jump). Acceptance for CP-1 validity: false-alarm rate <= 0.02 per 100 steps.
Everything else in S17 (calibration, fallback, envelope, generalisation, reward-mismatch) is exploratory.
## S19 Complexity: descriptive, no hypothesis tests.
## Rules
Negative or null results are reported; no run, seed or condition is dropped; sealed evaluations run once; thresholds in this file are never changed after freezing.
=== END FILE ===

==========================================================================================================
APPENDIX F - qphysics_ext.py and test_platform_ext.py (verified against Qiskit Aer 0.17; write VERBATIM)
==========================================================================================================
=== FILE: STUDY\qphysics_ext.py ===
"""Extensions for the platform study. Does NOT modify qphysics.py; defaults reproduce qphysics exactly."""
import numpy as np
from qphysics import F_NOM, CLIP, Drift, ramsey_p, readout

TAU_NOM = 50e-6   # simulation Ramsey time; all platform parameters are mapped to this by dimensionless products (delta*tau)

# ------------------------------------------------------------------ Ramsey contrast with shape and override
def contrast_ext(tau, t2, power=1.0, override=None):
    """C = exp(-(tau/T2)^power); power=1 exponential, power=2 Gaussian (quasi-static noise). override: fixed visibility (photonic)."""
    if override is not None:
        return np.asarray(override, dtype=float) * np.ones(np.broadcast(np.asarray(tau), np.asarray(t2 if t2 is not None else 1.0)).shape)
    t2a = np.inf if t2 is None else np.asarray(t2, dtype=float)
    return np.exp(-(tau / t2a) ** power)

def ramsey_p_ext(delta, tau=50e-6, t2=200e-6, quad="I", power=1.0, override=None):
    c = contrast_ext(tau, t2, power, override)
    phi = 2 * np.pi * np.asarray(delta, dtype=float) * tau
    return 0.5 * (1.0 + c * (np.cos(phi) if quad == "I" else np.sin(phi)))

def dephasing_flip_prob(c):
    """Aer: a Z-flip channel with probability p=(1-C)/2 on the wait gives Ramsey contrast C for ANY decay shape at fixed tau."""
    return (1.0 - np.asarray(c, dtype=float)) / 2.0

# ------------------------------------------------------------------ drift with explicit parameters
class DriftExt(Drift):
    """Like Drift but with: telegraph flip probability, sinusoid period range (steps), onef corner coefficients and total std."""
    def __init__(self, kind, rng, sigma=500.0, theta=0.05, amp=3000.0, flip=0.02, period_range=(60, 120), theta_list=None, std_total=None):
        super().__init__(kind, rng, sigma, theta, amp)
        self.flip, self.period_range = float(flip), period_range
        self.theta_list = None if theta_list is None else np.asarray(theta_list, float)
        self.std_total = std_total

    def reset(self, init_offset):
        if self.kind == "onef" and self.theta_list is not None:
            r = self.rng
            self.t, self.x = 0, float(init_offset)
            K = len(self.theta_list)
            v = (self.std_total ** 2) / K
            self.THETA = self.theta_list
            self.s = np.sqrt(v * (2 * self.theta_list - self.theta_list ** 2))
            self.comp = r.normal(0.0, np.sqrt(v), K)
            self.comp += (self.x - self.comp.sum()) / K
            return self.x
        x = super().reset(init_offset)
        if self.kind == "sinusoid":
            self.period = self.rng.uniform(*self.period_range)
        return x

    def step(self):
        r, k = self.rng, self.kind
        if k == "onef" and self.theta_list is not None:
            self.comp += -self.THETA * self.comp + self.s * r.normal(size=self.comp.shape)
            self.x = float(np.clip(self.comp.sum(), -CLIP, CLIP))
            return self.x
        if k == "telegraph":
            if r.random() < self.flip:
                self.state = -self.state
            self.y += -0.2 * self.y + 0.5 * self.sigma * r.normal()
            self.x = float(np.clip(self.base + self.amp * self.state * 0.5 + self.y, -CLIP, CLIP))
            return self.x
        return super().step()

# ------------------------------------------------------------------ platform profiles (physical -> simulation units)
# tags: G = grounded in published ranges (order of magnitude), A = assumed modelling choice, S = speculative (no demonstrated qubit)
PROFILES = {
 "sc_transmon":   dict(label="Superconducting transmon (fixed-frequency, TLS-dominated)", tags="G/A", tau=20e-6, t2=60e-6, power=1.0, override=None, e01=0.01, e10=0.04,
                       shot_time=300e-6, default_shots=1024, drift="telegraph", D=300.0, jump_hz=8000.0, jump_rate=1/600.0),
 "sc_flux":       dict(label="Superconducting tunable transmon (1/f flux noise)", tags="G/A", tau=10e-6, t2=20e-6, power=2.0, override=None, e01=0.01, e10=0.04,
                       shot_time=200e-6, default_shots=1024, drift="onef", std_total=11000.0, corner_s=(10.0, 100.0, 1000.0, 10000.0)),
 "si_spin":       dict(label="Electron spin qubit in silicon quantum dot (charge/nuclear noise)", tags="G/A", tau=3e-6, t2=20e-6, power=2.0, override=None, e01=0.03, e10=0.08,
                       shot_time=400e-6, default_shots=1024, drift="onef", std_total=11000.0, corner_s=(1.0, 10.0, 100.0, 1000.0, 10000.0)),
 "ion_hyperfine": dict(label="Trapped-ion hyperfine qubit (magnetic-field noise)", tags="G/A", tau=3e-3, t2=30e-3, power=2.0, override=None, e01=0.002, e10=0.002,
                       shot_time=5e-3, default_shots=256, drift="sinusoid", D=0.1, line_hz=3.0, line_period_s=600.0),
 "atom_tweezer":  dict(label="Neutral-atom tweezer qubit (slow light-shift/field drift)", tags="A", tau=500e-6, t2=5e-3, power=2.0, override=None, e01=0.005, e10=0.03,
                       shot_time=100e-3, default_shots=64, drift="gaussian", D=1.0),
 "photonic_mzi":  dict(label="Photonic interferometer phase lock (visibility-limited)", tags="G/A", tau=None, t2=None, power=1.0, override=0.97, e01=0.01, e10=0.01,
                       shot_time=10e-6, default_shots=1024, drift="gaussian", D_phase=0.01),
 "majorana_hyp":  dict(label="HYPOTHETICAL Majorana-class parity qubit (no demonstrated qubit; speculative)", tags="S", tau=1e-3, t2=1.0, power=1.0, override=None, e01=0.05, e10=0.05,
                       shot_time=10e-3, default_shots=256, drift="onef", std_total=50.0, corner_s=(10.0, 100.0, 1000.0, 10000.0)),
}

def profile_to_cfg(name, shots=None, drift_mult=1.0):
    """Physical profile -> BatchEnv cfg overrides in simulation units (tau fixed at TAU_NOM). Only products delta*tau and tau/T2 matter.
    Returns (cfg_overrides, info) ; info holds physical-time bookkeeping and normalised numbers."""
    p = PROFILES[name]
    N = int(shots or p["default_shots"])
    T_step = N * p["shot_time"]                                     # physical seconds per control step
    tau_p = p["tau"] if p["tau"] is not None else TAU_NOM
    k = tau_p / TAU_NOM                                             # sim_Hz = phys_Hz * k
    cfg = dict(shots=N, e01=p["e01"], e10=p["e10"], tau=TAU_NOM, contrast_power=p["power"], contrast_override=p["override"], profile=name)
    cfg["t2"] = None if p["t2"] is None else p["t2"] / k            # keeps tau/T2 identical
    kind = p["drift"]
    dp = {}
    if "D_phase" in p:                                              # photonic: rad^2/s -> sim Hz via phase = 2*pi*delta*TAU_NOM
        sigma_sim = np.sqrt(drift_mult * p["D_phase"] * T_step) / (2 * np.pi * TAU_NOM)
    elif "D" in p:
        sigma_sim = np.sqrt(drift_mult * p["D"] * T_step) * k
    else:
        sigma_sim = 0.0
    if kind == "telegraph":
        dp = dict(flip=1 - np.exp(-drift_mult * p["jump_rate"] * T_step), amp=p["jump_hz"] * k)
    elif kind == "sinusoid":
        period_steps = max(4.0, p["line_period_s"] / T_step)
        dp = dict(amp=p["line_hz"] * k * np.sqrt(drift_mult), period_range=(0.8 * period_steps, 1.2 * period_steps))
    elif kind == "onef":
        th = [1 - np.exp(-T_step / c) for c in p["corner_s"]]
        dp = dict(theta_list=th, std_total=p["std_total"] * k * np.sqrt(drift_mult))
        sigma_sim = max(sigma_sim, 1e-6)
    cfg.update(drift=kind, drift_sigma=float(sigma_sim), drift_params=dp)
    info = dict(T_step_s=T_step, k=k, sigma_n=float(sigma_sim * TAU_NOM), c=float(contrast_ext(TAU_NOM, cfg["t2"], p["power"], p["override"])),
                tau_over_t2=(None if p["t2"] is None else tau_p / p["t2"]))
    return cfg, info
=== END FILE ===

=== FILE: STUDY\tests\test_platform_ext.py ===
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("tests", 1)[0].rstrip("\\/"))
from qphysics import *
from qphysics_ext import *
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

# 1 defaults reproduce qphysics exactly
d = np.linspace(-9000, 9000, 37)
for q in "IQ":
    assert np.allclose(ramsey_p_ext(d, 50e-6, 200e-6, q), ramsey_p(d, 50e-6, 200e-6, q))
print("defaults identical: OK")
# 2 Gaussian / any-shape contrast realised in Aer by a Z-flip channel with p=(1-C)/2
SH = 200000
def run(qc, nm):
    qc.measure_all()
    r = AerSimulator(noise_model=nm).run(qc, shots=SH, seed_simulator=3).result().get_counts()
    return r.get("1", 0) / SH
for (tau, t2, power) in [(50e-6, 40e-6, 2.0), (50e-6, 80e-6, 1.5), (50e-6, 200e-6, 1.0)]:
    c = float(contrast_ext(tau, t2, power)); pz = float(dephasing_flip_prob(c))
    nm = NoiseModel(); nm.add_quantum_error(pauli_error([("Z", pz), ("I", 1 - pz)]), ["id"], [0])
    for dl in [-3000., 1000., 6000.]:
        for quad in "IQ":
            qc = QuantumCircuit(1); qc.rx(np.pi / 2, 0); qc.id(0); qc.rz(2 * np.pi * dl * tau, 0)
            (qc.rx if quad == "I" else qc.ry)(np.pi / 2, 0)
            pa, pe = run(qc, nm), float(ramsey_p_ext(dl, tau, t2, quad, power))
            assert abs(pa - pe) < 0.005, (tau, t2, power, dl, quad, pa, pe)
print("AER Z-FLIP DEPHASING EQUIVALENCE PASS")
# 3 profiles map to sane simulation parameters
for n in PROFILES:
    cfg, info = profile_to_cfg(n)
    assert cfg["shots"] > 0 and info["T_step_s"] > 0 and 0 < info["c"] <= 1.0, n
    rng = np.random.default_rng(0)
    dr = DriftExt(cfg["drift"], rng, sigma=cfg["drift_sigma"], **cfg["drift_params"]); dr.reset(0.0)
    xs = np.array([dr.step() for _ in range(500)])
    assert np.isfinite(xs).all() and np.abs(xs).max() <= CLIP, n
    for m in (1, 10, 100, 1000, 10000):
        c2, _ = profile_to_cfg(n, drift_mult=m)
        assert c2["drift_sigma"] >= cfg["drift_sigma"] - 1e-9 or c2["drift"] in ("onef", "telegraph", "sinusoid"), (n, m)
print("PROFILE MAPPING PASS")
=== END FILE ===

==========================================================================================================
APPENDIX G - DATABASE ADDITIONS (db.py, idempotent CREATE TABLE IF NOT EXISTS)
==========================================================================================================
platform_profiles(profile TEXT, field TEXT, value TEXT, tag TEXT, citation_status TEXT DEFAULT 'CITATION NEEDED');
env_variants(env_id TEXT PRIMARY KEY, lineshape TEXT, noise TEXT, engine TEXT, notes TEXT);
transfer(row_id TEXT PRIMARY KEY, train_env TEXT, eval_env TEXT, algo TEXT, arm TEXT, seed INT, ft_steps INT, lock_w1000 REAL, p2_macro REAL, eval_id TEXT);
correlations(corr_id TEXT PRIMARY KEY, level TEXT, x_name TEXT, y_name TEXT, n INT, spearman REAL, kendall REAL, ci_lo REAL, ci_hi REAL, notes TEXT);
lock_states(eval_id TEXT, method TEXT, env TEXT, profile TEXT, cond_id TEXT, p_locked REAL, p_transit REAL, p_lost REAL, t_first_lock_mean REAL, relock_mean REAL, losses_per_episode REAL, trans_json TEXT);
failures(eval_id TEXT, episode INT, onset INT, label TEXT, evidence_json TEXT);
platform_results(eval_id TEXT, profile TEXT, method TEXT, shots INT, drift_mult REAL, lock_w1000 REAL, rms REAL, cvar95 REAL, m_star REAL);
adaptive_results(eval_id TEXT, policy TEXT, lambda REAL, fixed_mode TEXT, fixed_shots INT, mean_shots REAL, lock_w1000 REAL, pareto_area REAL);
changepoint(eval_id TEXT, method TEXT, detector INT, detect_rate REAL, delay_mean REAL, false_alarm_per100 REAL, recovery_mean REAL, max_err_after_jump REAL);
calibration(eval_id TEXT, probe TEXT, level REAL, coverage REAL, mean_width REAL, log_score REAL);
complexity(item TEXT, param TEXT, value REAL, unit TEXT, p50 REAL, p95 REAL, p99 REAL, peak_rss_mb REAL, notes TEXT);
realism_audit(item TEXT, tag TEXT, sensitivity_rank_change REAL, notes TEXT);
==========================================================================================================
END OF PART 2
==========================================================================================================