QSTUDY CONTINUATION 3 - LEAN MODE   (executor: GLM 5.3)
THIS_FILE3 = path given in the launcher message.
PART1 = QSTUDY_PROMPT.md, PART2 = QSTUDY_PROMPT_PART2.md (same folder as THIS_FILE3). They remain AUTHORITATIVE (rules, stages, appendices, Part-2 overrides O1-O7, PREREG texts). DO NOT read them in full.
You continue an almost-finished build-up phase. Be economical: spend tokens on executing the project, not on re-reading history.

==============================================================================================
SECTION L0 - TOKEN DISCIPLINE (obey strictly)
==============================================================================================
K1 NEVER read the chat exports in the exports folder (PARENT\GLM Chat Exports) in full. Only if RESUME_CARD.md is missing or contradicts STATE.json: read the LAST 250 lines of the newest export (the file named like "qstudy-prompt-execution-and-state-recovery") and search it for the words FREEZE, APPROVED, BLOCKED, ANOMALY.
K2 NEVER read PART1/PART2 in full. Read one section at a time with the helper:  python WORK\tools\secget.py <file> --list   (heading index)   and   python WORK\tools\secget.py <file> "<regex of heading>" [maxlines]   (e.g. "STAGE S17" 150). Read only the sections the current step needs; do not re-read a section you already acted on.
K3 NEVER re-read project code, project documents, images or old results. Use WORK\understanding\, STUDY\docs\01_project_understanding.md only when a specific fact is needed, and use database queries (STUDY\db\qstudy.sqlite) instead of opening result files.
K4 Print at most 40 lines per command. Write long output to a file and print the first 20 lines. Never print whole job tables, whole logs or whole documents.
K5 WAITING: when nothing else can be done, use ONE command per poll:
      python WORK\tools\brief.py WORK && powershell -NoProfile -Command "Start-Sleep -Seconds 270"
   It prints one line. Print nothing else. Act only if NEW_FAILED is not "-" (read that job's last 15 log lines, apply X1-X4 of Part 1) or STALE_LANES is not "-" (relaunch that lane runner and the watchdog as in Part 1 Stage 0.7). Print the full job table (jobtools.py WORK -v | head -40) only every 40th poll.
K6 Do not re-verify steps that STATE.json marks DONE unless an anomaly indicator (NEW_FAILED, failing test, inconsistent count) points at them.
K7 No narration, no recaps, no summaries unless a stage ends (one STAGE line) or an explicit stop gate is reached.
K8 Batch work: submit many jobs with one JSON file; generate documents and tables with scripts from the database; write modules in pieces and do not reprint them.
K9 Update WORK\RESUME_CARD.md (max 100 lines, overwrite not append) after every milestone and at least every 30 minutes of wall-clock when working. A fresh session must be able to continue from the card alone.

==============================================================================================
SECTION L1 - LEAN BOOT (about 8 small commands; do exactly this)
==============================================================================================
1. Print your working directory (it must end in MiniProject_Sem4). Derive PARENT, NAME, WORK = PARENT\<NAME>_QSTUDY_WORK, STUDY = PROJECT_ROOT\qstudy. Read WORK\00_paths.txt. If WORK does not exist: STOP "WORK NOT FOUND - AWAITING INSTRUCTIONS".
2. Write the two helper scripts of Appendix H into WORK\tools\ verbatim (secget.py, brief.py).
3. If WORK\RESUME_CARD.md does not exist, create it from the INITIAL RESUME CARD in Appendix I (verbatim). Otherwise read it (it is the newest truth).
4. python WORK\tools\state.py WORK show | tail -25 ; python WORK\tools\brief.py WORK. If any lane is stale: relaunch it (and the watchdog) as in Part 1 Stage 0.7. If jobs are RUNNING with dead processes, the runner re-queues them by itself after restart.
5. python -c "import json;d=json.load(open(r'<WORK>\STATE.json'));print(str({k:v for k,v in d.items() if k!='steps'})[:800])"   (prints the saved resume note).
6. Verify the key invariants in ONE short script and print one line: no result file with split=sealed exists; no run directory for seeds 100-109 exists except the tuning runs; no gpu job has depends_on on another gpu job; unit-test suite file count; free disk space. If an invariant fails: fix per Part 1 X-rules and log it in events.md.
7. state.py set BOOT3 DONE. Continue with SECTION L2 at the first unfinished item.

==============================================================================================
SECTION L2 - REMAINING WORK (in this order; details live in the named sections; read only what you need)
==============================================================================================
W1 PART2 "STAGE S17": implement and test S17.2 (regime env + CUSUM change-point detector with the fixed thresholds, metrics per jump) and S17.5 (fallback wrapper with the fixed thresholds), plus the S17.3 calibration script and the S17.6 envelope/S17.7 generalisation/S17.8 reward-check evaluation drivers (code and smoke tests only; sealed runs wait for the freeze and PREREG2). Run the full unit-test suite after each module (job on cpu lane); it must stay green.
W2 PART2 "STAGE S19": write complexity.py and complexity_theory.md (theory table). The MEASUREMENTS must run on a quiet machine: schedule them for the very end (after all trainings and evaluations) and run them on one cpu lane while no other job runs.
W3 Exploratory documentation: check which of docs\approach_1_dither.md, approach_2_ramsey.md, approach_3_bayes_hybrid.md, approach_6_memory.md, approach_7_noise.md, approach_4_dr_curriculum.md, tuning.md exist and pass docs_check (list only the missing/failing ones). When the exploratory queue has drained and results are ingested (db.py ingest), write the missing documents with the Part 1 Appendix C template (generated tables only; read PART1 "APPENDIX C" once).
W4 PART1 "STAGE 11" step 11.8 as amended by Part 2 override O2: when ALL exploratory trainings, tuning runs and their val evaluations are DONE (failed ones recorded), compute the ranking tables and write configs\frozen\selection.json as a DRAFT. Print compactly (max 60 lines): validation score per config (mean, sd over seeds), the tuning table, the draft selection C1-C6 and the check results (no sealed files, no seeds 100-109 runs). Then print exactly: FREEZE-READY - AWAITING CONTROLLER APPROVAL, update the resume card, and STOP. Do not poll after this message.
W5 ONLY AFTER the controller writes "FREEZE APPROVED" in this chat (record the quote in the resume card): write the FREEZE ADDENDUM and PREREG.sha256 (Part 1 11.8); write PREREG2.md and PREREG2.sha256 (PART2 "APPENDIX E" verbatim; step S15.0); submit ALL remaining training jobs up front on the gpu lane (never use depends_on between gpu jobs): Part 1 Stage 12 confirmatory (seeds 100-109), then Part 2 S15, S16, S17 trainings; submit the evaluation jobs per model (depends only on that model's training job).
W6 Then, as models finish: sealed evaluations (cpu lanes, aer lane), then S18 analyses, S19 measurements, S15/S16/S17 documents, S20 integration, Part 1 Stage 13-14 and the final report (read each stage section with secget when you reach it). End with the exact completion lines of PART1 Stage 14.5 and PART2 S20.5.

==============================================================================================
SECTION L3 - ESSENTIAL RULES (full text in PART1 Sections 0-3 and PART2 P2; read the headings with secget only if in doubt)
==============================================================================================
- Never modify any pre-existing project file; new files only under qstudy\ and WORK. Original-repo git is read-only. No installs, no network.
- Every command you run directly finishes in under 60 s (K5 polling sleep excepted); anything longer is a job. Jobs are resumable; training/eval scripts print progress at least every 60 s.
- Never use depends_on between gpu-lane jobs. Failed jobs are recorded, not retried by hand unless the cause is fixed (X1-X4).
- Sealed evaluations only after the freeze hashes exist (PREREG.sha256 for the base study, PREREG2.sha256 for S15-S19); never repeat a sealed evaluation.
- Never invent numbers or citations; every number in a document comes from the database or a result file; docs_check must pass.
- Do not weaken a test to make it pass; fix the code (X4).
- The ONLY deliberate stops: FREEZE-READY (W4) and final completion. Otherwise never stop and never ask questions.

==============================================================================================
APPENDIX H - HELPER SCRIPTS (write VERBATIM into WORK\tools\)
==============================================================================================
=== FILE: secget.py ===
import re, sys
f, pat = sys.argv[1], sys.argv[2]
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 120
L = open(f, encoding="utf-8", errors="replace").read().splitlines()
H = re.compile(r"^(STAGE |SECTION |APPENDIX )")
if pat == "--list":
    for i, l in enumerate(L, 1):
        if H.match(l): print(i, l[:100])
    sys.exit(0)
rx = re.compile(pat)
for i, l in enumerate(L):
    if H.match(l) and rx.search(l):
        j = i + 1
        while j < len(L) and not H.match(L[j]): j += 1
        out = L[i:j]
        print("\n".join(out[:mx]))
        if len(out) > mx: print("... [%d more lines; rerun with a larger maxlines]" % (len(out) - mx))
        break
else:
    print("NOT FOUND")
=== END FILE ===

=== FILE: brief.py ===
import datetime, glob, json, os, sys
w = sys.argv[1]; q = os.path.join(w, "queue")
n = {}; run = []; failed = []
for p in glob.glob(os.path.join(q, "*.job.json")):
    sp = p.replace(".job.json", ".status.json")
    try:
        s = json.load(open(sp, encoding="utf-8"))
    except Exception:
        s = {"state": "PENDING"}
    st = s.get("state", "PENDING"); n[st] = n.get(st, 0) + 1
    jid = os.path.basename(p)[:-9]
    if st == "RUNNING": run.append(jid)
    if st == "FAILED": failed.append(jid)
mark = os.path.join(w, "brief_last_failed.json")
try:
    old = set(json.load(open(mark)))
except Exception:
    old = set()
new = sorted(set(failed) - old)
json.dump(sorted(failed), open(mark, "w"))
stale = []
for l in glob.glob(os.path.join(w, "heartbeat_*.txt")):
    age = datetime.datetime.now().timestamp() - os.path.getmtime(l)
    if age > 240: stale.append(os.path.basename(l)[10:-4] + ":%ds" % age)
print("JOBS", n, "| running:", (",".join(run))[:150], "| NEW_FAILED:", ((",".join(new))[:200] or "-"), "| STALE_LANES:", (",".join(stale) or "-"))
=== END FILE ===

==============================================================================================
APPENDIX I - INITIAL RESUME CARD (write to WORK\RESUME_CARD.md verbatim if it does not exist; afterwards you maintain it)
==============================================================================================
=== FILE: RESUME_CARD.md ===
# RESUME CARD (overwrite; max 100 lines)  - last handover state, before the lean session
## Authority
Controller prompts: QSTUDY_PROMPT.md (Part 1), QSTUDY_PROMPT_PART2.md (Part 2, overrides O1-O7), this continuation file (lean mode). Read sections only via WORK\tools\secget.py.
## State at handover
- Part 1 Stages 0-5 DONE. Stages 6-10: code+tests done, experiments queued/running. Stage 11.1-11.7 in progress (DR/curriculum, tuning, Aer fine-tunes, combos queued). BOOT (Part 2 P1) DONE.
- Part 2 overrides: O1 (no gpu->gpu deps) applied; O3 dr_support column done (68 IN / 43 OUT); O4 lineshape comparison table done; O5 evaluate.py risk metrics + --store_failures + PREREG2 sealed gate done (7 tests); O6 train_ppo.py (RSS_MB, snapshots, algo ppo|sac|td3, init_from) done (9 tests; TD3 train_freq=1 documented deviation); O2 NOT yet reached (no freeze, no sealed runs, 0 sealed files).
- Part-2 code built and tested pre-freeze: S15.1 HybridEngine (4 environments, legacy replication, KS tests); S16.1 qphysics_ext.py + test_platform_ext (pass); S16.2 BatchEnv/engine hooks (defaults identical); S17.1 adaptive environment; S18.1 lock-state machine + S18.2 failure taxonomy. Full unit-test suite: 100 tests, all OK.
- Anomalies recorded and recovered (A7/A8/A9): seed_3 corrupted artifacts renamed; DRcurr curriculum bug FIXED (retrains show the curriculum ramp); wiped tuning+smoke dirs being retrained (priority queue prefix s0z_); 19 retrains + 92 eval jobs submitted. Earlier anomalies A1-A6 are in STUDY\docs\03_session_handover.md.
- Queue at handover: 467 DONE / 163 PENDING / 1 RUNNING; all lanes healthy.
## Next actions (in order)
1. S17.2 CUSUM change-point + regime env; S17.5 fallback; S17.3 calibration script; S17.6/S17.7/S17.8 drivers (code + smoke tests only).
2. S19 complexity.py + theory table (measurements at the very end on a quiet machine).
3. Missing exploratory docs (approach_1/2/3/4/6/7, tuning) after ingest.
4. Stage 11.8 draft -> print compact tables -> FREEZE-READY - AWAITING CONTROLLER APPROVAL -> STOP.
## After approval
PREREG addendum + hashes; PREREG2 (S15.0); submit all training jobs (no gpu deps): Part 1 Stage 12 (seeds 100-109), Part 2 S15, S16, S17; evaluations; S18; S19 measurements; S20; Part 1 Stage 13-14; final report.
## Key facts
Seeds: exploratory 0-4; tuning 100 (8 runs); confirmatory 100-109; Part 2: transfer 200-204, specialists 300-302, generalist 310-312, LOPO 320-321, adaptive 400-404, DR-restricted 410-414, reward-check 420-422. Bases: tune 100000, val 500000, sealed 900000 (base), 920000 (transfer), 950000 (platform), 960000 (adaptive etc.).
Lanes: gpu, aer, cpu1..cpuK. Helpers: WORK\tools\{state,jobtools,manifest,job_runner,unchain,secget,brief}.py. Poll: brief.py && Start-Sleep 270.
## Controller approvals received
(none yet)
=== END FILE ===