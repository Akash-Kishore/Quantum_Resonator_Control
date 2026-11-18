QSTUDY AUDIT 6 - PROJECT AUDIT, SAFE CLEANUP, STRUCTURE FIX, THEN RESUME   (executor: GLM 5.3)
THIS_FILE6 = path given in the launcher message. Authority order: this file > CONTINUATION5 > CONTINUATION4 > CONTINUATION3 > PART2 > PART1.
Read it one section at a time:  python WORK\tools\secget.py <THIS_FILE6> "SECTION A0" 80
All rules of SECTION P0 of CONTINUATION5 stay in force (S1-S7, Q1-Q10). Section A0 only adds to them.

PURPOSE
The project grew over many sessions and one hard reboot. Before more work is stacked on it, the controller wants (1) a true report of the project state,
(2) removal of everything unnecessary from the working tree WITHOUT deleting anything, (3) a documented file structure, (4) fixes of confirmed defects that
are safe to fix. The z1 confirmatory run is IN PROGRESS and must not be disturbed: the lanes keep running, no confirmatory code or config may change.

==============================================================================================
SECTION A0 - ADDITIONAL RULES
==============================================================================================
N1  NOTHING IS DELETED. "Clean" means MOVE to quarantine: WORK\quarantine\<yyyymmdd>\ keeping the relative path, with one line per moved item in
    WORK\quarantine\<yyyymmdd>\MOVE_LOG.csv (src,dst,bytes,sha256,category,reason). tools\restore_quarantine.py <yyyymmdd> must be able to put everything back;
    write it and test it on 3 throwaway files BEFORE the first real move. The user may delete the quarantine later by hand.
N2  LANES KEEP RUNNING. Do not stop runners, watchdog, or any training. Phase A2-A3 are READ-ONLY (no moves, no edits, no renames, no new files except under
    WORK\audit\ and WORK\tools\).
N3  PROTECTED, never moved, edited, renamed or deleted, in any phase:
    (a) every path in the reference closure R (SECTION A1 step 4); (b) all of WORK\tools, WORK\logs, WORK\backup, WORK\queue and the job-state files;
    (c) STUDY\PREREG.md, PREREG.sha256, PREREG2.md, PREREG2.sha256, configs\frozen\*, everything under results\sealed, the database file(s) and their -wal/-shm;
    (d) all runs\ directories of seeds 0-4, 100, 101-110, 200-204 and every directory a PENDING/RUNNING/DONE z1* or z2* job writes to;
    (e) every file modified in the last 6 hours; (f) every prompt file in C:\Projects (list them, never move them: the launcher messages refer to their paths);
    (g) anything you cannot classify (category UNKNOWN is protected and only reported).
N4  CODE FREEZE FOR Z1. Pipeline code is every .py file in the import closure of the scripts named in the cmd of any job that is not FAILED (training, evaluate,
    eval_set, envs, engines, controllers, baselines, config loaders). During z1 these files are NOT edited. Defects found in them go to
    WORK\audit\post_z1_fixes.md (file, line, symptom, proposed fix, severity) and are fixed only after z1 sealed evaluation, on the controller's instruction.
N5  Fixes in this audit are limited to NON-pipeline files: documentation, WORK\audit scripts, README/INDEX files, tests that do not import changed code,
    non-referenced helper scripts. Each fix: backup first (Q2), py_compile, nearest test, full suite once at the end.
N6  Every number you put in the report must come from a command run in this session (tag VERIFIED with the command). Anything else is ASSUMPTION and is
    labelled so in the report. No guessing of causes: if the cause is unknown write "cause unknown" and list the evidence.

==============================================================================================
SECTION A1 - PREFLIGHT (no stop unless a check fails)
==============================================================================================
1. python WORK\tools\brief.py WORK. All six heartbeats younger than 60 s and STALE_LANES "-"; otherwise repeat P1 steps 6-7 for the stale lane first.
2. Freeze integrity: sha256 of PREREG.md and PREREG2.md against the .sha256 files. Mismatch: STOP "FREEZE TAMPERED - AWAITING CONTROLLER".
3. State the baseline in events.md: DONE/RUNNING/PENDING/FAILED counts, free disk on C:, last boot time, number of python processes.
4. Build the REFERENCE CLOSURE R with tools\ref_closure.py (read-only, output WORK\audit\R.txt, print only the counts per kind, max 12 lines):
   (i)   every path in cmd, cwd, expected_outputs, depends_on-model paths of every job whose state is not DONE (PENDING, RUNNING, FAILED, BLOCKED);
   (ii)  the python import closure (use the ast module, resolve imports inside STUDY only) of every script named in those cmds;
   (iii) every config file named by --cfg, init_from or a config-includes key of the above, transitively;
   (iv)  every path mentioned in PREREG.md, PREREG2.md and RESUME_CARD.md that exists on disk;
   (v)   all files of configs\frozen and all evalcfg__*.json files of the nine confirmatory models.
5. CODE FREEZE MANIFEST: write WORK\audit\CODE_FREEZE_Z1.sha256 with "sha256  relative_path" for every .py in R(ii). Print the count and the sha256 of the manifest file.
   This manifest is re-checked at every later step of this file; any difference means a pipeline file changed: STOP "CODE FREEZE BROKEN".

==============================================================================================
SECTION A2 - INVENTORY (read-only)
==============================================================================================
Write tools\inventory.py. It walks PROJECT_ROOT, STUDY and WORK (skip nothing, but do not follow links), and writes WORK\audit\inventory.csv with
path, bytes, mtime, category, protected_by (list of N3 letters or R-kind, empty if none), sha256 for files under 50 MB (skip hashing larger files and say so).
Categories (first match wins): PIPELINE_CODE (in R(ii)), TEST, CONFIG_FROZEN, CONFIG_EVAL, CONFIG_OTHER, RUN_CONFIRMATORY, RUN_EXPLORATORY, RESULT_SEALED,
RESULT_VAL_OR_EXPLORATORY, DATABASE, DOC, PROMPT, TOOL, LOG, BACKUP, CACHE (__pycache__, .pytest_cache, *.pyc, .ipynb_checkpoints, *.tmp, ~*, Thumbs.db),
SMOKE (runs_smoke, results_smoke, WORK\smoke, WORK\smoke_cfgs, p1_test and p3_ artefacts, p4_cal outputs), STALE_COPY (same sha256 as another file
outside the protected set and older, or names like *_old, *_copy, *.orig, *.bak outside WORK\backup), UNKNOWN.
Print (max 40 lines total): bytes and file count per category; top 10 directories by size; top 10 files by size; top 10 duplicate groups by wasted bytes.
Verify the inventory: total bytes in the CSV must be within 1 percent of the sum reported by an independent PowerShell measure of the same roots
(Get-ChildItem -Recurse -File | Measure-Object Length -Sum). Report both numbers. If they differ by more than 1 percent, find out why before continuing.

==============================================================================================
SECTION A3 - CORRECTNESS AUDIT (read-only) AND GATE G4
==============================================================================================
Write each check as a small script under WORK\tools\audit_*.py (<= 100 lines each), run it, store its output in WORK\audit\, and use the printed results only.
 C1 QUEUE: counts per state for each id group (z1a, z1b, z1e, z1v, z1x, p1/p3/p4 test jobs, legacy). List every job with attempts > 1 in scope z1*, p1*, p3*, p4* (max 15).
    List every FAILED or BLOCKED job outside the legacy groups (max 15). Legacy groups are only counted.
 C2 TRAINING HEALTH: for every finished confirmatory training: config, seed, final and best validation score, wall seconds, steps/s (table max 40 lines, or per-config
    summary if longer). Mark COLLAPSE_CANDIDATE when the final score is below 0.6 x the config median. Report only (Q7).
 C3 INTEGRITY: run tools\reboot_audit.py again (training outputs readable, DONE markers consistent with their files, no 0-byte or unparsable file in results\sealed).
 C4 CODE: py_compile on every .py in STUDY and WORK\tools (count OK/FAIL, list failures); full unit suite as a job on a cpu lane (count, skips). Pipeline code hashes against
    CODE_FREEZE_Z1.sha256 (must match).
 C5 DOCS VS FILES: compare the numbers stated in PREREG.md, PREREG2.md, RESUME_CARD.md and events.md (job counts 90/270/45/26, seeds, learning rate, entropy, gamma, steps, sealed
    seed bases, hashes) with what configs\frozen, the queue and the hash files really contain. List every mismatch (max 15): source line, stated value, actual value.
    Mismatches inside PREREG*.md are only REPORTED, never edited.
 C6 OPEN ITEMS: grep events.md for ANOMALY, anomaly, ASSUMPTION, TODO, BLOCKED, "unresolved". For every hit decide OPEN or CLOSED ONLY from evidence in events.md (a later line that says
    VERIFIED or fixed); otherwise OPEN. List the open ones (max 20) with id and one sentence.
 C7 CONFIG SANITY: every config in configs\frozen parses and has the keys the training script reads (print missing keys); every one of the nine models has its evalcfg; list config files
    that nothing references (names only, max 20).
 C8 STRUCTURE: list top-level directories of STUDY and WORK with a one-line purpose each IF the purpose is stated in a README or in a docstring; otherwise write "purpose not documented".
 C9 PROMPT FILES: list prompt files in C:\Projects with size and mtime (names only). Report which ones the current work refers to (CONTINUATION5, AUDIT6) and mark the others "historical".
 C10 CLEANUP CANDIDATES: from the inventory, the items in categories CACHE, SMOKE, STALE_COPY that are NOT protected (N3) and NOT in R. Print per category: item count, bytes, the 10 largest
    candidates. Items whose only reference is a DONE job's log are candidates; items referenced by a non-DONE job are never candidates.
Write the full report to WORK\audit\AUDIT_REPORT.md (max 150 lines): sections "Facts verified", "Findings", "Open items", "Cleanup candidates", "Proposed structure fixes".
Findings carry a severity: HIGH (could change a result or break a queued job), MEDIUM (wrong documentation or fragile tooling), LOW (cosmetic).
GATE G4  Print "GATE-G4 REPORT" (max 45 lines): (1) true state per workstream (z1 training seeds finished/total, evals done/total, baselines, Aer, tests), (2) findings by severity with
         counts and the HIGH ones in one line each, (3) open items count, (4) bytes protected versus bytes proposed for quarantine per category, (5) the proposed structure fixes (max 10 lines).
         STOP. Continue only when the user types:  GO-CLEAN

==============================================================================================
SECTION A4 - QUARANTINE (only after GO-CLEAN)
==============================================================================================
1. Write and test tools\restore_quarantine.py (N1) on 3 throwaway files under WORK\quarantine\selftest.
2. Move ONLY the categories CACHE, SMOKE and STALE_COPY from the approved candidate list of GATE G4, in batches of at most 200 items, writing MOVE_LOG.csv. Never touch N3 items; before each batch
   re-check that no candidate was modified in the last 6 hours and none appears in a freshly rebuilt R (re-run ref_closure.py before every batch; it reads the live queue).
3. After the last batch verify, each as a command with printed result: (a) every path in R still exists; (b) every non-DONE job's script, cfg and cwd exist; (c) CODE_FREEZE_Z1.sha256 matches;
   (d) PREREG hashes match; (e) brief.py shows all six heartbeats younger than 60 s and no new FAILED job; (f) full unit suite as a cpu job: same count as in C4; (g) one smoke eval cell
   (rl__dither__plain smoke model, dither nominal, 5 episodes, split val, output under WORK\smoke) writes its DONE marker. If the smoke model or smoke outputs were quarantined, regenerate them
   from the quarantine copy by running restore_quarantine.py for exactly those files.
4. If ANY of (a)-(g) fails: run restore_quarantine.py for the whole date, repeat the checks, print "CLEANUP ROLLED BACK" with the failing checks, and STOP.
5. Print bytes moved, items moved, and the free disk before and after (3 lines).

==============================================================================================
SECTION A5 - STRUCTURE AND DOCUMENT FIXES (non-pipeline only, N4-N5)
==============================================================================================
1. Write STUDY\README_STRUCTURE.md (max 80 lines): for each top-level directory of STUDY and WORK, its purpose (from the inventory and docstrings, VERIFIED), whether it is LOCKED, GENERATED or
   HAND-EDITED, and what may be cleaned. Add the rule "pipeline code is frozen during z1; see WORK\audit\CODE_FREEZE_Z1.sha256". Do not move or rename any directory or file in this step.
2. Write WORK\INDEX.md (max 60 lines): the lane and tool map (what each file in WORK\tools does in one line, taken from its docstring or first comment; "undocumented" where missing).
3. Fix every confirmed MEDIUM or LOW finding that sits in a non-pipeline file (typos in docs, wrong counts in RESUME_CARD or WORK\INDEX, stale paths in helper scripts, missing docstrings in
   tools). Each fix is one line in events.md: file, what, VERIFIED by.
4. Write WORK\audit\post_z1_fixes.md with every finding in pipeline files or in locked files (file, line, symptom, proposed fix, severity). Do NOT apply them.
5. Rewrite WORK\RESUME_CARD.md (max 100 lines) to the verified current state: queue counts, what is running, which approvals were received (GO-Z1, GO-CLEAN) and which are pending (GO-Z2A, GO-P6),
   the open items list, and the exact next actions. Keep the key-facts block (paths, lanes, seeds, frozen hyperparameters, poll command).
6. Re-check CODE_FREEZE_Z1.sha256 and the PREREG hashes once more (must match).

==============================================================================================
SECTION A6 - FINAL CHECK AND RESUME
==============================================================================================
Print "GATE-G5 REPORT" (max 30 lines): bytes and files before and after, items quarantined, checks A4(a)-(g) PASS/FAIL, fixes applied (count), findings left for after z1 (count), open items (count),
queue counts. If everything is PASS, do NOT stop: continue with the standing instructions of CONTINUATION5 as follows (the same ones the previous session was running):
 - poll loop (python WORK\tools\brief.py WORK && powershell -NoProfile -Command "Start-Sleep -Seconds 600"), act only on NEW_FAILED or STALE_LANES;
 - WP1 coding is AUTHORISED (design, tests, configs, gen_z2a.py, validate, smoke); submitting any z2a job still requires the user to type GO-Z2A after GATE G2;
 - WP2 only after the user types GO-P6;
 - when the last wave-A training job is DONE, print the P4.5 table and continue.
If any check is FAIL, print "AUDIT BLOCKED - AWAITING CONTROLLER" with the failing items and stop.

==============================================================================================
APPENDIX - LAUNCHER MESSAGE (for the user)
==============================================================================================
Read C:\Projects\QSTUDY_PROMPT_AUDIT6.md and execute it exactly. THIS_FILE6 = that file. Start at SECTION A1. Read SECTION A0 first. Do not read any other prompt file in full.
