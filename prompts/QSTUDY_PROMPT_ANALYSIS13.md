CONTROLLER PROMPT: ANALYSIS BLOCK (Z1 contrasts, Z2A check and analysis, WP2). Executor: GLM 5.3.
Do not read any QSTUDY prompt file in full. Read one section at a time with:
 python WORK\tools\secget.py <file> "<regex of heading>" [maxlines]
(works for headings starting STAGE/SECTION/APPENDIX; PREREG.md and PREREG2.md are
short: print them with Get-Content). Prompt files: C:\Projects\MiniProject_Sem4\prompts\
Every PowerShell command starts with: . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1";
RULES: never modify results\sealed; never edit pipeline code (train_ppo, evaluate, eval_set,
envs, engines, controllers, baselines); do not touch PREREG*, configs\frozen, or any
.sha256 file; new files only under qstudy\ and WORK; write files in pieces <= 80 lines;
print <= 30 lines per command; anything that runs > 60 s is a job on a cpu lane (first
run qs10_lanes.py ensure). After EACH phase append 3 lines to WORK\RESUME_CARD.md and
events.md. If you stop for any reason, a fresh session continues at the first phase whose
DONE line is missing in RESUME_CARD.md. No invented numbers: every number printed comes
from the database or a result file. Never weaken a test to make it pass.
Do not submit any training or evaluation job. Do not start S16/S17/S18/S19.

USER APPROVALS GIVEN IN THIS MESSAGE: (a) GO-UNBLIND scope Z1 FINAL contrasts;
(b) opening sealed Z2A results (base 920000) under PREREG2, scope TR-1/TR-2 and the
descriptive S15.4 analyses listed in PHASE C; (c) GO-P6 scope WP2 ONLY (tools
training_health.py and paired_stats.py). Nothing else is approved.

PHASE 0 (state): run brief.py. Expect DONE=1611, nothing RUNNING/FAILED. Read the last 40
lines of RESUME_CARD.md. The DB was backed up after the ingest (backup\post_ingest_*);
verify integrity_check = ok.

PHASE A: Z1 CONFIRMATORY CONTRASTS C-1..C-8 (preregistered; PREREG.md "Confirmatory
contrasts" and "Statistics" sections; sealed base 900000; seeds 101-110, n=10 per config).
A1 (read-only, max 30 lines): in db.evals seed_base 900000: rows per (method, seed);
   duplicate groups on the natural key (method, seed, cell, split); why some baselines
   have 111 rows vs ~102 for RL configs (list extra cells); is the ingest idempotent.
   Print the OFAT cell list per probe from stress.py applies_to with counts (about 40 per
   probe expected: all OFAT levels minus omega for ramsey, minus tau for rabi/dither); compare with
   the PREREG stress list and the cells that exist in the sealed data. Print the sides of
   C-1..C-8 from configs\frozen\selection.json. If any defect (duplicates, missing
   cells, unresolved pairing): STOP and report. Do not fix silently.
A2 write qstudy\contrasts.py (new file):
   - per-seed score S(config, seed) = macro-average over that probe's applicable OFAT cells
     of the cell's mean lock_w1000 (endpoint P2). Cells from stress.py applies_to.
   - RL-vs-RL (C-1,2,5,6,7,8): exact two-sided Mann-Whitney U, Cliff's delta, bootstrap
     95% CI of the mean difference (10000 resamples of seeds, fixed seed).
   - RL-vs-baseline (C-3,4): paired differences S(seed) minus the baseline's score,
     bootstrap 95% CI, number of seeds above baseline, and the p-value by exact two-sided
     Wilcoxon signed-rank (scipy, method exact). Log in events.md: "p-value test for paired
     baseline contrasts fixed before confirmatory contrasts were computed; PREREG
     specified CI and count only."
   - Holm over the 8. Pairings only from selection.json.
   - Asserts: exactly 10 seeds per RL config; seeds 0-4 absent; seed_base 900000 only
     (read from PREREG/config, no hard-coded value); baseline score exists; all 8 pairings
     resolve. A failed assert aborts loudly.
A3 test on SYNTHETIC data only (no DB writes): known shift (CI contains the true shift,
   p small), null (CI contains 0, p > 0.05), paired case. Print results. All must pass.
A4 run on the real data ONCE. Store in db.contrasts with status FINAL and n_a, n_b.
   (The old n=5 rows stay SUPERSEDED.) Print: contrast table with n_a, n_b, mean diff, CI,
   p_raw, p_holm, Cliff's delta / seeds above baseline (max 30 lines). Then a descriptive
   table of P1 and P2 per config (mean, sd) marked EXPLORATORY. Flag any contrast whose
   direction contradicts the preregistered expectation, without interpreting. If the code
   crashes, fix the code defect and re-run the SAME analysis; log every re-run.
Print "PHASE A DONE".

PHASE B: Z2A INTEGRITY CHECK (read-only except the backup).
 - state counts for z2a, z2b, z2e, z2v, z2x (expected 10/25/175/25/21, all DONE).
 - results of the Z2A sealed evals: file count vs expected from the job files, 0-byte
   or unparsable files, duplicate or .partial files. Do NOT read result values.
 - PREREG2.md vs PREREG2.sha256: MATCH required.
 - back up the Z2A sealed results and the DB to WORK\backup\pre_z2a_analysis_<date>,
   read-only, with a sha256 list.
 If anything fails: print "PHASE B BLOCKED", STOP. Else print "PHASE B DONE".

PHASE C: Z2A ANALYSIS under PREREG2 ("S15 Transfer study" and S15.4 of PART2 via secget).
 Seed base 920000, seeds 200-204 (n=5 per arm). Use a NEW script qstudy\transfer_analysis.py;
 do not mix with Z1 data or seed base 900000. Assert n=5 per arm and the seed base.
 C1 Confirmatory (family size 5 for Holm): compute ONLY TR-1 (LG zero-shot vs RB 2M from
    scratch, P2T) and TR-2 (LG+FT 200k vs RB 200k snapshot, P2T) with exact two-sided
    Mann-Whitney U, Cliff's delta, bootstrap 95% CI. Apply Holm with family size m=5 (NOT
    2): TR-3 (E6), TR-4 (LB arm) and TR-5 (SAC/TD3) need arms not yet trained; record
    them in the db as status PENDING_ARMS_NOT_RUN. Note this in events.md and say that
    with m=5 the correction is conservative. Do not shrink the family.
    With n=5 vs 5 the smallest exact two-sided p is 0.0079 (x5 for Holm = 0.0397):
    state the minimum attainable p-value next to the table.
 C2 Descriptive (EXPLORATORY): S15.4 a) transfer matrix for the arms that exist (rows E1
    and E4, columns the 4 environments), transfer gap G with CI; c) sample efficiency
    lock (P1, P2) versus quantum-env steps for scratch (50k/200k/500k/1M snapshots) and
    classical pretrain + FT (budgets as run), with wall-clock and break-even from measured
    steps/s; d) classical-vs-quantum environment correlation as specified in S15.4 d.
    NOT possible now (arms not run): b) gap decomposition by lineshape/noise, SAC/TD3
    comparisons, E5/E6 rows. e) failure/lock-state analysis is S18 and is not started.
    State this in the output instead of inventing values.
 C3 Store in the database (status FINAL for C1; EXPLORATORY for C2). Print TR table
    and the key descriptive tables (max 40 lines total). Print "PHASE C DONE".

PHASE D: WP2 tooling (CPU only; no new science).
 D1 tools\training_health.py: per config over finished seeds: final validation score, best
    validation score, steps to reach 90 percent of best, flag COLLAPSE_CANDIDATE when final
    < 0.6 x the config median. Run for Z1 (9 configs x 10 seeds) and Z2A arms. Report only.
 D2 tools\paired_stats.py as described in WP2 (b): paired difference per seed index of
    lock_w1000 at the nominal condition for the eight C-1..C-8, mean, bootstrap 95% CI
    (10000, fixed seed), Wilcoxon signed-rank p, Cohen dz. Test on synthetic data: known shift
    (CI contains it, p small), null (CI contains 0). IMPORTANT: its outputs are labelled
    EXPLORATORY and must NEVER be written to db.contrasts as FINAL or used in place of
    contrasts.py. Pairing by seed index is a convenience, not the preregistered analysis.
 D3 Print GATE G3 (max 15 lines): tests only. Print "PHASE D DONE". Do not start P6.

END: print "ANALYSIS BLOCK COMPLETE" with one line per phase (DONE or BLOCKED), then a
5-line list of anything left unchecked or any risk. Update RESUME_CARD.md. STOP.