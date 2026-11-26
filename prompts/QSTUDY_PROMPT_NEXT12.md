QSTUDY NEXT 12 - S15 TIER A (TRANSFER STUDY, MINIMUM SET): INSTALL, VALIDATE, SMOKE-TEST, GATE G2, SUBMIT   (executor: Nemotron 3.5 Super VL)
THIS_FILE12 = path given in the launcher message. z1 (the confirmatory run) is still running and must NOT be disturbed.
Rules: SECTION M0 of C:\Projects\MiniProject_Sem4\prompts\QSTUDY_PROMPT_MASTER11.md applies completely (read that one section first: python <WORK>\tools\secget.py <file> "SECTION M0" 80).
Read this file one section at a time with secget as well. The generator scripts were written and tested by the controller: you only RUN them.

WHAT THIS FILE DOES
 N1 lanes and state -> N2 inventory (read-only) -> N3 create the S15 Tier A configs and job files (nothing submitted) -> N4 validate -> N5 smoke tests (small jobs) -> N6 GATE G2 (STOP, user types GO-Z2A)
 -> N7 submit -> N8 standing work. Tier A = arms E1 (classical env LG) and E4 (quantum env RB), fine-tune E1->RB at 50k/200k/500k/1M (FT-B) and 200k (FT-A), seeds 200-204.
 It does NOT contain Tier B/C (E2, E3, E5, E6, SAC/TD3, other FT-A budgets), S16, S17-S20, nor any unblinding of z1.

==============================================================================================
SECTION N0 - ADDITIONAL RULES
==============================================================================================
Y1  Never edit gen_z2a.py or qs13_smoke_s15.py. If a script prints REFUSED or crashes: print the last 15 lines and STOP.
Y2  Never overwrite a file under STUDY\configs\frozen. The generator refuses by itself; do not work around it.
Y3  Sealed evaluation (split sealed, seed base 920000) is only submitted in N7 after GO-Z2A. Smoke tests use split val only and write outside STUDY\results.
Y4  Do not open sealed results. Do not run any analysis of z1 results. (Unblinding is a separate controller decision: GO-UNBLIND.)
Y5  The gpu lane is busy with z1 wave B. New gpu jobs sort after z1 by id (z1b < z2a < z2b) and start only when z1 training is finished. This is intended.

==============================================================================================
SECTION N1 - SESSION START
==============================================================================================
1. Lanes (always first):  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs10_lanes.py" ensure      EXPECTED: last line ALL LANES ACTIVE (wait up to 7 minutes; run it once more if it says STILL STARTING).
   If not active after a second run: STOP "LANES NOT ACTIVE - AWAITING CONTROLLER" and print  & $py "$w\tools\qs10_lanes.py" status.
2. Restart repair: run SECTION M1b of MASTER11 (R1 to R4) ONLY IF the laptop was restarted since the last session (the status counter in step 3 shows FAILED or BLOCKED z1 jobs, or lanes had to be started).
3. z1 progress (prints counts per group; no judgement):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-ChildItem "$w\queue\z1*.job.json" | ForEach-Object { $id = $_.Name -replace '\.job\.json$',''; $sp = "$w\queue\$id.status.json"; $st = if (Test-Path $sp) { (Get-Content $sp -Raw | ConvertFrom-Json).state } else { 'PENDING' }; "{0} {1}" -f $id.Substring(0,3), $st } | Group-Object | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Count)" }
   Tell the user the lines in one message. Check freeze: PREREG.md and PREREG2.md hashes (MASTER11 M1 step 5). Mismatch: STOP "FREEZE TAMPERED".

==============================================================================================
SECTION N2 - INVENTORY (read-only)
==============================================================================================
1. . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\state.py" $w show 2>&1 | Tee-Object "$w\audit\STAGE_STATUS.txt" | Select-Object -First 60
   Tell the user: "stage status saved to audit\STAGE_STATUS.txt (60 lines shown)". Do not interpret.
2. Check that no S15 Tier A job or config exists yet:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; (Get-ChildItem "$w\queue\z2*.job.json" -ErrorAction SilentlyContinue | Measure-Object).Count; (Get-ChildItem "$s\configs\frozen\s15__*.json" -ErrorAction SilentlyContinue | Measure-Object).Count
   EXPECTED: 0 and 0. If not 0: STOP and print them (a previous attempt exists).

==============================================================================================
SECTION N3 - CREATE CONFIGS AND JOB FILES (nothing is submitted)
==============================================================================================
1. Files present?  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-ChildItem "$w\tools\gen_z2a.py","$w\tools\qs13_smoke_s15.py","$w\tools\validate_jobs.py" | Select-Object Name,Length
   EXPECTED: three files. If gen_z2a.py or qs13_smoke_s15.py is missing: STOP "TOOLS MISSING - the user must copy gen_z2a.py and qs13_smoke_s15.py into WORK\tools".
2. Run the generator:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\gen_z2a.py" --study $s --work $w --py $py
   EXPECTED lines: z2a_train.json 35, z2a_eval.json 175, z2a_base.json 25, z2a_aer.json 21, and "GEN-Z2A-DONE total=256 (nothing submitted)". Any other number or REFUSED: STOP.
3. List the new config files (names only):  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-ChildItem "$s\configs\frozen\s15__*.json","$s\configs\eval__rabi__*.json" | Select-Object Name
   EXPECTED: s15__E1, s15__E4, s15__FTA_200000, s15__FTB_50000, s15__FTB_200000, s15__FTB_500000, s15__FTB_1000000, eval__rabi__LG, eval__rabi__LB, eval__rabi__RG.

==============================================================================================
SECTION N4 - VALIDATE
==============================================================================================
For each of the four files z2a_train.json, z2a_eval.json, z2a_base.json, z2a_aer.json (in WORK\jobs_json):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\validate_jobs.py" "$w\jobs_json\<file>"
EXPECTED for each: all checks PASS (check (j) may print n/a). Any FAIL: STOP and print the output (max 25 lines).

==============================================================================================
SECTION N5 - SMOKE TESTS (small jobs; split val; outputs outside STUDY\results)
==============================================================================================
1. Generate:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs13_smoke_s15.py" gen --study $s --work $w --py $py      EXPECTED: "p12_smoke.json: 6 jobs" and QS13-GEN-DONE.
   Validate it as in N4 (file WORK\jobs_json\p12_smoke.json).
2. Submit and count:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\jobtools.py" $w submit "$w\jobs_json\p12_smoke.json"
   Count the lines "SUBMITTED": must be 6; "EXISTS": must be 0. Otherwise STOP.
3. Wait until all six are DONE (poll every 10 minutes; the gpu job starts when the lane is free and takes about 1-2 minutes):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\brief.py" $w; Start-Sleep -Seconds 600
   A failed smoke job: print the last 15 lines of its log (WORK\results\<id>.log, decode utf-8 or utf-16) and STOP "SMOKE FAILED". Do not re-run and do not edit.
4. Check:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs13_smoke_s15.py" check --study $s --work $w
   EXPECTED: last line "SMOKE PASS", no NaN lines, "training outputs ... OK" for both arms. Print the output (max 20 lines).
5. Unit tests (re-queue the project's test job exactly as in MASTER11 M5 step 2). EXPECTED: OK and at least 138 tests.

==============================================================================================
SECTION N6 - GATE G2
==============================================================================================
Print "GATE-G2 REPORT" (max 35 lines): the four counts (35/175/25/21), the validator results, the smoke result (SMOKE PASS or FAIL), the unit-test result, the config list, and the GPU-time estimate:
  gpu_hours = (10 runs x 2,000,000 steps + 5 x (50,000 + 200,000 + 500,000 + 1,000,000 + 200,000) steps) / steps_per_s / 3600 with steps_per_s = the measured rate from the z1 training table (about 1,500).
  Also state how many gpu hours of z1 training are still ahead (count of z1b jobs not DONE x 0.4 h).
STOP. Continue only when the user types:  GO-Z2A

==============================================================================================
SECTION N7 - SUBMIT (only after GO-Z2A)
==============================================================================================
Submit in this order, each with validate_jobs before and the count check after (SUBMITTED equals the number of jobs, EXISTS equals 0): z2a_train.json (35), z2a_base.json (25), z2a_eval.json (175), z2a_aer.json (21).
Record the four counts in events.md and RESUME_CARD.md. Nothing else is submitted.

==============================================================================================
SECTION N8 - STANDING WORK
==============================================================================================
Poll (. env; & $py "$w\tools\brief.py" $w; Start-Sleep -Seconds 600), act only on NEW_FAILED or STALE_LANES (MASTER11 rule G and M1b for failed jobs; lanes with qs10_lanes.py ensure). Do not unblind. Do not start S16/S17. After 60 tool calls hand off.

APPENDIX - LAUNCHER MESSAGE
Read C:\Projects\MiniProject_Sem4\prompts\QSTUDY_PROMPT_NEXT12.md and execute it exactly. THIS_FILE12 = that file. FIRST run the lane check of SECTION N1, then read SECTION N0 of this file and MASTER11 SECTION M0.
