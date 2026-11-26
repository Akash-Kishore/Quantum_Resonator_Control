QSTUDY CLEANUP 7 - RESTART-DAMAGE REPAIR, FOLDER TIDY, RUN-AND-REPORT   (executor: GLM 5.3)
THIS_FILE7 = path given in the launcher message. This file REPLACES QSTUDY_PROMPT_AUDIT6.md (do NOT run AUDIT6).
Authority: this file > CONTINUATION5 > CONTINUATION4 > CONTINUATION3 > PART2 > PART1. All rules of SECTION P0 (S1-S7, Q1-Q10) of CONTINUATION5 stay in force.
Read one section at a time:  python WORK\tools\secget.py <THIS_FILE7> "SECTION C0" 80

WHAT THIS FILE DOES (in this order)
 C1 setup and checks  ->  C2 find damaged files (read-only) and prepare a repair plan  ->  GATE G6 (user types GO-DELETE)
 C3 move damaged files to a graveyard and re-queue their jobs  ->  C4 tidy C:\Projects (GATE G7, user types GO-ROOT)
 C5 run tests and build the interim report  ->  C6 wait for the re-queued jobs, re-check, final report  ->  resume normal work.
The controller wrote and TESTED every script used here (WORK\tools\qs7_*.py). You only RUN them. The z1 training and evaluation keep running all the time.

==============================================================================================
SECTION C0 - RULES (read first; they override habit)
==============================================================================================
W1  Do the steps in the given order. Do not improvise. Do not fix "other things you notice". Anything not in this file is forbidden.
W2  Every step has an EXPECTED result. If the real result is different, STOP, print the output (max 30 lines) and write "STEP MISMATCH - AWAITING CONTROLLER".
W3  You never write or edit the qs7_*.py scripts. If one crashes with a Python error, print the last 15 lines and STOP. Do not repair the script.
W4  You never delete anything yourself and you never use Remove-Item, del, rd, rmdir, or shutil.rmtree. Files are removed from the project only by qs7_apply.py,
    which MOVES them to WORK\graveyard, and only when the user has typed GO-DELETE. Folders of C:\Projects are moved only by qs7_root.py after GO-ROOT.
W5  Sealed results are BLINDED. Never open, print or summarise values from STUDY\results\sealed. The scripts only check file integrity. Do not run any
    analysis on sealed data.
W6  Never stop, pause or restart training or the lanes unless a lane is stale (then use P1 steps 6-7 of CONTINUATION5 for that lane only).
W7  Never modify: PREREG*.md, PREREG*.sha256, STUDY\configs\frozen, pipeline code, WORK\queue files by hand. (Re-queueing is done by qs7_apply.py or, in C5/C6, by the exact
    command given.)
W8  USER-APPROVED EXCEPTION to the sealed write-once rule: files that fail the integrity scan inside the restart window may be moved to the graveyard (even sealed ones) by
    qs7_apply.py. They are regenerated ONLY by re-running the SAME jobs (same ids, seeds, configs). Never by new seeds, new configs or new job ids. Log: "INFRASTRUCTURE REGENERATION".
W9  Every PowerShell command in this file starts with the line that loads the variables. Copy it exactly:
    . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1";
W10 Keep commands short. Print at most 30 lines. After each phase overwrite WORK\RESUME_CARD.md (max 100 lines) with: phase done, next step, approvals received (GO-Z1, GO-DELETE, GO-ROOT).
W11 After 70 tool calls: finish the current step, update RESUME_CARD.md, print "SESSION-HANDOFF - START A NEW CHAT WITH THE LAUNCHER" and stop.

==============================================================================================
SECTION C1 - SETUP AND CHECKS
==============================================================================================
1. Create the folder and the variables file (one command; the here-string must be written exactly):
   New-Item -ItemType Directory -Force "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit" | Out-Null
   Set-Content -Encoding utf8 "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1" -Value @(
     '$py   = "C:\Users\akash\anaconda3\envs\quantum_control\python.exe"',
     '$w    = "C:\Projects\MiniProject_Sem4_QSTUDY_WORK"',
     '$s    = "C:\Projects\MiniProject_Sem4\qstudy"',
     '$root = "C:\Projects"')
   EXPECTED: no output. Then run:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; "$py | $w | $s | $root"   -> four paths printed.
2. Check the controller scripts exist:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-ChildItem "$w\tools\qs7_*.py" | Select-Object Name,Length; Test-Path "$w\audit\Q1_PAPER_ROADMAP.md"
   EXPECTED: six files: qs7_apply.py, qs7_plan.py, qs7_report.py, qs7_root.py, qs7_scan.py, qs7_summaries.py, and True.
   If anything is missing: STOP "SCRIPTS MISSING - the user must unzip qstudy_tools_v7.zip into C:\Projects\MiniProject_Sem4_QSTUDY_WORK".
   Then save their hashes for the controller:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-ChildItem "$w\tools\qs7_*.py" | Get-FileHash -Algorithm SHA256 | ForEach-Object { "$($_.Hash.Substring(0,16)) $(Split-Path $_.Path -Leaf)" } | Tee-Object "$w\audit\qs7_hashes.txt"
3. Lanes: . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\brief.py" $w
   EXPECTED: six heartbeats younger than 60 s and STALE_LANES "-". If a lane is stale: P1 steps 6-7 of CONTINUATION5 for that lane only, then repeat this step.
4. Freeze check: compute SHA256 of STUDY\PREREG.md and PREREG2.md and compare with the first token of PREREG.sha256 and PREREG2.sha256 (same as P1 step 3 of CONTINUATION5).
   EXPECTED: both equal. Otherwise STOP "FREEZE TAMPERED - AWAITING CONTROLLER".
5. Restart times (one command, then read it):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $ev = Get-WinEvent -FilterHashtable @{LogName='System'; Id=6005; StartTime=(Get-Date).AddDays(-8)} -ErrorAction SilentlyContinue | Sort-Object TimeCreated; $b = @($ev | ForEach-Object { $_.TimeCreated.ToString('yyyy-MM-dd HH:mm') }); if ($b.Count -eq 0) { $b = @((Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm')) }; $b | Set-Content "$w\audit\boots.txt" -Encoding utf8; $b
   Then: count the lines of boots.txt that are >= "2026-10-17 12:00" (the time z1 training started). EXPECTED: exactly 1 line.
   If 0 or 2 or more: STOP "RESTART COUNT UNCLEAR" and print the boots.txt lines; the user will say which restart to use.
   If exactly 1: write that single line to WORK\audit\boot.txt (Set-Content -Encoding utf8). Tell the user in one line: "Restart used: <that time>".
6. Write RESUME_CARD.md (Next actions: C2). Continue.

==============================================================================================
SECTION C2 - FIND DAMAGED FILES (read-only) AND PREPARE THE PLAN
==============================================================================================
The scan reads many files and may take several minutes, so it runs in the background and writes to a file.
1. Start the scan:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $boot = (Get-Content "$w\audit\boot.txt" -TotalCount 1).Trim(); $a1 = @("$w\tools\qs7_scan.py","--roots","C:\Projects\MiniProject_Sem4",$w,"--study",$s,"--work",$w,"--boot",('"' + $boot + '"'),"--out","$w\audit\scan_1.csv"); Start-Process -FilePath $py -ArgumentList $a1 -RedirectStandardOutput "$w\audit\scan_1.out.txt" -RedirectStandardError "$w\audit\scan_1.err.txt" -WindowStyle Hidden
2. Wait: repeat up to 15 times (each time one command):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Start-Sleep -Seconds 60; Get-Content "$w\audit\scan_1.out.txt" -Tail 2; (Get-Item "$w\audit\scan_1.err.txt").Length
   Stop waiting when the last output line is SCAN-DONE. If the err file has length above 0: print it (30 lines) and STOP. If not done after 15 minutes: STOP "SCAN TOO SLOW".
3. Print the result: . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-Content "$w\audit\scan_1.out.txt" | Select-Object -First 40
   If the output contains a line starting with "  !! QUEUE_CORRUPT" or "  !! DB_CORRUPT": STOP "CRITICAL FILE DAMAGED - AWAITING CONTROLLER" (print those lines).
   If it shows "CORRUPT total=0" or "in_window=0": write "NO DAMAGE IN RESTART WINDOW" to events.md, skip to SECTION C4 (no stop).
4. Make the plan: . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_plan.py" --scan "$w\audit\scan_1.csv" --work $w --study $s --out "$w\audit\plan_1.csv" | Select-Object -First 60
   EXPECTED: last line PLAN-DONE. If it prints "NO JOBS FOUND": STOP.
   Lines starting with "  !!" mean the controller must decide: STOP and print them.
5. Dry run: . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_apply.py" "$w\audit\plan_1.csv" --work $w --study $s | Select-Object -First 30
   EXPECTED: "problems=0" and "DRY RUN ONLY". If it prints REFUSED: STOP and print it.
GATE G6  Print "GATE-G6 REPORT" (max 45 lines) containing: the scan summary lines (counts), the PLAN counts line, the list of items to be moved (count, then the first 40 paths with their reasons),
         the ids of the jobs to be re-queued (all), how many of the moved items are inside results\sealed (a number only, no values), and any "!!" lines.
         STOP. Continue only when the user types:  GO-DELETE

==============================================================================================
SECTION C3 - REPAIR (only after GO-DELETE)
==============================================================================================
1. Check lanes again (brief.py, same as C1 step 3). Then run the repair:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_apply.py" "$w\audit\plan_1.csv" --work $w --study $s --confirm GO-DELETE
   EXPECTED: "MOVED n FAILED 0", "RESET_JOBS m", "APPLY-DONE". If it prints REFUSED or FAILED: STOP and print it (nothing is lost; files are only moved).
2. Save the ids of the re-queued jobs:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Import-Csv "$w\audit\plan_1.csv" | Where-Object { $_.action -eq 'RESET_JOB' } | ForEach-Object { $_.target } | Set-Content "$w\audit\reset_ids.txt" -Encoding utf8; (Get-Content "$w\audit\reset_ids.txt").Count
3. Status counter (use it whenever asked to check the re-queued jobs; prints one line):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-Content "$w\audit\reset_ids.txt" | ForEach-Object { $sp = "$w\queue\$_.status.json"; if (Test-Path $sp) { (Get-Content $sp -Raw | ConvertFrom-Json).state } else { 'PENDING' } } | Group-Object | ForEach-Object { "$($_.Name)=$($_.Count)" }
   A re-queued job that becomes FAILED: apply X1-X4 of CONTINUATION5 once. If it fails twice: STOP.
4. Update RESUME_CARD.md (GO-DELETE received; waiting for re-queued jobs; next: C4). Do NOT wait here: continue with SECTION C4 while the jobs run.

==============================================================================================
SECTION C4 - TIDY C:\Projects
==============================================================================================
Rules built into the script: only items named MiniProject_Sem4* can be moved; the two live folders, every QSTUDY_* file and all other projects stay untouched; anything a script, config or
not-finished job refers to is KEPT. Nothing is deleted; items go to C:\Projects\_LEGACY_MiniProject_Sem4.
1. Report (background, may take minutes because of large folders):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $a2 = @("$w\tools\qs7_root.py","report","--root",$root,"--scan-roots","C:\Projects\MiniProject_Sem4",$w,"--out","$w\audit\root_report.csv"); Start-Process -FilePath $py -ArgumentList $a2 -RedirectStandardOutput "$w\audit\root_report.out.txt" -RedirectStandardError "$w\audit\root_report.err.txt" -WindowStyle Hidden
   Wait as in C2 step 2 (file root_report.out.txt, last line ROOT-REPORT-DONE, err file empty). Then print the file (max 40 lines).
GATE G7  Print "GATE-G7 REPORT": the table printed by the script and the PROPOSED_MOVES line. STOP. Continue only when the user types:  GO-ROOT
2. After GO-ROOT:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_root.py" apply --root $root --plan "$w\audit\root_report.csv" --confirm GO-ROOT
   EXPECTED: "MOVED n FAILED 0" and ROOT-APPLY-DONE. If FAILED is above 0 (a folder in use): print it and STOP.
3. Check that the live project is unharmed: brief.py (six heartbeats younger than 60 s, no new FAILED) and Test-Path of $s and $w and of $s\PREREG.md (all True).
4. Write the file C:\Projects\README_PROJECT_MAP.md with exactly this content:
   # Project map (written 2026-10-18)
   - MiniProject_Sem4\                    LIVE study code (qstudy\ = code, configs, runs, results). Do not rename or move.
   - MiniProject_Sem4_QSTUDY_WORK\        LIVE operations: job queue, lane runners, tools, logs, audit reports, graveyard of damaged files.
   - _LEGACY_MiniProject_Sem4\            Old audits, sandboxes, backups and zip files moved here. Nothing deleted. Delete only after the paper is submitted.
   - QSTUDY_PROMPT_*.md                   Controller prompt files. Launcher messages refer to these paths: do not move.
   - Birthday, Q-Quest, Quantum_VLSI, GLM Chat Exports   Other projects, untouched.
5. Update RESUME_CARD.md (GO-ROOT received; C4 done). Continue with SECTION C5.

==============================================================================================
SECTION C5 - RUN THE TESTS AND BUILD THE INTERIM REPORT
==============================================================================================
1. Unit tests, the project way (same job as before, re-queued):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Remove-Item "$w\queue\m2_suite_unittests.status.json","$w\queue\m2_suite_unittests.ok" -ErrorAction SilentlyContinue; Get-Content "$w\queue\m2_suite_unittests.job.json" -Raw | ConvertFrom-Json | Select-Object lane,cmd | Format-List
   (This is the only place where Remove-Item is allowed, exactly as written; it only re-queues the test job.) Poll with the poll command of CONTINUATION5 P4.3 until the job is DONE
   (check: if (Test-Path "$w\queue\m2_suite_unittests.ok") {'DONE'}). Then find its output file: the path after the ">" in the cmd printed above; copy its last 15 lines to WORK\audit\unit_tests.txt
   (Get-Content <file> -Tail 15 | Set-Content "$w\audit\unit_tests.txt" -Encoding utf8). EXPECTED: a line "OK" and a number of tests of at least 138. If the tests fail: STOP and print the last 30 lines.
2. Training table (background, same pattern as C2 step 1 and 2):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_summaries.py" --study $s --out "$w\audit\train_summaries.csv" | Select-Object -First 40
   EXPECTED: last line SUMMARIES-DONE. (If it takes longer than 4 minutes, use Start-Process with output files exactly as in C2.)
3. Find the existing exploratory review document and check that it holds no sealed data (it is added to the report as an appendix):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-ChildItem -Recurse -File -Include *review_pack* -Path $w,$s -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 8 FullName,Length,LastWriteTime
   Pick the NEWEST file whose name contains "review_pack" and whose extension is .md or .txt. Then compare dates:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $old = (Get-ChildItem "$s\results\sealed" -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -First 1).LastWriteTime; "oldest sealed file: $old"
   EXPECTED: the review pack is OLDER than the oldest sealed file (it was written before the freeze). If it is newer, or no review pack exists: do NOT append it; write "no review pack appended" in events.md and continue (do not stop).
   If it is appended, save its full path as one line in WORK\audit\appendix_paths.txt (Set-Content -Encoding utf8). If not appended, create that file empty.
4. Build interim report A (detailed; includes per-run tables, compute accounting, incidents and the appendix):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $ap = @(Get-Content "$w\audit\appendix_paths.txt" | Where-Object { $_.Trim() }); $ra = @("$w\tools\qs7_report.py","--work",$w,"--study",$s,"--out","$w\audit\RESULTS_REPORT_INTERIM_A.md","--scan","$w\audit\scan_1.csv","--plan","$w\audit\plan_1.csv","--summaries","$w\audit\train_summaries.csv","--root-report","$w\audit\root_report.csv","--tests","$w\audit\unit_tests.txt","--boots","$w\audit\boots.txt"); if ($ap.Count -gt 0) { $ra += "--appendix"; $ra += $ap }; & $py @ra
   EXPECTED: "REPORT written" and REPORT-DONE. If the scan or plan file does not exist (because C2 found no damage), leave those two options out.
5. Show what the report contains (max 30 lines): . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Select-String -Path "$w\audit\RESULTS_REPORT_INTERIM_A.md" -Pattern '^#' | Select-Object -First 30 | ForEach-Object { $_.Line }
   Tell the user the path and the line count in one line. Update RESUME_CARD.md.

==============================================================================================
SECTION C6 - WAIT FOR THE RE-QUEUED JOBS, RE-CHECK, FINAL REPORT
==============================================================================================
1. Check the re-queued jobs with the status counter of C3 step 3, then the poll command of CONTINUATION5 P4.3 (sleep 600). While waiting you may continue WP1 coding of CONTINUATION5 P5.
   Continue when the counter shows only DONE. Failures: C3 step 3 rule.
2. Output check (prints the number of missing expected outputs; EXPECTED 0):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $miss = 0; foreach ($id in Get-Content "$w\audit\reset_ids.txt") { $j = Get-Content "$w\queue\$id.job.json" -Raw | ConvertFrom-Json; foreach ($o in $j.expected_outputs) { $p = if ([System.IO.Path]::IsPathRooted($o)) { $o } else { Join-Path $j.cwd $o }; if (-not (Test-Path $p)) { $miss++; "MISSING $p" } } }; "missing=$miss"
3. Scan again (same command as C2 step 1 but with --out scan_2.csv, scan_2.out.txt, scan_2.err.txt). EXPECTED: "in_window=0" (regenerated files are newer than the restart window) and no "!!" lines.
   Then run the tests again as in C5 step 1 (copy to unit_tests.txt) and the summaries again as in C5 step 2.
4. Final report (same as C5 step 4 but with scan_2.csv and the output name RESULTS_REPORT_INTERIM.md):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $ap = @(Get-Content "$w\audit\appendix_paths.txt" | Where-Object { $_.Trim() }); $ra = @("$w\tools\qs7_report.py","--work",$w,"--study",$s,"--out","$w\audit\RESULTS_REPORT_INTERIM.md","--scan","$w\audit\scan_2.csv","--plan","$w\audit\plan_1.csv","--summaries","$w\audit\train_summaries.csv","--root-report","$w\audit\root_report.csv","--tests","$w\audit\unit_tests.txt","--boots","$w\audit\boots.txt"); if ($ap.Count -gt 0) { $ra += "--appendix"; $ra += $ap }; & $py @ra
5. Print "GATE-G8 REPORT" (max 20 lines): corrupt files after repair (expected 0), re-queued jobs done n/n, missing outputs (expected 0), unit tests result, the full path and size of RESULTS_REPORT_INTERIM.md,
   the graveyard folder path, and the line: "User: upload RESULTS_REPORT_INTERIM.md to the controller. You may purge the graveyard folder yourself after the controller confirms."
   Do NOT stop. Continue with the standing instructions of CONTINUATION5: poll loop (sleep 600), WP1 coding (authorised), submit nothing for z2a until the user types GO-Z2A, WP2 only after GO-P6,
   and when the last wave-A training is DONE print the P4.5 table.

==============================================================================================
APPENDIX - LAUNCHER MESSAGE (for the user)
==============================================================================================
Read C:\Projects\QSTUDY_PROMPT_CLEANUP7.md and execute it exactly. THIS_FILE7 = that file. Start at SECTION C1. Read SECTION C0 first. Do not read any other prompt file in full.
