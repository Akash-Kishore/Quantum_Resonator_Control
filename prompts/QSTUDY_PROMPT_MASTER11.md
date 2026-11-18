QSTUDY MASTER 11 - ONE FILE: RESUME CLEANUP, FINISH THE REPORT, PURGE SANDBOXES, LEGACY CHECK, REPAIR CHECK, CODE AUDIT   (executor: Nemotron 3.5 Super VL)
THIS_FILE11 = path given in the launcher message. This file is SELF-CONTAINED: it replaces QSTUDY_PROMPT_BRIDGE10, AUDIT8, CLEANUP7 and AUDIT6. You do not need to open those.
Only CONTINUATION5 (SECTION P5, packet WP1) is needed later, in section M7.
Authority: this file > CONTINUATION5 > earlier files. Read one section at a time:  python C:\Projects\MiniProject_Sem4_QSTUDY_WORK\tools\secget.py <THIS_FILE11> "SECTION M0" 80

WHAT HAPPENS IN THIS FILE (in order)
 M1 check state and start the lanes -> M1b repair round after any restart (GATE G6b) -> M2 finish report A -> M3 purge test sandboxes (GATE G7b) -> M4 legacy dependency report (GATE G7c) -> M5 wait for the 5 re-queued jobs, re-check, final report (GATE G8)
 -> M6 read-only code audit and as-run snapshot (GATE G9) -> M7 standing work. All scripts were written and tested by the controller. You only RUN them.

==============================================================================================
SECTION M0 - RULES (read first; they override habit)
==============================================================================================
A. HOW TO WORK
 A1 Work in the given order. Do not improvise. Do not fix other things you notice. Anything not written here is forbidden.
 A2 Every step has an EXPECTED result. If the real result differs: STOP, print the output (max 30 lines), write "STEP MISMATCH - AWAITING CONTROLLER".
 A3 Keep reasoning short (at most 10 lines per decision). Print at most 30 lines per command. Never read a file in full; use Select-String or Get-Content -First/-Tail.
 A4 Your reply is limited to about 8,000 tokens. Write files in pieces of at most 80 lines.
 A5 Ignore editor/LSP diagnostics completely.
 A6 After 60 tool calls: finish the step, overwrite WORK\RESUME_CARD.md (max 100 lines: done, next step, approvals), print "SESSION-HANDOFF - START A NEW CHAT WITH THE LAUNCHER" and stop.
 A7 Tag any statement you did not verify in this session as ASSUMPTION. Log decisions as one line in WORK\logs\events.md with the CORRECT date (run Get-Date, never guess).
B. WHAT IS FORBIDDEN
 B1 Never edit or create any qs*.py script. If one crashes, print the last 15 lines and STOP.
 B2 Never delete anything. The ONLY deleting command is "qs7_root.py purge" in M3, and only after the user typed GO-PURGE. Moving files to the graveyard is done only by qs7_apply.py after GO-DELETE.
    The only other allowed Remove-Item commands are the ones written exactly in M5 step 2 and rule G. Re-queueing and folder renames after a restart are done only by qs10_triage.py --apply.
 B3 Sealed results are BLINDED: never open or print values from STUDY\results\sealed. Never drop, re-run or re-seed a confirmatory result because of its numbers.
 B4 Never modify: PREREG*.md, PREREG*.sha256, STUDY\configs\frozen, pipeline code, queue files by hand. Never stop or restart training or lanes (a stale lane is relaunched with M1 step 3).
 B5 WORKING DIRECTORY is C:\Projects. Inside it you may only touch MiniProject_Sem4, MiniProject_Sem4_QSTUDY_WORK, _LEGACY_MiniProject_Sem4 and the prompts folder.
    OUT OF BOUNDS: Birthday, Q-Quest, Quantum_VLSI, AI-Vault, GLM Chat Exports, .git.
C. LOCATIONS
 C1 Prompt files live in C:\Projects\MiniProject_Sem4\prompts\ . Write that path into RESUME_CARD.md.
 C2 Every PowerShell command starts with:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1";
D. MODEL PILOT: you are a new model on this project. Steps M1.1 to M1.5 are checkpoints. If you invent a command that is not written here, edit a script, ignore a STOP, or fail the same step twice,
   write "MODEL PILOT FAILED" with the reason and stop. The user will then switch back to the previous model.
E0. EVERY session starts with M1 step 3 (lanes). The user restarts the laptop between tasks; lanes that are not active get started before anything else.
E. GATES need the exact string from the user: GO-DELETE (M1b, only when its auto-approval conditions are not met), GO-PURGE (M3), DECIDE-LEGACY (M4). Gates G8 and G9 do not stop you.
F. WAIT PATTERN for background commands (used in M4, M5, M6): after Start-Process, repeat up to 15 times, one command each time:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Start-Sleep -Seconds 60; Get-Content "$w\audit\<name>.out.txt" -Tail 2; (Get-Item "$w\audit\<name>.err.txt").Length
   Stop waiting when the last output line is the DONE word named in the step. If the err file is longer than 0: print it (30 lines) and STOP. After 15 minutes: STOP "TOO SLOW".
G. FAILED JOB RULE: a job that FAILED because of the machine (crash, missing file, restart) may be re-queued ONCE: Remove-Item "$w\queue\<id>.status.json","$w\queue\<id>.ok" -ErrorAction SilentlyContinue (this is the third allowed Remove-Item).
   A second failure: STOP and report. Never re-queue because of a result value.

==============================================================================================
SECTION M1 - CHECK STATE (pilot checkpoints)
==============================================================================================
1. Read the last 60 lines of C:\Projects\MiniProject_Sem4_QSTUDY_WORK\RESUME_CARD.md. Tell the user in 3 lines where the work stopped (it stopped at "C5 step 2": qs7_summaries.py crashed with a seed collision; the script is now fixed).
2. Variables file. Check:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; "$py | $w | $s | $root"      EXPECTED: four paths. If env.ps1 is missing: STOP and tell the user.
3. LANES (this step opens EVERY session, whatever the task; the user restarts the laptop between tasks, so the lanes are usually down at session start):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs10_lanes.py" ensure
   It starts the missing or stale lanes one at a time, starts the watchdog if none exists, and waits up to 150 s. EXPECTED: last line  ALL LANES ACTIVE  (it takes 1-4 minutes after a restart; wait for it).
   If the last line is "NOT ACTIVE: <lanes>": run it ONE more time. If still not active, run this READ-ONLY check, print the output of both commands (max 30 lines) and STOP "LANES NOT ACTIVE - AWAITING CONTROLLER":
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs10_lanes.py" status; foreach ($l in 'gpu','aer','cpu1','cpu2','cpu3','cpu4') { "--- $l"; & $py "$w\tools\qs7_patch_runner2.py" --work $w --selftest $l }
   If qs10_lanes.py is missing, use the old manual method: relaunch each lane with  powershell -NoProfile -ExecutionPolicy Bypass -File "$w\tools\launch_runner.ps1" -Py "$py" -Work "$w" -Lane <lane>  one at a time with 10 s pauses, then start the watchdog with
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="powershell -NoProfile -ExecutionPolicy Bypass -File `"$w\tools\watchdog.ps1`" -Py `"$py`" -Work `"$w`" -Lanes `"gpu,aer,cpu1,cpu2,cpu3,cpu4`""; CurrentDirectory="$w"}; "PID=$($r.ProcessId) RC=$($r.ReturnValue)"
   and check with  & $py "$w\tools\brief.py" $w  after 120 s.
4. Tools are in place:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-ChildItem "$w\tools\qs7_*.py","$w\tools\qs8_*.py","$w\tools\qs9_*.py","$w\tools\qs10_*.py" | Select-Object Name
   EXPECTED: qs7_apply, qs7_plan, qs7_report, qs7_root, qs7_scan, qs7_summaries, qs8_compare, qs8_scan, qs8_snapshot, qs8_tree, qs9_legacy_refs, qs10_lanes, qs10_triage (all .py).
   Fixed script check:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Select-String -Path "$w\tools\qs7_summaries.py" -Pattern "summary_" -Quiet      EXPECTED: True
   If a tool is missing or False: STOP "TOOLS MISSING - the user must unzip qstudy_tools_v9.zip into C:\Projects\MiniProject_Sem4_QSTUDY_WORK".
5. Freeze check (one command):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $h1=(Get-FileHash "$s\PREREG.md" -Algorithm SHA256).Hash.ToLower(); $e1=(Get-Content "$s\PREREG.sha256" -TotalCount 1).Split(' ')[0].ToLower(); $h2=(Get-FileHash "$s\PREREG2.md" -Algorithm SHA256).Hash.ToLower(); $e2=(Get-Content "$s\PREREG2.sha256" -TotalCount 1).Split(' ')[0].ToLower(); "PREREG match: $($h1 -eq $e1)  PREREG2 match: $($h2 -eq $e2)"
   EXPECTED: both True. Otherwise STOP "FREEZE TAMPERED - AWAITING CONTROLLER".
6. Status of the 5 re-queued jobs (prints one line, e.g. DONE=3 RUNNING=2). No stop whatever the result:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-Content "$w\audit\reset_ids.txt" | ForEach-Object { $sp = "$w\queue\$_.status.json"; if (Test-Path $sp) { (Get-Content $sp -Raw | ConvertFrom-Json).state } else { 'PENDING' } } | Group-Object | ForEach-Object { "$($_.Name)=$($_.Count)" }
   Create the folder for this file's outputs:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; New-Item -ItemType Directory -Force "$w\audit8" | Out-Null

==============================================================================================
SECTION M1b - REPAIR ROUND AFTER A RESTART (repeat this section after every new restart; GATE G6b)
==============================================================================================
A restart kills running jobs. The runners mark such jobs FAILED when they start. This round (1) finds damaged files near every restart since z1 started, (2) lists every FAILED or BLOCKED z1 job.
Jobs that failed because the machine restarted are re-queued ONCE (rule G). Nothing is re-queued because of a result value.
R1. Restart times since z1 started (2026-10-17 12:00):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $ev = Get-WinEvent -FilterHashtable @{LogName='System'; Id=6005; StartTime=(Get-Date).AddDays(-8)} -ErrorAction SilentlyContinue | Sort-Object TimeCreated; $b = @($ev | ForEach-Object { $_.TimeCreated.ToString('yyyy-MM-dd HH:mm') } | Where-Object { $_ -ge '2026-10-17 12:00' }); $b | Set-Content "$w\audit\boot.txt" -Encoding utf8; $b | Set-Content "$w\audit\boots.txt" -Encoding utf8; $b
   EXPECTED: at least the three known lines (2026-10-17 18:27, 2026-10-18 06:14, 2026-10-18 07:32) plus the newer restarts. Tell the user the number of restarts and their times in one line.
   If the list is empty or has fewer than three lines: STOP "RESTART LIST UNCLEAR".
R2. Damage scan with all restart times (background; DONE word SCAN-DONE; file name scan_3), then the plan and its dry run:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $bootArgs = @(); foreach ($b in (Get-Content "$w\audit\boot.txt" | Where-Object { $_.Trim() })) { $bootArgs += "--boot"; $bootArgs += ('"' + $b.Trim() + '"') }; $a1 = @("$w\tools\qs7_scan.py","--roots","C:\Projects\MiniProject_Sem4",$w,"--study",$s,"--work",$w) + $bootArgs + @("--out","$w\audit\scan_3.csv"); Start-Process -FilePath $py -ArgumentList $a1 -RedirectStandardOutput "$w\audit\scan_3.out.txt" -RedirectStandardError "$w\audit\scan_3.err.txt" -WindowStyle Hidden
   Wait with the WAIT PATTERN (rule F), then print:  Get-Content "$w\audit\scan_3.out.txt" | Select-Object -First 30
   If it shows a line starting with "  !! QUEUE_CORRUPT" or "  !! DB_CORRUPT": STOP "CRITICAL FILE DAMAGED - AWAITING CONTROLLER".
   Plan:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_plan.py" --scan "$w\audit\scan_3.csv" --work $w --study $s --out "$w\audit\plan_2.csv" | Select-Object -First 60
   EXPECTED: last line PLAN-DONE (or "PLAN empty"). Lines starting with "  !!" mean the controller must decide: STOP and print them.
   Dry run:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_apply.py" "$w\audit\plan_2.csv" --work $w --study $s | Select-Object -First 30      EXPECTED: "problems=0" and "DRY RUN ONLY". REFUSED: STOP and print it.
R3. Triage of all FAILED and BLOCKED z1 jobs (read-only dry run; the tool enforces the rules: only z1 ids, never a RUNNING job, a FAILED job at most 2 times in total, incomplete training folders are renamed never deleted):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs10_triage.py" --work $w | Select-Object -First 50
   EXPECTED: "TRIAGE failed_or_blocked=<n> incomplete_training_folders=<m>" and "DRY RUN ONLY". A line "!! NEEDS CONTROLLER" means a job already failed twice: STOP and print it.
   Sealed items in the plan (a number):  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; (Import-Csv "$w\audit\plan_2.csv" | Where-Object { $_.action -like 'GRAVEYARD*' -and $_.target -like '*\results\sealed\*' } | Measure-Object).Count
   Old damage (REPORT_ONLY rows, outside the restart windows; report only, do nothing with them):  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Import-Csv "$w\audit\plan_2.csv" | Where-Object { $_.action -eq 'REPORT_ONLY' } | Select-Object -First 30 target,reason | Format-Table -AutoSize | Out-String -Width 220
GATE G6b  Print "GATE-G6b REPORT" (max 50 lines): the restart times; the scan_3 summary lines; the PLAN counts; the items the plan would move (count and first 20 paths) and the jobs it would re-queue;
          the triage list (count and first 30 lines); the number of sealed items; the number of REPORT_ONLY rows and the first 15 of them.
          AUTO-APPROVAL: if the number of sealed items is 0 AND there is no line starting with "  !!" anywhere AND the plan has no BLOCK or NEEDS_CONTROLLER row, write
          "auto-approved: no sealed items" to events.md and run R4 WITHOUT stopping. In every other case STOP and continue only when the user types:  GO-DELETE
          If the plan is empty and the triage list is empty, write "REPAIR ROUND: nothing to do" and continue.
R4. Repair (each command once; EXPECTED results in brackets):
   a. Plan (skip if the plan is empty):  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_apply.py" "$w\audit\plan_2.csv" --work $w --study $s --confirm GO-DELETE      [MOVED n FAILED 0, RESET_JOBS m, APPLY-DONE; on REFUSED or FAILED: STOP]
      Add the plan's re-queued ids to the list:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Import-Csv "$w\audit\plan_2.csv" | Where-Object { $_.action -eq 'RESET_JOB' } | ForEach-Object { $_.target } | Add-Content "$w\audit\reset_ids.txt"
   b. Triage:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs10_triage.py" --work $w --apply | Select-Object -First 50      [APPLIED renamed=<n> requeued=<m>, TRIAGE-DONE]
   c. Remove duplicates in the list and count:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-Content "$w\audit\reset_ids.txt" | Where-Object { $_.Trim() } | Sort-Object -Unique | Set-Content "$w\audit\reset_ids.txt" -Encoding utf8; (Get-Content "$w\audit\reset_ids.txt").Count
   d. Append to events.md (correct date from Get-Date): "restart repair round done". Update RESUME_CARD.md. Continue with SECTION M2 (do not wait here; M5 waits for these jobs).

==============================================================================================
SECTION M2 - FINISH THE INTERIM REPORT A
==============================================================================================
1. Training table:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_summaries.py" --study $s --out "$w\audit\train_summaries.csv" | Select-Object -First 40
   EXPECTED: "SUMMARIES runs_with_summary=<n> unreadable=0" and last line SUMMARIES-DONE. If it crashes: print the last 15 lines and STOP.
2. Find the existing exploratory review document (it is added to the report as an appendix; it must hold no sealed data):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-ChildItem -Recurse -File -Include *review_pack* -Path $w,$s -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 8 FullName,Length,LastWriteTime
   Pick the NEWEST file whose name contains "review_pack" and whose extension is .md or .txt. Compare dates:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $old = (Get-ChildItem "$s\results\sealed" -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -First 1).LastWriteTime; "oldest sealed file: $old"
   EXPECTED: the review pack is OLDER than the oldest sealed file. If it is newer or does not exist: do NOT append it; write "no review pack appended" in events.md and continue.
   If appended: save its full path as one line in WORK\audit\appendix_paths.txt (Set-Content -Encoding utf8). If not appended: create that file empty.
3. Build report A:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $ap = @(Get-Content "$w\audit\appendix_paths.txt" | Where-Object { $_.Trim() }); $ra = @("$w\tools\qs7_report.py","--work",$w,"--study",$s,"--out","$w\audit\RESULTS_REPORT_INTERIM_A.md","--scan","$w\audit\scan_1.csv","--plan","$w\audit\plan_1.csv","--summaries","$w\audit\train_summaries.csv","--root-report","$w\audit\root_report.csv","--tests","$w\audit\unit_tests.txt","--boots","$w\audit\boots.txt"); if ($ap.Count -gt 0) { $ra += "--appendix"; $ra += $ap }; & $py @ra
   EXPECTED: "REPORT written" and REPORT-DONE.
4. Contents (max 30 lines): . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Select-String -Path "$w\audit\RESULTS_REPORT_INTERIM_A.md" -Pattern '^#' | Select-Object -First 30 | ForEach-Object { $_.Line }
   Tell the user the path and the line count in one line. Update RESUME_CARD.md.

==============================================================================================
SECTION M3 - PURGE OF TEST SANDBOXES (needs GO-PURGE)
==============================================================================================
Only folders inside C:\Projects\_LEGACY_MiniProject_Sem4 whose name contains sandbox, audit or fixrun, and that no script, config or unfinished job refers to, can be purged. ARCHIVE, BACKUP and zip items are NEVER purged by this command.
1. Dry run (nothing is deleted):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_root.py" purge --root $root --scan-roots "C:\Projects\MiniProject_Sem4" $w | Select-Object -First 30
   EXPECTED: PURGE for MiniProject_Sem4_AUDIT, _AUDIT3, _SANDBOX, _SANDBOX2 (about 8 GB) and "DRY RUN ONLY". The folders AUDIT2, FIXRUN and SANDBOX3 are not in the legacy folder and must not appear.
GATE G7b  Print "GATE-G7b REPORT" (the dry-run list). STOP. Continue only when the user types:  GO-PURGE
2. After GO-PURGE:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_root.py" purge --root $root --scan-roots "C:\Projects\MiniProject_Sem4" $w --confirm GO-PURGE
   EXPECTED: "PURGED n FAILED 0" and ROOT-PURGE-DONE. If FAILED is above 0: print it and STOP. Then run M1 step 3 (lanes) to confirm the live project is unharmed. Update RESUME_CARD.md.
If the user types something else instead of GO-PURGE, follow that instruction and skip M3.

==============================================================================================
SECTION M4 - LEGACY DEPENDENCY REPORT (read-only; needs DECIDE-LEGACY)
==============================================================================================
Three old folders are still referenced: MiniProject_Sem4_AUDIT2 (by verification_tools\aggregate2.py), MiniProject_Sem4_FIXRUN and MiniProject_Sem4_SANDBOX3 (by result files). We must know WHAT is referenced before anything is done with them.
1. Start (background; DONE word is QS9-LEGACY-REFS-DONE; file name is legacy_refs):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $a4 = @("$w\tools\qs9_legacy_refs.py","--root",$root,"--scan-roots","C:\Projects\MiniProject_Sem4",$w,"--names","MiniProject_Sem4_AUDIT2","MiniProject_Sem4_FIXRUN","MiniProject_Sem4_SANDBOX3","--csv","$w\audit\legacy_refs.csv"); Start-Process -FilePath $py -ArgumentList $a4 -RedirectStandardOutput "$w\audit\legacy_refs.out.txt" -RedirectStandardError "$w\audit\legacy_refs.err.txt" -WindowStyle Hidden
   Wait with the WAIT PATTERN (rule F).
GATE G7c  Print "GATE-G7c REPORT": the full text of legacy_refs.out.txt (max 40 lines). Do NOT move or delete the three folders. STOP. Continue only when the user types:  DECIDE-LEGACY

==============================================================================================
SECTION M5 - WAIT FOR THE RE-QUEUED JOBS, RE-CHECK, FINAL REPORT (GATE G8)
==============================================================================================
1. Check the 5 re-queued jobs with the counter of M1 step 6, then poll until all are DONE (one command per poll):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\brief.py" $w; Start-Sleep -Seconds 600
   A re-queued job that FAILED: apply rule G once. While waiting you may do SECTION M6 (it is read-only).
2. Unit tests again (re-queues the project's test job; Remove-Item is allowed here exactly as written):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Remove-Item "$w\queue\m2_suite_unittests.status.json","$w\queue\m2_suite_unittests.ok" -ErrorAction SilentlyContinue; Get-Content "$w\queue\m2_suite_unittests.job.json" -Raw | ConvertFrom-Json | Select-Object lane,cmd | Format-List
   Poll until  if (Test-Path "$w\queue\m2_suite_unittests.ok") {'DONE'}  prints DONE (poll command above). The output file is the path after the ">" in the cmd printed above.
   Copy its last 15 lines:  Get-Content <that file> -Tail 15 | Set-Content "$w\audit\unit_tests.txt" -Encoding utf8
   EXPECTED: a line "OK" and a number of tests of at least 138. If the tests fail: STOP and print the last 30 lines.
3. Output check (EXPECTED missing=0):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $miss = 0; foreach ($id in Get-Content "$w\audit\reset_ids.txt") { $j = Get-Content "$w\queue\$id.job.json" -Raw | ConvertFrom-Json; foreach ($o in $j.expected_outputs) { $p = if ([System.IO.Path]::IsPathRooted($o)) { $o } else { Join-Path $j.cwd $o }; if (-not (Test-Path $p)) { $miss++; "MISSING $p" } } }; "missing=$miss"
4. Second damage scan, with all three restart times (background; DONE word SCAN-DONE; file name scan_2):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $bootArgs = @(); foreach ($b in (Get-Content "$w\audit\boot.txt" | Where-Object { $_.Trim() })) { $bootArgs += "--boot"; $bootArgs += ('"' + $b.Trim() + '"') }; $a1 = @("$w\tools\qs7_scan.py","--roots","C:\Projects\MiniProject_Sem4",$w,"--study",$s,"--work",$w) + $bootArgs + @("--out","$w\audit\scan_2.csv"); Start-Process -FilePath $py -ArgumentList $a1 -RedirectStandardOutput "$w\audit\scan_2.out.txt" -RedirectStandardError "$w\audit\scan_2.err.txt" -WindowStyle Hidden
   Wait with the WAIT PATTERN, then print:  Get-Content "$w\audit\scan_2.out.txt" | Select-Object -First 30
   EXPECTED: three BOOT_WINDOWS, "in_window=0" and no line starting with "  !!". Otherwise STOP and print the output.
5. Training table again (as M2 step 1) and the FINAL report:
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs7_summaries.py" --study $s --out "$w\audit\train_summaries.csv" | Select-Object -First 40
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $ap = @(Get-Content "$w\audit\appendix_paths.txt" | Where-Object { $_.Trim() }); $ra = @("$w\tools\qs7_report.py","--work",$w,"--study",$s,"--out","$w\audit\RESULTS_REPORT_INTERIM.md","--scan","$w\audit\scan_2.csv","--plan","$w\audit\plan_1.csv","--summaries","$w\audit\train_summaries.csv","--root-report","$w\audit\root_report.csv","--tests","$w\audit\unit_tests.txt","--boots","$w\audit\boots.txt"); if ($ap.Count -gt 0) { $ra += "--appendix"; $ra += $ap }; & $py @ra
GATE G8  Print "GATE-G8 REPORT" (max 20 lines): corrupt files after repair (expected 0), re-queued jobs done n/n, missing outputs (expected 0), unit tests result, the full path and size of RESULTS_REPORT_INTERIM.md,
         the graveyard folder path, and the line: "User: upload RESULTS_REPORT_INTERIM.md to the controller." DO NOT STOP: continue with SECTION M6.

==============================================================================================
SECTION M6 - READ-ONLY CODE AUDIT AND AS-RUN SNAPSHOT (nothing is edited; all outputs go to WORK\audit8)
==============================================================================================
Purpose: the controller will later restructure the project into one folder. That can only start when z1 has finished. This section prepares it. You change NOTHING: no edit, move, delete or refactor.
1. AS-RUN SNAPSHOT first (immutable zip of the code and configs exactly as they ran):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs8_snapshot.py" --roots "C:\Projects\MiniProject_Sem4" $w --out "$w\audit8\AS_RUN_v1.zip" | Select-Object -First 12
   EXPECTED: "SNAPSHOT files=<n>" with n larger than 30, a ZIP_SHA256 line, QS8-SNAPSHOT-DONE. Otherwise STOP.
   Save and lock:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; (Get-FileHash "$w\audit8\AS_RUN_v1.zip" -Algorithm SHA256).Hash.ToLower() | Set-Content "$w\audit8\AS_RUN_v1.sha256.txt" -Encoding utf8; Set-ItemProperty "$w\audit8\AS_RUN_v1.zip" -Name IsReadOnly -Value $true; Get-Content "$w\audit8\AS_RUN_v1.sha256.txt"
   Check:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); print('members',len(z.namelist()),'testzip',z.testzip())" "$w\audit8\AS_RUN_v1.zip"      EXPECTED: testzip None
   Append to events.md: "AS_RUN_v1 snapshot created, sha256 <first 16 characters>".
2. Static scan:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs8_scan.py" --roots "C:\Projects\MiniProject_Sem4" $w --queue "$w\queue" --csv "$w\audit8\code_scan.csv" --md "$w\audit8\CODE_AUDIT_STATIC.md" | Select-Object -First 40
   EXPECTED: first line "QS8 python_files=<n> lines=<n> findings=<n>", counts per kind, last line QS8-SCAN-DONE. Tell the user in 3 lines: python files and lines, total findings, the three most frequent kinds. No interpretation. Do not fix anything.
3. Size map (background; DONE word QS8-TREE-DONE; file name tree in audit8, so wait on "$w\audit8\tree.out.txt" and "$w\audit8\tree.err.txt"):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $a3 = @("$w\tools\qs8_tree.py","--roots","C:\Projects\MiniProject_Sem4",$w,"--md","$w\audit8\PROJECT_SIZES.md"); Start-Process -FilePath $py -ArgumentList $a3 -RedirectStandardOutput "$w\audit8\tree.out.txt" -RedirectStandardError "$w\audit8\tree.err.txt" -WindowStyle Hidden
4. DETERMINISM PROBE (non-blocking: if it takes more than 8 tool calls, skip it and write "determinism probe skipped" in the report). Question: does the same evaluation, run twice, give identical numbers?
   a. List the smoke jobs:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-ChildItem "$w\queue\p3_*.job.json" | Select-Object -First 8 Name
      If none exist: write "determinism probe skipped (no p3 jobs)" and go to step 5. Otherwise print the cmd of one rl evaluation job and one ctrl:pid evaluation job.
   b. Create 4 jobs by copying those two job files and changing only id, the output folder in cmd and expected_outputs: p8_det_rl_run1, p8_det_rl_run2, p8_det_pid_run1, p8_det_pid_run2 (lanes cpu1-cpu3).
      Rules: split val (NEVER sealed), the same episodes and the same seed base in run1 and run2, outputs in WORK\audit8\det_rl_run1, det_rl_run2, det_pid_run1, det_pid_run2.
      Submit with: & $py "$w\tools\jobtools.py" $w submit <file.json>  (count the SUBMITTED lines: they must equal 4 and EXISTS must be 0), then poll as in M5 step 1 until all four are DONE.
   c. Compare:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs8_compare.py" "$w\audit8\det_rl_run1" "$w\audit8\det_rl_run2" | Tee-Object "$w\audit8\determinism.txt"; & $py "$w\tools\qs8_compare.py" "$w\audit8\det_pid_run1" "$w\audit8\det_pid_run2" | Tee-Object -Append "$w\audit8\determinism.txt"
      EXPECTED: "EXACT MATCH" twice. If NOT EXACT: do not investigate; the numbers go into the report as they are.
5. Assemble the report (one command):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $r = "$w\audit8\CODE_AUDIT_REPORT.md"; ("# Code and structure audit report`n`nGenerated: " + (Get-Date -Format 'yyyy-MM-dd HH:mm') + "`n`nAS_RUN_v1.zip sha256: " + (Get-Content "$w\audit8\AS_RUN_v1.sha256.txt" -TotalCount 1) + "`n`n## Folder sizes`n") | Set-Content $r -Encoding utf8; if (Test-Path "$w\audit8\PROJECT_SIZES.md") { Get-Content "$w\audit8\PROJECT_SIZES.md" | Add-Content $r -Encoding utf8 }; "`n" | Add-Content $r -Encoding utf8; Get-Content "$w\audit8\CODE_AUDIT_STATIC.md" | Add-Content $r -Encoding utf8; if (Test-Path "$w\audit8\determinism.txt") { "`n## Determinism probe (same evaluation run twice)`n" | Add-Content $r -Encoding utf8; Get-Content "$w\audit8\determinism.txt" | Add-Content $r -Encoding utf8 }; (Get-Item $r).Length
GATE G9  Print "GATE-G9 REPORT" (max 25 lines): paths and sizes of AS_RUN_v1.zip, CODE_AUDIT_REPORT.md and code_scan.csv; python files and lines; findings per kind (counts only); the determinism result (EXACT MATCH / NOT EXACT / skipped);
         and the line: "User: upload AS_RUN_v1.zip, CODE_AUDIT_REPORT.md and code_scan.csv to the controller. Do NOT start any refactoring." DO NOT STOP: continue with SECTION M7.

==============================================================================================
SECTION M7 - STANDING WORK
==============================================================================================
1. Poll loop when idle (act only if NEW_FAILED or STALE_LANES is not "-"; apply rule G to a failed job):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\brief.py" $w; Start-Sleep -Seconds 600
2. Coding packet WP1 (SECTION P5 of C:\Projects\MiniProject_Sem4\prompts\QSTUDY_PROMPT_CONTINUATION5.md, read with secget) is AUTHORISED. Submitting any z2a job requires the user to type GO-Z2A after GATE G2. WP2 only after GO-P6.
3. When the last wave-A training job (z1a_*) is DONE: run the training table (M2 step 1) and print it. Report only; collapsed seeds are data.
4. The restructuring into one folder, code shortening and path clean-up are NOT part of this file. They start only when the user types START-RESTRUCTURE after z1 has finished.

==============================================================================================
APPENDIX - LAUNCHER MESSAGE (for the user; start opencode in C:\Projects)
==============================================================================================
Read C:\Projects\MiniProject_Sem4\prompts\QSTUDY_PROMPT_MASTER11.md and execute it exactly. THIS_FILE11 = that file. Start with SECTION M0, then M1. Do not read any other prompt file in full.

STANDARD PREFACE for ANY future launcher message (put it first):
  FIRST, before anything else: run  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\qs10_lanes.py" ensure  and continue only when its last line is ALL LANES ACTIVE.
