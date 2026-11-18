QSTUDY BRIDGE 10 - RESUME CLEANUP7 AFTER THE SUMMARIES CRASH, PURGE SANDBOXES, LEGACY DEPENDENCY REPORT   (new executor model)
THIS_FILE10 = path given in the launcher message. This file only SEQUENCES the work; the commands come from QSTUDY_PROMPT_CLEANUP7.md and QSTUDY_PROMPT_AUDIT8.md.
Authority: this file > AUDIT8 > CLEANUP7 > CONTINUATION5 > earlier files. SECTION P0 (S1-S7, Q1-Q10) of CONTINUATION5 and SECTION C0 (W1-W11) of CLEANUP7 stay in force.
Read one section at a time:  python WORK\tools\secget.py <file> "SECTION D0" 80

==============================================================================================
SECTION D0 - RULES FOR THIS SESSION
==============================================================================================
X1  WORKING DIRECTORY is C:\Projects. Ignore any older rule that says the working directory must end in MiniProject_Sem4. Inside C:\Projects you may only touch:
    MiniProject_Sem4, MiniProject_Sem4_QSTUDY_WORK, _LEGACY_MiniProject_Sem4, and the prompt files. These are OUT OF BOUNDS: Birthday, Q-Quest, Quantum_VLSI, AI-Vault, GLM Chat Exports, .git.
X2  PROMPT FILES now live in  C:\Projects\MiniProject_Sem4\prompts\  (QSTUDY_PROMPT_*.md and QSTUDY_PONYTAIL_RULES.md). Whenever an older file names "C:\Projects\QSTUDY_PROMPT_<x>.md",
    read it from the prompts folder. Write the path "C:\Projects\MiniProject_Sem4\prompts" into RESUME_CARD.md.
X3  OUTPUT LIMIT: a single reply of yours is limited to about 8,000 tokens. Write files in pieces of at most 80 lines. Print at most 30 lines per command. Keep reasoning short.
X4  MODEL PILOT: you are a new model on this project. Steps D1.1 to D1.5 are checkpoints. If you invent a command that is not written in these files, edit a script, ignore a STOP, or fail the same step twice,
    write "MODEL PILOT FAILED" with the reason and stop. The user will then switch back to the previous model.
X5  Every PowerShell command starts with:  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1";
X6  Gates need the exact string from the user: GO-PURGE (D3), DECIDE-LEGACY (D4). Nothing is deleted without GO-PURGE. Sealed results stay blinded.

==============================================================================================
SECTION D1 - RESUME AND CHECK
==============================================================================================
1. Read the last 60 lines of C:\Projects\MiniProject_Sem4_QSTUDY_WORK\RESUME_CARD.md. Tell the user in 3 lines where CLEANUP7 stopped (it stopped at C5 step 2: qs7_summaries.py crashed with a seed collision).
2. . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; "$py | $w | $s | $root"      EXPECTED: four paths. (If env.ps1 is missing: STOP and tell the user.)
3. . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; & $py "$w\tools\brief.py" $w      EXPECTED: six heartbeats younger than 60 s, STALE_LANES "-". Otherwise P1 steps 6-7 of CONTINUATION5 for the stale lane only.
4. The fixed script is in place?  . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Select-String -Path "$w\tools\qs7_summaries.py" -Pattern "summary_" -Quiet      EXPECTED: True
   If False: STOP "FIXED SCRIPT NOT IN PLACE - the user must unzip qstudy_tools_v9.zip into C:\Projects\MiniProject_Sem4_QSTUDY_WORK".
   Also: . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; Get-ChildItem "$w\tools\qs9_*.py","$w\tools\qs8_*.py" | Select-Object Name      EXPECTED: qs9_legacy_refs.py, qs8_compare.py, qs8_scan.py, qs8_snapshot.py, qs8_tree.py
5. Status of the five re-queued jobs (CLEANUP7 C3 step 3 counter). Print the one line. No stop whatever the result; failed ones are handled in D5.

==============================================================================================
SECTION D2 - FINISH CLEANUP7 SECTION C5
==============================================================================================
Run, in this order, exactly as written in SECTION C5 of C:\Projects\MiniProject_Sem4\prompts\QSTUDY_PROMPT_CLEANUP7.md: step 2 (training table), step 3 (review pack check), step 4 (report A), step 5 (contents of the report).
Steps 1 of C5 (unit tests) is already DONE (138 tests OK, skipped=1): do not repeat it.
EXPECTED at step 2: "SUMMARIES runs_with_summary=<n> unreadable=0" and SUMMARIES-DONE. If it crashes again: print the last 15 lines and STOP.
Tell the user the path of RESULTS_REPORT_INTERIM_A.md and its line count in one line.

==============================================================================================
SECTION D3 - PURGE OF TEST SANDBOXES (user approval needed)
==============================================================================================
Run CLEANUP7 SECTION C4 step 3b: first the DRY RUN. EXPECTED candidates: MiniProject_Sem4_AUDIT, _AUDIT3, _SANDBOX, _SANDBOX2 (about 8 GB). The three folders MiniProject_Sem4_AUDIT2, _FIXRUN and _SANDBOX3
are not in the legacy folder (code or results still mention them) and must NOT appear. ARCHIVE, BACKUP and the zip are never purged by this command.
GATE G7b  Print "GATE-G7b REPORT" (the dry-run list). STOP. Continue only when the user types:  GO-PURGE
After GO-PURGE run the purge command of step 3b. EXPECTED: "PURGED n FAILED 0" and ROOT-PURGE-DONE. Then check the live project (brief.py, STALE_LANES "-"). Update RESUME_CARD.md.
If the user does not type GO-PURGE but another instruction, follow that instruction and skip D3.

==============================================================================================
SECTION D4 - LEGACY DEPENDENCY REPORT (read-only)
==============================================================================================
Three old folders are still referenced: AUDIT2 (by verification_tools\aggregate2.py), FIXRUN and SANDBOX3 (by result files). Before anything is done with them we need to know WHAT is referenced.
1. Run in the background and wait as in CLEANUP7 C2 steps 1-2 (output file audit\legacy_refs.out.txt, last line QS9-LEGACY-REFS-DONE, err file empty, up to 15 minutes):
   . "C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit\env.ps1"; $a4 = @("$w\tools\qs9_legacy_refs.py","--root",$root,"--scan-roots","C:\Projects\MiniProject_Sem4",$w,"--names","MiniProject_Sem4_AUDIT2","MiniProject_Sem4_FIXRUN","MiniProject_Sem4_SANDBOX3","--csv","$w\audit\legacy_refs.csv"); Start-Process -FilePath $py -ArgumentList $a4 -RedirectStandardOutput "$w\audit\legacy_refs.out.txt" -RedirectStandardError "$w\audit\legacy_refs.err.txt" -WindowStyle Hidden
2. GATE G7c  Print "GATE-G7c REPORT": the full text of legacy_refs.out.txt (max 40 lines) and the sizes of the three folders (the script prints them). Do NOT move or delete these three folders.
   STOP. Continue only when the user types:  DECIDE-LEGACY   (the controller decides what to keep after reading the report)

==============================================================================================
SECTION D5 - CONTINUE
==============================================================================================
1. Re-queued jobs: use the counter of CLEANUP7 C3 step 3 and the poll command of CONTINUATION5 P4.3 (sleep 600). A re-queued job that FAILED: apply X1-X4 of CONTINUATION5 once; second failure: STOP and report.
2. When all five are DONE run CLEANUP7 SECTION C6 steps 2-5 (output check, second scan with the three-restart command, tests again, final report, GATE G8).
3. Then run QSTUDY_PROMPT_AUDIT8.md (SECTION B1 onward; launcher message in its appendix, prompt path as in X2). Do not start any refactoring.
4. Standing instructions (CONTINUATION5): poll loop, WP1 coding authorised, nothing for z2a before GO-Z2A, WP2 only after GO-P6.

==============================================================================================
APPENDIX - LAUNCHER MESSAGE (for the user; start opencode in C:\Projects)
==============================================================================================
Read C:\Projects\MiniProject_Sem4\prompts\QSTUDY_PROMPT_BRIDGE10.md and execute it exactly. THIS_FILE10 = that file. Start with SECTION D0, then D1. Do not read any other prompt file in full.
