# VERIFY-FINAL Report
**Date:** 2026-11-18
**Workspace:** C:\Projects\MiniProject_Sem4_QSTUDY_WORK

---

## Summary Table

| Step | Check | Verdict | Reason |
|------|-------|---------|--------|
| 1 | Code Freeze | PASS | All 6 pipeline files match AS_RUN_v1.zip hashes exactly. Original six mtimes ≤ 17-10-2026 (before Z1 seal 25-10-2026). New analysis tools post-Z1 listed and expected; none of the original six are newer. |
| 2 | Preregistration Integrity | PASS* | PREREG.md (7B6236DB7681DDB5637C3FCD8FBC57339A6A428AEA2A12945EE8C2CBE0E8DD8E) and PREREG2.md (871729E2DDFA488B5490C706C7A0C4FB6CF7333E5DE87C9C8C395975A1A4236A) hashes match .sha256 files. FLAG: 7 files in configs\frozen (s15__E1/E4/FTA_200000/FTB_*.json) have mtime 23-10-2026 > earliest sealed 17-10-2026 — these are Z2A/TierB configs added after Z1 seal; Z1 frozen configs (rl__*, selection.json) mtime 15-17 Oct are before seal. |
| 3 | Sealed Results Integrity | PASS | File count 30,106 (expected ~30,106). Prefix breakdown: z1=22035, z2=3240, z2s=1620, z2f=1620, z2h=810, z1aer=346, z2aer=273, z2r=162. 300 random sample: 276/300 match 26-Oct manifest (exact SHA256). 24 new files post-26-Oct: z2f=20, z2h=3, z2r=1 (proportions consistent with z2f_=100 jobs, z2h_=50 jobs, z2r_=10 jobs). 20 spot-check mtime vs creation: all diffs <1s (max 0.39s). Zero hangtest* files in sealed. |
| 4 | Database State | PASS | 21 rows in contrasts. C-1..C-8: each exactly 2 rows (1×SUPERSEDED n=5 + 1×FINAL n=10 confirmatory). TR-1 FINAL (2026-10-26), TR-2 FINAL (2026-10-26), TR-4 FINAL (2026-11-14 TIERB-3). TR-3 & TR-5: exactly 1 each, status=PENDING_ARMS_NOT_RUN, NULL statistic/p_value/effect_size/ci. Zero sens2/sens2r rows. TR-4 recorded on TIERB-3 completion day (expected). 19 tables, no unexpected new tables. |
| 5 | Queue & Lane State | PASS | brief.py: DONE=1881, RUNNING=0, PENDING=0, NEW_FAILED=-, STALE_LANES=- (matches TIERB-2 completion). Exactly 6 job_runner.py processes: gpu(7412), aer(17960), cpu1(6580), cpu2(18580), cpu3(17852), cpu4(8604). All queue prefixes legitimate per project history. |
| 6 | GO-CLEANUP Side Effects | PASS | NESTED folder (C:\Projects\MiniProject_Sem4\MiniProject_Sem4_QSTUDY_WORK) gone. Stray $w\events.md gone. Both archived unmodified at $w\archive\pre_cleanup\20261118_044743\ (SHA256 verified against backup). Full backup zips exist non-zero: MiniProject_Sem4.zip (5.76 GB), MiniProject_Sem4_QSTUDY_WORK.zip (46.68 MB). |
| 7 | Log/Card Consistency | PASS | events.md: no dangling "CORRECTION" after "DONE" without resolution. DECISION PENDING (thread-cap) remains open, correctly labeled as user decision. RESUME_CARD.md final entry GO-CLEANUP DONE; implicit next step = VERIFY-FINAL completion. Coherent. |

*FLAG noted but not a failure — expected Z2A/TierB config additions post-Z1-seal.

---

## Overall Verdict

**CLEAN — SAFE TO START PAPER**

All 7 steps pass. The FLAG on Step 2 is documented and explained (Z2A/TierB configs legitimately added after Z1 seal; Z1 frozen configs remain pristine). No critical integrity violations found. The codebase, sealed results, database, queue, and logs are in a consistent, auditable state suitable for paper writing.