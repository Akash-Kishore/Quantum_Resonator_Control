# Paper status after the first full look at the interim report (controller notes, 2026-10-20)
Source: RESULTS_REPORT_INTERIM.md (13:18) and its appendix freeze_review_pack_v2.md. Everything below is from training-time validation scores and the pre-freeze validation split.
NO sealed (confirmatory) value has been looked at. Numbers are preliminary (5 seeds in the exploratory phase, 6 of 10 seeds in the confirmatory phase).

## 1. Where z1 stands
- Wave A training: 45 of 45 done (17.9 GPU hours, about 24 min per run). Wave B: 6 of 45 done when the report was built; more since. About 16-20 GPU hours remain if the lanes keep running.
- RL evaluations 152 of 270, baselines 45 of 45, Aer 26 of 26 (at 13:18). Sealed: 6981 condition files, 223 DONE markers, none opened.
- Five restarts so far (17 Oct 18:27, 18 Oct 06:14 and 07:32, 19 Oct 16:13, 20 Oct 05:21). One broken run folder was found and re-run. Disclose this in the methods.

## 2. What the validation data says (4-cell validation mean; nominal cell in brackets)
| probe | method | val mean (nominal) |
|---|---|---|
| any | oracle (ceiling) | 0.889 (0.957) |
| dither | PID | 0.879 (0.952) |
| dither | Bayes, assumed nominal | 0.881 (0.953) |
| dither | RL plain | 0.832 (0.909), seed sd 0.017 |
| dither | RL DRcurr / filter / full / DR | 0.780 / 0.764 / 0.551 / 0.453 |
| ramsey_iq | PID | 0.816 (0.904) |
| ramsey_iq | RL plain | 0.780 (0.899) |
| rabi (amplitude only) | hill climbing | 0.549 (0.616) |
| rabi | Bayes, assumed nominal | 0.540 (0.567) |
| rabi | RL plain / RL filter | 0.593 (0.663) / 0.635 (0.692) |
| rabi | no control | 0.129 (0.143) |

## 3. What this suggests (hypotheses, not findings)
1. When the probe gives a direction (dither, ramsey_iq), PID and Bayes are close to the ceiling and RL plain is a few points below. A claim "RL beats PID" is NOT supported.
2. When only amplitude is observed (rabi), RL plain and especially RL with a filter score above hill climbing and Bayes. That is the regime where learning could add value. Gaps are 0.04 to 0.09 with seed sd 0.05 to 0.07, so it needs the 10-seed confirmatory data.
3. Domain-randomised and "full" configurations score lower on the same nominal cells. Training seeds of RL plain on dither are very consistent (0.82 to 0.85 over 6 seeds); no collapsed seed so far.
4. The confirmatory design does not contain the rabi comparison (RL vs hill, RL vs Bayes). Contrasts C-1 to C-8 only compare RL with PID and Bayes on dither.

## 4. Proposed addendum (needs the user's OK; must be written and hashed BEFORE any unblinding)
- C-9: rl__rabi__plain versus ctrl:hill on rabi, seeds 101-110, same sealed seeds.
- C-10: rl__rabi__plain versus ctrl:bayes:nominal on rabi, seeds 101-110.
- Both are motivated by validation data and will be tested on sealed data that nobody has seen. Declare that motivation in the paper. Holm correction over all 10 contrasts.
- The sealed data for these baselines already exists (z1 baselines 45 of 45), so no new runs are needed.

## 5. Old numbers that must not be cited without re-verification
Anomalies A1-A4 in events.md: the stored files for the earlier study (V4 Seed 3 MAE 1510 Hz / 94.8 percent near resonance, QACT v5 34.3 percent) disagree with the documented numbers, one verification folder is corrupt, and an old PID baseline was unseeded. The paper should rest on z1.

## 6. Open items
- Determinism probe: the first comparison was a tool error (files were never matched). Re-run with qs8_compare.py --match-after __.
- Legacy folders: keep FIXRUN (4.7 MB) and AUDIT2 (141 MB); inspect SANDBOX3 (6.7 GB) before deciding.
- Never delete rl_training\trained_models: it holds the original study models. The scan flagged them only because of an older folder layout.
- Restructure into one folder, hard-coded path removal, code shortening: after z1 finishes (AS_RUN_v1.zip, sha256 a6c06b08db4d3b73..., is the as-run record).
