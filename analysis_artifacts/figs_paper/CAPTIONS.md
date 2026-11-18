# Figure Captions (Draft, Facts Only)

## F-A: Z1 P2 per config (10 seeds) with baselines
Bar heights: mean P2 final_val over 10 seeds (9 RL configs). Error bars: ±1 SD. Horizontal dashed lines: baseline methods (oracle, bayes oracle, PID, bayes nominal, no control) evaluated on same env. Source: training_health_report.json (aggregated from stored per-seed values). Exploratory.

## F-B: Transfer matrix heatmap
Nominal lock_w1000 (mean over 5 seeds) for E1-trained and E4-trained policies evaluated on LG, LB, RG, RB. Source: sealed nominal RB evaluations (aggregated from stored per-seed values). Exploratory.

## F-C: Sample efficiency curves
Nominal RB lock_w1000 vs training steps in quantum env. FTB (classical pretrain + FT) vs E4 (quantum scratch). Shaded/ error bars: ±1 SD over 5 seeds. Claim: fewer quantum-env steps for FTB, not less total compute. Source: sealed nominal RB evaluations at 50k/200k/500k/1M (aggregated from stored per-seed values). Exploratory.
