# Table and Figure Captions

## T4_transfer_matrix.csv / FB_transfer_heatmap_4x4 (Figure F-B)
**Transfer Matrix: Mean P2T ± SD (5 seeds per arm)**

Rows = training environment (E1, E2, E3, E4). Columns = evaluation cell (LG, LB, RG, RB).
Values are macro-average lock_w1000 over RB OFAT cells (34 cells × 300 episodes), averaged over 5 seeds (200–204).
**Important naming convention**: The row label (e.g., E1) means "arm trained for that environment" (e.g., E1 = LG zero-shot). The column label (e.g., LG) means "evaluated on that environment's P2T cells" — but ALL evaluations are P2T on RB OFAT, NOT on the column's native environment. So A_LG means "E1 arm (trained for LG) evaluated via P2T on RB OFAT". This naming convention caused a verification error earlier in the project and is stated explicitly here so a reader doesn't make the same misreading.

## T6_gap_decomposition.csv
**Gap Decomposition (Exploratory)**

Decomposition of the E1→E4 transfer gap on RB P2T into three effects (lineshape, noise, interaction), per QSTUDY_PROMPT_TIERB14.md lines 39-42. n=5 seeds per arm (200–204). CIs are bootstrap (10,000 resamples, seed 20261012). **No hypothesis test attached** (exploratory only, per P3 spec).

- lineshape_effect = mean(A_RG - A_LG, A_RB - A_LB)
- noise_effect = mean(A_LB - A_LG, A_RB - A_RG)
- interaction = (A_RB - A_RG) - (A_LB - A_LG)

**Important**: A_LG/A_LB/A_RG/A_RB denote "arm trained for that env, evaluated via P2T on RB OFAT" — NOT "evaluated on that env". Same naming convention as T4.

## T7_TR4.csv
**TR-4: E2 LB zero-shot vs E1 LG zero-shot on P2T**

Paired two-sided Mann-Whitney U, Cliff's delta, bootstrap 95% CI (10,000, seed 20261012), Holm m=5 (family: TR-1, TR-2, TR-3=PENDING, TR-4, TR-5=PENDING). p_holm=0.0476 (significant). Endpoint: P2T = macro-average lock_w1000 over RB OFAT (34 cells × 300 eps). Seeds 200–204.