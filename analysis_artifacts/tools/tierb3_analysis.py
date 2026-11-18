#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TIERB-3: TR-4 and Gap Decomposition analysis.
TR-4: LB zero-shot (E2) vs LG zero-shot (E1) on P2T, n=5 per arm, exact MWU, Cliff's delta, bootstrap CI (10000, seed 20261012).
Holm family size 5: TR-1, TR-2, TR-3(PENDING=1.0), TR-4, TR-5(PENDING=1.0).
Gap decomposition (exploratory, P3): lineshape effect, noise effect, interaction, 4x4 transfer matrix.
"""
import json, os, numpy as np
from scipy import stats as sps

SEALED = r"C:\Projects\MiniProject_Sem4\qstudy\results\sealed"
WORK_RES = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\results"
SEEDS = [200, 201, 202, 203, 204]
BOOT_SEED = 20261012
N_BOOT = 10000
HOLM_M = 5

def mannwhitney_exact(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        u, p = sps.mannwhitneyu(a, b, alternative="two-sided", method="exact")
    except Exception:
        u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)

def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    gt = sum(int((x > b).sum()) for x in a)
    lt = sum(int((x < b).sum()) for x in a)
    return float((gt - lt) / (len(a) * len(b)))

def boot_ci_diff(a, b, seed=BOOT_SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    diffs = []
    for _ in range(n):
        ra = rng.choice(a, len(a), replace=True)
        rb = rng.choice(b, len(b), replace=True)
        diffs.append(float(ra.mean() - rb.mean()))
    diffs = sorted(diffs)
    return float(diffs[int(0.025 * n)]), float(diffs[int(0.975 * n)])

def holm(pvals, m):
    order = np.argsort(pvals)
    k = len(pvals)
    adj = np.empty(k)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()

def p2t_for_arm_seed(prefix, seed):
    """Macro-average lock_w1000 over RB OFAT cells for one (arm, seed).
    prefix: e.g., 'z2h__s15__E2_s' or 'z2__s15__E1_s'
    Returns (p2t, n_cells).
    """
    full_prefix = "%s%d__RB__ofat_" % (prefix, seed)
    files = []
    for fn in os.listdir(SEALED):
        if fn.startswith(full_prefix) and fn.endswith(".json") and not fn.endswith(".DONE.json"):
            files.append(fn)
    files.sort()
    vals = []
    for fn in files:
        fp = os.path.join(SEALED, fn)
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("status") != "ok":
            continue
        vals.append(float(d["lock_w1000_mean"]))
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)

def nominal_lock(arm_tag, seed, env):
    """Nominal lock_w1000 for one (arm, seed, env).
    arm_tag: 'z2h__s15__E2' or 'z2__s15__E1' etc.
    env: 'LG', 'LB', 'RG', 'RB'
    """
    fn = "%s_s%d__%s__nom.json" % (arm_tag, seed, env)
    fp = os.path.join(SEALED, fn)
    if not os.path.exists(fp):
        return float("nan")
    with open(fp, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("status") != "ok":
        return float("nan")
    return float(d["lock_w1000_mean"])

def main():
    os.makedirs(WORK_RES, exist_ok=True)

    # --- TR-4: E2 (LB zero-shot) vs E1 (LG zero-shot) on P2T ---
    # E2 prefix: z2h__s15__E2_s
    # E1 prefix: z2__s15__E1_s
    e2_p2t = {}
    e1_p2t = {}
    for s in SEEDS:
        v2, n2 = p2t_for_arm_seed("z2h__s15__E2_s", s)
        v1, n1 = p2t_for_arm_seed("z2__s15__E1_s", s)
        e2_p2t[s] = v2
        e1_p2t[s] = v1
        print("Seed %d: E2 P2T=%.4f (%d cells), E1 P2T=%.4f (%d cells)" % (s, v2, n2, v1, n1))

    a = [e2_p2t[s] for s in SEEDS]
    b = [e1_p2t[s] for s in SEEDS]
    u, p = mannwhitney_exact(a, b)
    cd = cliffs_delta(a, b)
    lo, hi = boot_ci_diff(a, b)
    diff = float(np.mean(a) - np.mean(b))

    # Holm family: TR-1, TR-2, TR-3(1.0), TR-4, TR-5(1.0) -> m=5
    pvals = [0.007937, 0.007937, 1.0, p, 1.0]  # TR-1, TR-2 exact p from S7
    adj = holm(pvals, HOLM_M)
    tr4_p_holm = adj[3]  # TR-4 is 4th in order
    tr4_sig = bool(tr4_p_holm < 0.05)

    print("\n=== TR-4 ===")
    print("E2 (LB zero-shot) vs E1 (LG zero-shot) on P2T")
    print("Arm A (E2) mean=%.4f sd=%.4f" % (np.mean(a), np.std(a, ddof=1)))
    print("Arm B (E1) mean=%.4f sd=%.4f" % (np.mean(b), np.std(b, ddof=1)))
    print("diff=%.4f U=%s p_exact=%.6g Cliff=%.3f boot_CI=[%.4f, %.4f] p_holm_m5=%.6g sig=%s"
          % (diff, u, p, cd, lo, hi, tr4_p_holm, tr4_sig))

    # --- 4x4 Transfer Matrix (nominal lock_w1000) ---
    # Rows: trained on (E1=LG, E2=LB, E3=RG, E4=RB)
    # Cols: evaluated on (LG, LB, RG, RB)
    arms = {
        "E1": ("z2__s15__E1", None),
        "E2": ("z2h__s15__E2", None),
        "E3": ("z2h__s15__E3", None),
        "E4": ("z2__s15__E4", None),
    }
    eval_envs = ["LG", "LB", "RG", "RB"]

    matrix_mean = {}
    matrix_sd = {}
    matrix_seeds = {}

    print("\n=== 4x4 Transfer Matrix (nominal lock_w1000) ===")
    hdr = ["trained\\eval"] + eval_envs
    print(" ".join("%-12s" % c for c in hdr))
    for arm_name, (tag, step) in arms.items():
        matrix_mean[arm_name] = {}
        matrix_sd[arm_name] = {}
        matrix_seeds[arm_name] = {}
        row_vals = [arm_name]
        for env in eval_envs:
            vals = []
            for s in SEEDS:
                v = nominal_lock(tag, s, env)
                vals.append(v)
                matrix_seeds[arm_name].setdefault(env, {})[str(s)] = v
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            matrix_mean[arm_name][env] = m
            matrix_sd[arm_name][env] = sd
            row_vals.append("%.4f+/-%.4f" % (m, sd))
        print(" ".join("%-12s" % c for c in row_vals))

    # --- Gap Decomposition (P3) ---
    # A_LG = E1, A_LB = E2, A_RG = E3, A_RB = E4 (all on RB OFAT P2T)
    # Need P2T for E3 and E4 as well
    e3_p2t = {}
    e4_p2t = {}
    for s in SEEDS:
        v3, _ = p2t_for_arm_seed("z2h__s15__E3_s", s)
        v4, _ = p2t_for_arm_seed("z2__s15__E4_s", s)
        e3_p2t[s] = v3
        e4_p2t[s] = v4

    A_LG = [e1_p2t[s] for s in SEEDS]
    A_LB = [e2_p2t[s] for s in SEEDS]
    A_RG = [e3_p2t[s] for s in SEEDS]
    A_RB = [e4_p2t[s] for s in SEEDS]

    print("\n=== P2T per seed ===")
    print("seed  E1(LG)  E2(LB)  E3(RG)  E4(RB)")
    for s in SEEDS:
        print("%d  %.4f  %.4f  %.4f  %.4f" % (s, e1_p2t[s], e2_p2t[s], e3_p2t[s], e4_p2t[s]))

    # lineshape effect = mean(A_RG - A_LG, A_RB - A_LB)
    # noise effect = mean(A_LB - A_LG, A_RB - A_RG)
    # interaction = (A_RB - A_RG) - (A_LB - A_LG)

    # Compute per-seed effects
    ls_eff = [(A_RG[i] - A_LG[i] + A_RB[i] - A_LB[i]) / 2.0 for i in range(5)]
    noise_eff = [(A_LB[i] - A_LG[i] + A_RB[i] - A_RG[i]) / 2.0 for i in range(5)]
    interact = [(A_RB[i] - A_RG[i]) - (A_LB[i] - A_LG[i]) for i in range(5)]

    ls_mean = float(np.mean(ls_eff))
    ls_sd = float(np.std(ls_eff, ddof=1))
    noise_mean = float(np.mean(noise_eff))
    noise_sd = float(np.std(noise_eff, ddof=1))
    interact_mean = float(np.mean(interact))
    interact_sd = float(np.std(interact, ddof=1))

    # Bootstrap CIs for effects
    def boot_ci_effects(seed=BOOT_SEED, n=N_BOOT):
        rng = np.random.default_rng(seed)
        ls_boot = []
        noise_boot = []
        int_boot = []
        for _ in range(n):
            # Resample 5 seeds independently within each arm
            idx = rng.choice(5, 5, replace=True)
            A_LG_b = [A_LG[i] for i in idx]
            A_LB_b = [A_LB[i] for i in idx]
            A_RG_b = [A_RG[i] for i in idx]
            A_RB_b = [A_RB[i] for i in idx]
            ls_b = [(A_RG_b[i] - A_LG_b[i] + A_RB_b[i] - A_LB_b[i]) / 2.0 for i in range(5)]
            noise_b = [(A_LB_b[i] - A_LG_b[i] + A_RB_b[i] - A_RG_b[i]) / 2.0 for i in range(5)]
            int_b = [(A_RB_b[i] - A_RG_b[i]) - (A_LB_b[i] - A_LG_b[i]) for i in range(5)]
            ls_boot.append(float(np.mean(ls_b)))
            noise_boot.append(float(np.mean(noise_b)))
            int_boot.append(float(np.mean(int_b)))
        ls_boot = sorted(ls_boot)
        noise_boot = sorted(noise_boot)
        int_boot = sorted(int_boot)
        return (ls_boot[int(0.025*n)], ls_boot[int(0.975*n)]), \
               (noise_boot[int(0.025*n)], noise_boot[int(0.975*n)]), \
               (int_boot[int(0.025*n)], int_boot[int(0.975*n)])

    ls_ci, noise_ci, int_ci = boot_ci_effects()

    print("\n=== Gap Decomposition (P3) ===")
    print("Lineshape effect: mean=%.4f sd=%.4f boot_CI=[%.4f, %.4f]" % (ls_mean, ls_sd, ls_ci[0], ls_ci[1]))
    print("Noise effect:     mean=%.4f sd=%.4f boot_CI=[%.4f, %.4f]" % (noise_mean, noise_sd, noise_ci[0], noise_ci[1]))
    print("Interaction:      mean=%.4f sd=%.4f boot_CI=[%.4f, %.4f]" % (interact_mean, interact_sd, int_ci[0], int_ci[1]))
    print("Per-seed effects:")
    for i, s in enumerate(SEEDS):
        print("  seed %d: ls=%.4f noise=%.4f int=%.4f" % (s, ls_eff[i], noise_eff[i], interact[i]))

    # --- TR-1/TR-2 sanity check ---
    print("\n=== TR-1/TR-2 Sanity Check ===")
    # TR-1: E1 vs E4 (final 2M)
    tr1_a = [e1_p2t[s] for s in SEEDS]
    tr1_b = [e4_p2t[s] for s in SEEDS]
    u1, p1 = mannwhitney_exact(tr1_a, tr1_b)
    diff1 = float(np.mean(tr1_a) - np.mean(tr1_b))
    print("TR-1 (E1 vs E4 2M): diff=%.4f p=%.6g (expected diff=-0.3136 p=0.00794)" % (diff1, p1))

    # TR-2: FTB_200k vs E4 200k snapshot
    ftb200_p2t = {}
    e4_200k_p2t = {}
    for s in SEEDS:
        v_ftb, _ = p2t_for_arm_seed("z2s__s15__FTB_200000_s", s) if False else p2t_for_arm_seed("z2__s15__FTB_200000_s", s)
        # Actually FTB evals are z2f__ for last-step and z2s__ for snapshots
        # Let me check what's available
        pass

    # Save report
    report = {
        "spec_citation": {
            "TR-4": "QSTUDY_PROMPT_TIERB14.md lines 36-38 (P2)",
            "Gap_Decomposition": "QSTUDY_PROMPT_TIERB14.md lines 39-42 (P3)",
        },
        "TR-4": {
            "contrast": "TR-4",
            "description": "E2 LB zero-shot vs E1 LG zero-shot on P2T",
            "arm_a": "E2 (env LB)",
            "arm_b": "E1 (env LG)",
            "endpoint": "P2T = macro-average lock_w1000 over RB OFAT cells (34 cells x 300 eps)",
            "seeds": SEEDS,
            "arm_a_mean": float(np.mean(a)),
            "arm_a_sd": float(np.std(a, ddof=1)),
            "arm_b_mean": float(np.mean(b)),
            "arm_b_sd": float(np.std(b, ddof=1)),
            "mean_diff_a_minus_b": diff,
            "u": u,
            "p_exact": p,
            "cliffs_delta": cd,
            "boot_ci_lo": lo,
            "boot_ci_hi": hi,
            "holm_m": HOLM_M,
            "p_holm_m5": tr4_p_holm,
            "holm_significant_m5": tr4_sig,
            "seeds_a": {str(s): e2_p2t[s] for s in SEEDS},
            "seeds_b": {str(s): e1_p2t[s] for s in SEEDS},
        },
        "transfer_matrix_nominal": {
            "rows": ["E1", "E2", "E3", "E4"],
            "cols": eval_envs,
            "mean": matrix_mean,
            "sd": matrix_sd,
            "seeds": matrix_seeds,
            "note": "E1 and E4 from prior z2a/z2s evals; E2 and E3 NEW from z2h_ evals",
        },
        "gap_decomposition": {
            "lineshape_effect": {"mean": ls_mean, "sd": ls_sd, "boot_ci_lo": ls_ci[0], "boot_ci_hi": ls_ci[1], "per_seed": {str(s): ls_eff[i] for i, s in enumerate(SEEDS)}},
            "noise_effect": {"mean": noise_mean, "sd": noise_sd, "boot_ci_lo": noise_ci[0], "boot_ci_hi": noise_ci[1], "per_seed": {str(s): noise_eff[i] for i, s in enumerate(SEEDS)}},
            "interaction": {"mean": interact_mean, "sd": interact_sd, "boot_ci_lo": int_ci[0], "boot_ci_hi": int_ci[1], "per_seed": {str(s): interact[i] for i, s in enumerate(SEEDS)}},
            "note": "Exploratory only; n=5 seeds per arm -> CIs wide; no hypothesis test attached (per P3).",
        },
    }

    out_path = os.path.join(WORK_RES, "tierb3_tr4_gap.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\nwrote", out_path)

    # Print TR-4 row for db insertion
    tr4_row = {
        "contrast_id": "TR-4",
        "endpoint": "P2T",
        "arm_a": "E2_LB_zero_shot",
        "arm_b": "E1_LG_zero_shot",
        "n_a": 5,
        "n_b": 5,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": diff,
        "u": u,
        "p_exact": p,
        "cliffs_delta": cd,
        "boot_ci_lo": lo,
        "boot_ci_hi": hi,
        "holm_m": HOLM_M,
        "p_holm": tr4_p_holm,
        "status": "FINAL" if tr4_sig else "NOT_SIGNIFICANT",
    }
    print("\n=== TR-4 DB Row ===")
    print(json.dumps(tr4_row, indent=1))

    return tr4_row

if __name__ == "__main__":
    main()