"""TR-1 / TR-2 transfer analysis (PREREG2 S15).
P2T = macro-average lock_w1000 over RB OFAT cells (34 cells, 300 eps each) per seed.
Contrasts (Holm m=5):
  TR-1: E1 (LG zero-shot) vs E4 final 2M (RB from scratch) on P2T
  TR-2: FTB_200000 (LG+FT 200k) vs E4 200k snapshot on P2T
Stats: Mann-Whitney U exact two-sided (n=5v5), Cliff's delta, bootstrap 95% CI
       of mean difference (10000 resamples), Holm m=5.
Output: WORK\results\tr1_tr2_report.json (+ printed tables).
Read-only: reads results\sealed only; writes only under WORK.
"""
import glob, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats as sps

SEALED = r"C:\Projects\MiniProject_Sem4\qstudy\results\sealed"
WORK_RES = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\results"
OUT = os.path.join(WORK_RES, "tr1_tr2_report.json")
SEEDS = [200, 201, 202, 203, 204]
STEPS = [50000, 200000, 500000, 1000000]
BOOT_SEED = 20261012
N_BOOT = 10000
HOLM_M = 5


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def p2t_for_arm_seed(arm_tag, seed, step=None):
    """Macro-average lock_w1000 over RB OFAT cells for one (arm, seed, step).
    arm_tag: 'z2__s15__E1' etc. (z2 results, final model)
             'z2s__s15__E4' etc. (z2s results, per-step snapshot)
    Returns (p2t, n_cells, file_list).
    """
    if step is None:
        prefix = "z2__%s_s%d__RB__ofat_" % (arm_tag, seed)
    else:
        prefix = "z2s__%s_s%d__RB__s%d__ofat_" % (arm_tag, seed, step)
    files = []
    for fn in os.listdir(SEALED):
        if fn.startswith(prefix) and fn.endswith(".json") and not fn.endswith(".DONE.json"):
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
        return float("nan"), 0, []
    return float(np.mean(vals)), len(vals), files


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
    """Holm step-down adjusted p-values with family size m (may exceed len(pvals))."""
    order = np.argsort(pvals)
    k = len(pvals)
    adj = np.empty(k)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def main():
    os.makedirs(WORK_RES, exist_ok=True)

    # --- Per-seed P2T for each arm ---
    arms = {
        "TR1_A_E1_LG_zero_shot":    ("s15__E1", None),
        "TR1_B_E4_RB_2M_final":     ("s15__E4", None),
        "TR2_A_FTB_200k":           ("s15__FTB_200000", None),
        "TR2_B_E4_200k_snapshot":   ("s15__E4", 200000),
    }
    p2t = {}  # arm -> {seed: p2t}
    for name, (tag, step) in arms.items():
        p2t[name] = {}
        for seed in SEEDS:
            v, n, _ = p2t_for_arm_seed(tag, seed, step)
            p2t[name][seed] = v

    # --- Contrasts ---
    contrasts = [
        ("TR-1", "E1 LG zero-shot vs E4 RB-from-scratch 2M",
         p2t["TR1_A_E1_LG_zero_shot"], p2t["TR1_B_E4_RB_2M_final"]),
        ("TR-2", "FTB_200k (LG+FT 200k) vs E4 RB 200k snapshot",
         p2t["TR2_A_FTB_200k"], p2t["TR2_B_E4_200k_snapshot"]),
    ]

    rows = []
    pvals = []
    for label, desc, a_dict, b_dict in contrasts:
        a = [a_dict[s] for s in SEEDS]
        b = [b_dict[s] for s in SEEDS]
        u, p = mannwhitney_exact(a, b)
        cd = cliffs_delta(a, b)
        lo, hi = boot_ci_diff(a, b)
        diff = float(np.mean(a) - np.mean(b))
        rows.append({
            "contrast": label,
            "description": desc,
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
            "seeds_a": {str(s): a_dict[s] for s in SEEDS},
            "seeds_b": {str(s): b_dict[s] for s in SEEDS},
        })
        pvals.append(p)

    # --- Holm m=5 ---
    adj = holm(pvals, HOLM_M)
    for i, r in enumerate(rows):
        r["p_holm_m5"] = adj[i]
        r["holm_significant_m5"] = bool(adj[i] < 0.05)

    report = {
        "prereg": "PREREG2 S15 Transfer",
        "endpoint": "P2T = macro-average lock_w1000 over RB OFAT cells (34 cells x 300 eps)",
        "seeds": SEEDS,
        "holm_m": HOLM_M,
        "boot_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "p2t_by_arm_seed": {k: {str(s): v for s, v in d.items()} for k, d in p2t.items()},
        "contrasts": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote", OUT)

    # --- Print tables ---
    print("\n=== P2T per seed (macro-avg lock_w1000 over RB OFAT cells) ===")
    hdr = ["seed"] + list(arms.keys())
    print(" ".join("%-12s" % c for c in hdr))
    for s in SEEDS:
        print(" ".join("%-12s" % str(s)) + " " +
              " ".join("%-12.4f" % p2t[name][s] for name in arms))

    print("\n=== TR-1 / TR-2 contrasts (Holm m=5) ===")
    print("%-6s %-45s %-12s %-12s %-12s %-10s %-10s %-10s %-10s %-10s"
            % ("ID", "Description", "meanA", "meanB", "diff", "U", "p_exact", "p_holm", "Cliff", "sig"))
    for r in rows:
        print("%-6s %-45s %-12.4f %-12.4f %-12.4f %-10s %-10g %-10g %-10.3f %-10s"
              % (r["contrast"], r["description"][:45], r["arm_a_mean"], r["arm_b_mean"],
                 r["mean_diff_a_minus_b"], r["u"], r["p_exact"], r["p_holm_m5"],
                 r["cliffs_delta"], r["holm_significant_m5"]))

    print("\n=== Per-seed detail ===")
    for r in rows:
        print("\n%s: %s" % (r["contrast"], r["description"]))
        print("  Arm A seeds:", {str(s): round(r["seeds_a"][str(s)], 4) for s in SEEDS})
        print("  Arm B seeds:", {str(s): round(r["seeds_b"][str(s)], 4) for s in SEEDS})
        print("  U=%s p_exact=%.4g Cliff=%.3f boot_CI=[%.4f, %.4f] p_holm_m5=%.4g sig=%s"
              % (r["u"], r["p_exact"], r["cliffs_delta"], r["boot_ci_lo"], r["boot_ci_hi"],
                 r["p_holm_m5"], r["holm_significant_m5"]))


if __name__ == "__main__":
    main()
