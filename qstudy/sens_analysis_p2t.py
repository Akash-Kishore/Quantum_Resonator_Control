"""SENS-2 on P2T: FTB_200000 raw last-step vs E4 200k raw snapshot.
P2T = macro-average lock_w1000 over 34 OFAT cells on RB probe, per seed.
Exact two-sided Mann-Whitney U, Cliff's delta, bootstrap 95% CI.
Stored as SENSITIVITY exploratory=1 (never FINAL). Not part of Holm family.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats import mannwhitney_exact, cliffs_delta, boot_ci_diff

SEALED = os.path.join(HERE, "results", "sealed")
WORK = os.path.join(os.path.dirname(HERE), "MiniProject_Sem4_QSTUDY_WORK")
OUTDIR = os.path.join(WORK, "results")
os.makedirs(OUTDIR, exist_ok=True)
SEEDS = [200, 201, 202, 203, 204]
BOOT_SEED = 20261012
N_BOOT = 10000


def p2t_ofat(seed, prefix):
    """Macro-average lock_w1000 over 34 OFAT cells for a given seed and model prefix."""
    vals = []
    pattern = os.path.join(SEALED, prefix.format(seed=seed) + "__RB__s200000__ofat_*.json")
    files = sorted(glob.glob(pattern))
    cell_files = [f for f in files if "__ofat.DONE.json" not in f]
    if len(cell_files) != 34:
        print(f"WARNING seed {seed} prefix {prefix}: {len(cell_files)} cell files (expected 34)")
    for f in cell_files:
        with open(f) as fh:
            d = json.load(fh)
        if d.get("status") == "ok":
            vals.append(d["lock_w1000_mean"])
    return float(np.mean(vals)) if vals else np.nan


def main():
    # FTB_200000_last: z2f__s15__FTB_200000_last_s{seed}__RB__s200000__ofat_*.json
    # E4_200k: z2s__s15__E4_s{seed}__RB__s200000__ofat_*.json
    ftb_prefix = "z2f__s15__FTB_200000_last_s{seed}"
    e4_prefix = "z2s__s15__E4_s{seed}"

    ftb_p2t = {}
    e4_p2t = {}
    for s in SEEDS:
        ftb_p2t[s] = p2t_ofat(s, ftb_prefix)
        e4_p2t[s] = p2t_ofat(s, e4_prefix)

    a = [ftb_p2t[s] for s in SEEDS]
    b = [e4_p2t[s] for s in SEEDS]

    u, p = mannwhitney_exact(a, b)
    cd = cliffs_delta(a, b)
    lo, hi = boot_ci_diff(a, b, seed=BOOT_SEED, n=N_BOOT)

    print("=" * 90)
    print("SENS-2 on P2T: FTB_200000 raw last-step vs E4 200k raw snapshot")
    print("Endpoint: P2T = macro-average lock_w1000 over 34 OFAT cells on RB")
    print("Seeds:", SEEDS, "| Exact Mann-Whitney U | Cliff's delta | Bootstrap 95% CI (seed 20261012, n=10000)")
    print("=" * 90)
    print()
    print(f"{'Seed':<6} {'FTB_200k P2T':>14} {'E4_200k P2T':>14} {'Diff':>10}")
    print("-" * 90)
    for s in SEEDS:
        print(f"{s:<6} {ftb_p2t[s]:>14.4f} {e4_p2t[s]:>14.4f} {ftb_p2t[s]-e4_p2t[s]:>10.4f}")
    print()
    print(f"FTB_200k mean P2T: {np.mean(a):.4f}  sd: {np.std(a, ddof=1):.4f}")
    print(f"E4_200k mean P2T:  {np.mean(b):.4f}  sd: {np.std(b, ddof=1):.4f}")
    print(f"Mean diff (FTB - E4): {np.mean(a) - np.mean(b):.4f}")
    print(f"U = {u:.1f}, p_exact = {p:.6f}")
    print(f"Cliff's delta = {cd:.3f}")
    print(f"Bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
    print()
    print("TR-2 FINAL row (side-by-side):")
    print(f"FTB_200k mean P2T: 0.4104  E4_200k mean P2T: 0.2658  Diff: 0.1446")
    print(f"U = 25.0, p_exact = 0.00793651, Cliff's delta = 1.000")
    print(f"Bootstrap 95% CI: [0.1066, 0.1845]")
    print()
    # Pre-declared interpretation rule
    ft_adv = np.mean(a) > np.mean(b)
    sig = p < 0.05
    print("PRE-DECLARED INTERPRETATION RULE (verbatim):")
    print("  'if the FT advantage remains (same direction, exact p < 0.05 unadjusted)")
    print("   the TR-2 selection asymmetry is closed as a limitation;'")
    print(f"  FT advantage remains (FTB > E4): {ft_adv}")
    print(f"  exact p < 0.05 (unadjusted): {sig} (p={p:.6f})")
    if ft_adv and sig:
        print("  => TR-2 selection asymmetry CLOSED as a limitation.")
    else:
        print("  => Rule NOT met; both numbers reported; paper says checkpoint selection contributed.")
    print()
    print("STATUS: SENSITIVITY exploratory=1 (never FINAL). Nominal-proxy result marked PROXY.")
    print("Nothing written to db.contrasts.")

    # Save results
    out = {
        "sens": "SENS-2",
        "endpoint": "P2T",
        "comparison": "FTB_200000 raw last-step vs E4 200k raw snapshot",
        "p2t_definition": "macro-average lock_w1000 over 34 OFAT cells on RB probe",
        "seeds": SEEDS,
        "ftb_p2t": {str(s): ftb_p2t[s] for s in SEEDS},
        "e4_p2t": {str(s): e4_p2t[s] for s in SEEDS},
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "sd_a": float(np.std(a, ddof=1)),
        "sd_b": float(np.std(b, ddof=1)),
        "u": float(u),
        "p_exact": float(p),
        "cliffs_delta": float(cd),
        "boot_ci_lo": float(lo),
        "boot_ci_hi": float(hi),
        "tr2_final_mean_a": 0.41041405228758177,
        "tr2_final_mean_b": 0.2658057734204793,
        "tr2_final_mean_diff": 0.14460827886710248,
        "tr2_final_u": 25.0,
        "tr2_final_p_exact": 0.007936507936507936,
        "tr2_final_cliffs_delta": 1.0,
        "tr2_final_boot_ci_lo": 0.10662516339869288,
        "tr2_final_boot_ci_hi": 0.18453431372549028,
        "ft_advantage_remains": bool(ft_adv),
        "p_less_than_05": bool(sig),
        "interpretation_rule_met": bool(ft_adv and sig),
        "exploratory": 1,
        "sensitivity": True,
        "nominal_proxy_marked": "PROXY",
    }
    outpath = os.path.join(OUTDIR, "sens2_p2t.json")
    with open(outpath, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
