"""SENS-2 sensitivity analysis: FTB_200000 raw last-step vs E4 200k raw snapshot.
Nominal lock_w1000 comparison (P2T/OFAT blocked: OFAT jobs incomplete, E4 OFAT not submitted).
Uses exact Mann-Whitney U, Cliff's delta, bootstrap 95% CI (seed 20261012, 10000 resamples).
Stored as SENSITIVITY exploratory=1. Not part of Holm family.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats import mannwhitney_exact, cliffs_delta, boot_ci_diff

SEALED = os.path.join(HERE, "results", "sealed")
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "MiniProject_Sem4_QSTUDY_WORK")
os.makedirs(OUTDIR, exist_ok=True)
SEEDS = [200, 201, 202, 203, 204]
PROBES = ["LB", "LG", "RB", "RG"]
BOOT_SEED = 20261012
N_BOOT = 10000


def load_nominal(tag, model_prefix):
    """Load nominal lock_w1000 per seed per probe from sealed JSON."""
    data = {}
    for seed in SEEDS:
        for probe in PROBES:
            fname = f"z2{tag}__s15__{model_prefix}_s{seed}__{probe}__s{'' if False else ''}"
            # z2f: z2f__s15__FTB_200000_last_s200__LB__s200000__nom.json
            # z2s: z2s__s15__E4_s200__LB__s200000__nom.json
            if tag == "f":
                fname = f"z2f__s15__FTB_200000_last_s{seed}__{probe}__s200000__nom.json"
            else:
                fname = f"z2s__s15__E4_s{seed}__{probe}__s200000__nom.json"
            path = os.path.join(SEALED, fname)
            if os.path.exists(path):
                with open(path) as fh:
                    d = json.load(fh)
                if d.get("status") == "ok":
                    data.setdefault(probe, {})[seed] = d["lock_w1000_mean"]
            else:
                print(f"MISSING: {path}")
    return data


def summarize(name, a, b):
    u, p = mannwhitney_exact(a, b)
    cd = cliffs_delta(a, b)
    lo, hi = boot_ci_diff(a, b, seed=BOOT_SEED, n=N_BOOT)
    return dict(
        name=name,
        n_a=len(a), n_b=len(b),
        mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
        sd_a=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        sd_b=float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
        mean_diff=float(np.mean(a) - np.mean(b)),
        u=u, p_exact=p, cliffs_delta=cd,
        boot_ci_lo=lo, boot_ci_hi=hi,
        seeds_a={str(s): a[i] for i, s in enumerate(SEEDS[:len(a)])},
        seeds_b={str(s): b[i] for i, s in enumerate(SEEDS[:len(b)])},
    )


def main():
    ftb = load_nominal("f", "FTB_200000_last")
    e4 = load_nominal("s", "E4")

    results = {"probes": {}, "aggregate": None}

    # Per-probe
    for probe in PROBES:
        a = [ftb[probe][s] for s in SEEDS if s in ftb.get(probe, {})]
        b = [e4[probe][s] for s in SEEDS if s in e4.get(probe, {})]
        if len(a) == 5 and len(b) == 5:
            results["probes"][probe] = summarize(f"FTB_200000_last vs E4_200k ({probe})", a, b)

    # Aggregate across all probes
    a_all = []
    b_all = []
    for probe in PROBES:
        for s in SEEDS:
            if s in ftb.get(probe, {}):
                a_all.append(ftb[probe][s])
            if s in e4.get(probe, {}):
                b_all.append(e4[probe][s])
    results["aggregate"] = summarize("FTB_200000_last vs E4_200k (all probes)", a_all, b_all)

    # TR-2 FINAL row (from tr1_tr2_report.json)
    tr2 = dict(
        contrast="TR-2 FINAL",
        description="FTB_200k (LG+FT 200k) vs E4 RB 200k snapshot on P2T",
        mean_a=0.41041405228758177, mean_b=0.2658057734204793,
        mean_diff=0.14460827886710248,
        u=25.0, p_exact=0.007936507936507936, cliffs_delta=1.0,
        boot_ci_lo=0.10662516339869288, boot_ci_hi=0.18453431372549028,
        endpoint="P2T (OFAT macro-average, RB)",
    )

    # Print tables
    print("=" * 90)
    print("SENS-2: FTB_200000 raw last-step vs E4 200k raw snapshot (nominal lock_w1000)")
    print("Endpoint: nominal lock_w1000 per probe (P2T/OFAT blocked - see limitations)")
    print("Seeds:", SEEDS, "| Exact Mann-Whitney U | Cliff's delta | Bootstrap 95% CI (seed 20261012, n=10000)")
    print("=" * 90)
    print()
    print(f"{'Contrast':<45} {'mean_A':>8} {'mean_B':>8} {'diff':>9} {'U':>6} {'p_exact':>10} {'Cliff_d':>9} {'95% CI lo':>10} {'95% CI hi':>10}")
    print("-" * 90)
    for probe, r in results["probes"].items():
        print(f"{'FTB_200k vs E4_200k (' + probe + ')':<45} {r['mean_a']:>8.4f} {r['mean_b']:>8.4f} "
              f"{r['mean_diff']:>9.4f} {r['u']:>6.1f} {r['p_exact']:>10.6f} {r['cliffs_delta']:>9.3f} "
              f"{r['boot_ci_lo']:>10.4f} {r['boot_ci_hi']:>10.4f}")
    r = results["aggregate"]
    print(f"{'FTB_200k vs E4_200k (aggregate)':<45} {r['mean_a']:>8.4f} {r['mean_b']:>8.4f} "
          f"{r['mean_diff']:>9.4f} {r['u']:>6.1f} {r['p_exact']:>10.6f} {r['cliffs_delta']:>9.3f} "
          f"{r['boot_ci_lo']:>10.4f} {r['boot_ci_hi']:>10.4f}")
    print()
    print("TR-2 FINAL row (side-by-side):")
    print(f"{'TR-2 FINAL':<45} {tr2['mean_a']:>8.4f} {tr2['mean_b']:>8.4f} "
          f"{tr2['mean_diff']:>9.4f} {tr2['u']:>6.1f} {tr2['p_exact']:>10.6f} {tr2['cliffs_delta']:>9.3f} "
          f"{tr2['boot_ci_lo']:>10.4f} {tr2['boot_ci_hi']:>10.4f}")
    print(f"{'  endpoint: ' + tr2['endpoint']:<45} {'':>8} {'':>8} {'':>9} {'':>6} {'':>10} {'':>9} {'':>10} {'':>10}")
    print()
    print("Interpretation (pre-declared): if FT advantage remains (same direction, exact p < 0.05 unadjusted)")
    print("the TR-2 selection asymmetry is closed as a limitation; otherwise both numbers are reported.")
    print()
    print("LIMITATIONS: P2T (OFAT macro-average) NOT available - OFAT z2f jobs incomplete (4 zombie killed,")
    print("16 pending) and E4 OFAT never submitted. Nominal lock_w1000 used as proxy endpoint.")

    # Save results
    out = {
        "sens": "SENS-2",
        "comparison": "FTB_200000 raw last-step vs E4 200k raw snapshot",
        "endpoint_nominal": "lock_w1000 (nominal condition)",
        "endpoint_p2t": "BLOCKED - OFAT incomplete",
        "seeds": SEEDS,
        "probes": PROBES,
        "results": results,
        "tr2_final": tr2,
        "exploratory": 1,
        "sensitivity": True,
    }
    outpath = os.path.join(OUTDIR, "results", "sens2_nominal.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
