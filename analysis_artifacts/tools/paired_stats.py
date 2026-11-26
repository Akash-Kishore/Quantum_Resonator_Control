"""PHASE D D2 - tools\paired_stats.py (WP2 tooling, CPU only, EXPLORATORY).
For the eight contrasts C-1..C-8 (selection.json): paired difference per seed
index of lock_w1000 at the nominal condition, mean, bootstrap 95 percent CI
(10000 resamples, fixed seed), exact two-sided Wilcoxon signed-rank p, Cohen dz.
Outputs are EXPLORATORY: NEVER written to db.contrasts as FINAL, never replace
contrasts.py. Pairing by seed index is a convenience, not the preregistered
analysis. Also runs synthetic tests (known shift / null).
"""
import json, os, statistics
import numpy as np
from scipy import stats

STUDY = r"C:\Projects\MiniProject_Sem4\qstudy"
SEALED = os.path.join(STUDY, "results", "sealed")
OUT = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\results\paired_stats_report.json"
SEEDS = list(range(101, 111))
RNG = np.random.default_rng(0)
CONTRASTS = {
    "C-1": ("rl__dither__plain", "rl__rabi__plain"),
    "C-2": ("rl__ramsey_iq__plain", "rl__dither__plain"),
    "C-3": ("rl__dither__plain", "ctrl:pid (dither)"),
    "C-4": ("rl__dither__plain", "ctrl:bayes:nominal (dither)"),
    "C-5": ("rl__dither__filter", "rl__dither__plain"),
    "C-6": ("rl__rabi__gru8", "rl__rabi__plain"),
    "C-7": ("rl__dither__DRcurr", "rl__dither__plain"),
    "C-8": ("rl__dither__DRcurr", "rl__dither__DR"),
}


def rl_nominal(cfg):
    short = cfg[4:] if cfg.startswith("rl__") else cfg
    vals = []
    for s in SEEDS:
        p = os.path.join(SEALED, "z1__rl__%s_s%d__nom.json" % (short, s))
        try:
            d = json.load(open(p, encoding="utf-8"))
            vals.append(float(d["lock_w1000_mean"]))
        except Exception:
            vals.append(None)
    return vals


def wilcoxon_exact(d):
    d = np.asarray(d, dtype=float)
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue)


def bootstrap_ci(d, n=10000):
    d = np.asarray(d, dtype=float)
    boots = [RNG.choice(d, size=d.size, replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def cohen_dz(d):
    d = np.asarray(d, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def synth_test():
    # known shift
    a = RNG.normal(0.7, 0.05, 10)
    b = a + 0.10
    dk = b - a
    lo, hi = bootstrap_ci(dk)
    w = wilcoxon_exact(dk)
    shift_ok = (lo <= 0.10 <= hi) and w < 0.05
    # null
    c = RNG.normal(0.7, 0.05, 10)
    dn = c - c
    lo2, hi2 = bootstrap_ci(dn)
    w2 = wilcoxon_exact(dn)
    null_ok = (lo2 <= 0.0 <= hi2) and w2 > 0.05
    return shift_ok, null_ok, w, w2


def main():
    data = {}
    for name, (a, b) in CONTRASTS.items():
        va = rl_nominal(a) if a.startswith("rl__") else None
        vb = rl_nominal(b) if b.startswith("rl__") else None
        entry = {"arm_a": a, "arm_b": b, "computable": False}
        if va is not None and vb is not None and None not in va and None not in vb:
            diffs = [va[i] - vb[i] for i in range(10)]
            lo, hi = bootstrap_ci(diffs)
            w = wilcoxon_exact(diffs)
            entry.update(computable=True, diffs=diffs, mean=float(np.mean(diffs)),
                         ci_lo=lo, ci_hi=hi, p_wilcoxon=w,
                         cohen_dz=cohen_dz(diffs))
        else:
            entry["reason"] = "arm(s) lack per-seed nominal data (ctrl baseline at non-nom cond)"
        data[name] = entry
    shift_ok, null_ok, p_shift, p_null = synth_test()
    lines = []
    for name in sorted(data):
        e = data[name]
        if e["computable"]:
            lines.append("%s %-22s vs %-22s mean=%+.4f CI[%.4f,%.4f] p=%.4f dz=%.3f" % (
                name, e["arm_a"], e["arm_b"], e["mean"], e["ci_lo"], e["ci_hi"],
                e["p_wilcoxon"], e["cohen_dz"]))
        else:
            lines.append("%s %-22s vs %-22s NOT_COMPUTABLE: %s" % (name, e["arm_a"], e["arm_b"], e["reason"]))
    lines.append("synthetic known_shift(ok=%s,p=%.4f) null(ok=%s,p=%.4f)" % (shift_ok, p_shift, null_ok, p_null))
    report = {"contrasts": data, "synthetic": {"known_shift_ok": shift_ok, "p_shift": p_shift,
                                                 "null_ok": null_ok, "p_null": p_null},
              "label": "EXPLORATORY - not written to db.contrasts"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\n".join(lines))
    print("report -> %s" % OUT)


if __name__ == "__main__":
    main()
