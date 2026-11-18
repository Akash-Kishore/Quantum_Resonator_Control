"""qstudy stats — statistical plan (Appendix D / Stage 13.1).
Implements: per-seed scores; endpoints P1 (nominal lock_w1000) and P2 (OFAT macro-average);
the 8 contrasts C-1..C-8 on P2 with Mann-Whitney U (exact, two-sided), Cliff's delta, bootstrap 95%
CI of the mean difference (10000 resamples, seeded), Holm correction over the 8; paired RL-vs-baseline
differences with bootstrap CI and count of seeds above the baseline; secondary metric tables with
mean, sd, 95% CI; everything outside the 8 contrasts labelled EXPLORATORY; CRB floor vs achieved rms;
Allan deviation of residual error series; Welch PSD of residual error.
Outputs -> results\\tables\\*.csv, ingested into the database table contrasts."""
import glob, itertools, json, os, sqlite3, sys

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
T = os.path.join(HERE, "results", "tables")
os.makedirs(T, exist_ok=True)
DB = os.path.join(HERE, "db", "qstudy.sqlite")
BOOT_SEED = 20261012
N_BOOT = 10000

OFAT_APPLIES = "ofat_"


def load_evals():
    """evals from db -> list of dicts."""
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM evals")]
    con.close()
    return rows


def seed_score(rows, method, model_dir_prefix, cond_filter, metric="lock_w1000_mean"):
    """Per-seed score = mean over episodes for the (method, seed, cond-filter) cells.
    Returns dict seed -> score over the matching conds."""
    out = {}
    for r in rows:
        if r["method"] != method:
            continue
        md = r["model_dir"] or ""
        if model_dir_prefix and not md.startswith(model_dir_prefix):
            continue
        conds = cond_filter(r["cond_id"])
        if conds is None or not conds:
            continue
        import re
        m = re.search(r"seed_(\d+)", md)
        seed = int(m.group(1)) if m else -1
        out.setdefault(seed, []).append((conds, r[metric]))
    return {s: float(np.mean([v for _, v in vs])) for s, vs in out.items()}


def p2_ofat_conds(probe):
    """All OFAT cells that apply to this probe (stress.applies_to)."""
    import stress
    return [c for c in stress.list_conds("ofat") if stress.applies_to(c, probe)]


def mannwhitney_exact(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    try:
        u, p = sps.mannwhitneyu(a, b, alternative="two-sided", method="exact")
    except Exception:
        u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def boot_ci_diff(a, b, seed=BOOT_SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = []
    for _ in range(n):
        ra = rng.choice(a, len(a), replace=True)
        rb = rng.choice(b, len(b), replace=True)
        diffs.append(ra.mean() - rb.mean())
    diffs = np.sort(diffs)
    return float(diffs[int(0.025 * n)]), float(diffs[int(0.975 * n)])


def holm(pvals):
    """Holm step-down adjusted p-values."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def allan_deviation(x, taus=(1, 2, 4, 8, 16, 32)):
    """Allan deviation of a residual error series for averaging times tau (in steps)."""
    x = np.asarray(x, float)
    out = {}
    for tau in taus:
        if len(x) < 2 * tau:
            out[tau] = np.nan
            continue
        xa = x[: (len(x) // tau) * tau].reshape(-1, tau).mean(axis=1)
        d = np.diff(xa)
        out[tau] = float(np.sqrt(0.5 * np.mean(d ** 2)))
    return out


def welch_psd(x, fs=1.0):
    from scipy.signal import welch
    f, p = welch(np.asarray(x, float), fs=fs, nperseg=min(256, len(x)))
    return f, p


def crb_vs_rms(npz_paths_by_probe, shots_list=(64, 128, 256, 512, 1024, 4096)):
    """Static CRB floor vs achieved rms per probe and shot budget."""
    import qphysics as qp
    rows = []
    for probe, paths in npz_paths_by_probe.items():
        for shots in shots_list:
            # Fisher floor at delta 0 (best case)
            if probe == "ramsey_iq":
                fl = qp.fisher_ramsey_iq(0.0, shots / 2, tau=50e-6, t2=200e-6, e01=0.02, e10=0.02)
            else:
                fl = qp.fisher_rabi(0.0, shots, omega=5000.0, e01=0.02, e10=0.02)
            floor = 1.0 / np.sqrt(fl)
            ach = []
            for p in paths:
                z = np.load(p, allow_pickle=True)
                if "rms" in z.files:
                    ach.extend(z["rms"].tolist())
            rows.append(dict(probe=probe, shots=shots, crb_floor_hz=float(floor),
                             achieved_rms=float(np.mean(ach)) if ach else np.nan))
    return rows


def ci95(x):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return float(m - 1.96 * se), float(m + 1.96 * se)


def write_csv(name, header, rows):
    import csv
    with open(os.path.join(T, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", name, len(rows), "rows")


def main():
    rows = load_evals()
    print("loaded %d eval rows" % len(rows))
    # Secondary metrics table (all evals; EXPLORATORY)
    hdr = ["method", "model_dir", "cond_id", "split", "episodes", "lock_w1000_mean", "lock_w1000_sd",
           "rms_mean", "rms_sd", "p95_abs_mean", "lock_lost_rate", "mean_reward_mean"]
    write_csv("secondary_metrics.csv", hdr,
              [[r[h] for h in hdr] for r in rows])

    # P1/P2 per-seed scores for RL configs and baselines (EXPLORATORY until the freeze)
    import re
    score_rows = []
    for r in rows:
        if r["split"] == "val" and r["method"].startswith("rl:"):
            m = re.search(r"runs\\([\w_]+)\\seed_(\d+)", r["model_dir"] or "")
            if m:
                score_rows.append([m.group(1), int(m.group(2)), r["cond_id"], r["lock_w1000_mean"]])
    write_csv("rl_val_scores.csv", ["config", "seed", "cond", "lock_w1000"], score_rows)

    # Allan deviation + PSD from stored traces (EXPLORATORY)
    import glob as g
    ad_rows = []
    for p in sorted(g.glob(os.path.join(HERE, "results", "val", "*.npz")))[:40]:
        z = np.load(p, allow_pickle=True)
        if "trace_e" in z.files:
            tr = z["trace_e"]
            for b in range(min(3, tr.shape[1])):
                ad = allan_deviation(tr[:, b])
                ad_rows.append([os.path.basename(p), b] + [ad[t] for t in (1, 2, 4, 8, 16, 32)])
    write_csv("allan_deviation.csv", ["file", "episode", "tau1", "tau2", "tau4", "tau8", "tau16", "tau32"], ad_rows)
    print("STATS stage-A tables written (secondary metrics, RL val scores, Allan). "
          "Contrast computation runs after the Stage-11 freeze with the frozen selection.")


if __name__ == "__main__":
    main()
