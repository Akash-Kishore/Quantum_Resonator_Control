import json, re, sqlite3, sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
sys.path.insert(0, r"C:\Projects\MiniProject_Sem4")
from qstudy.db import connect
from qstudy.stress import applies_to, list_conds
S = Path(r"C:\Projects\MiniProject_Sem4")
SEL = json.load(open(S / "qstudy" / "configs" / "frozen" / "selection.json"))
PREREG = (S / "qstudy" / "PREREG.md").read_text(encoding="utf-8")
SEED_BASE = int(re.search(r"sealed\s+(\d+)", PREREG).group(1))
assert SEED_BASE == 900000, SEED_BASE
CONFIGS = SEL["confirmatory_configs"]
SEEDS = list(SEL["confirmatory_seeds"])
assert SEEDS == list(range(101, 111)), SEEDS
SIDES = SEL["contrast_sides"]
assert set(SIDES) == {"C-1","C-2","C-3","C-4","C-5","C-6","C-7","C-8"}
BOOT = 10000
BOOT_SEED = 1729
con = connect()
OFAT = list_conds("ofat")

def probe_of(cfg):
    if cfg.startswith("rl__rabi"): return "rabi"
    if cfg.startswith("rl__ramsey"): return "ramsey_iq"
    if cfg.startswith("rl__dither"): return "dither"
    raise KeyError(cfg)

def cells_of(probe):
    return [c for c in OFAT if applies_to(c, probe)]

def method_of(cfg, seed):
    return "rl:runs\\%s\\seed_%d" % (cfg, seed)

def cell_value(method, cond):
    r = con.execute("SELECT lock_w1000_mean FROM evals WHERE seed_base=? AND method=? AND cond_id=? AND split='sealed'", (SEED_BASE, method, cond)).fetchone()
    assert r is not None, (method, cond)
    return float(r[0])

def score(cfg, seed):
    cells = cells_of(probe_of(cfg))
    assert len(cells) > 0, cfg
    return float(np.mean([cell_value(method_of(cfg, seed), c) for c in cells]))

def baseline_score(method, probe):
    cells = cells_of(probe)
    assert len(cells) > 0, probe
    vals = [cell_value(method, c) for c in cells]
    return float(np.mean(vals))

found = {}
for (mth,) in con.execute("SELECT method FROM evals WHERE seed_base=?", (SEED_BASE,)):
    if not mth.startswith("rl:runs\\"):
        continue
    sm = re.search(r"seed_(\d+)$", mth)
    body = mth[len("rl:runs\\"):]
    cfg = body.split("\\seed_")[0]
    found.setdefault(cfg, set()).add(int(sm.group(1)))
for cfg in CONFIGS:
    assert cfg in found, cfg
    assert found[cfg] == set(SEEDS), (cfg, sorted(found[cfg]))
for cfg, ss in found.items():
    assert all(s >= 101 for s in ss), (cfg, ss)
for cid, sides in SIDES.items():
    assert len(sides) == 2, cid
    for side in sides:
        if side.startswith("ctrl:"):
            assert con.execute("SELECT 1 FROM evals WHERE seed_base=? AND method=? AND split='sealed' LIMIT 1", (SEED_BASE, side.split(" ")[0])).fetchone(), side
        else:
            assert side in found, side

def cliff(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    gt = float(np.sum(x[:, None] > y[None, :])); lt = float(np.sum(x[:, None] < y[None, :]))
    return float((gt - lt) / (len(x) * len(y)))

def bootstrap_ci(diffs, seed=BOOT_SEED, n=BOOT):
    rng = np.random.RandomState(seed)
    base = np.asarray(diffs, float)
    out = [float(np.mean(rng.choice(base, len(base), replace=True))) for _ in range(n)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0.0
    for i, idx in enumerate(order):
        v = min(1.0, pvals[idx] * (m - i))
        v = max(v, prev)
        adj[idx] = v
        prev = v
    return [float(v) for v in adj]

def main():
    cols = ["contrast_id", "endpoint", "method_a", "method_b", "n_a", "n_b", "mean_diff", "ci_low", "ci_high", "p_raw", "effect", "seeds_above", "p_holm", "direction_ok"]
    rows = []
    pvals = {}
    for cid in ["C-1","C-2","C-3","C-4","C-5","C-6","C-7","C-8"]:
        a, b = SIDES[cid]
        if a.startswith("ctrl:") or b.startswith("ctrl:"):
            rl = a if not a.startswith("ctrl:") else b
            bl = b if not a.startswith("ctrl:") else a
            probe = probe_of(rl)
            bs = baseline_score(bl.split(" ")[0], probe)
            seed_scores = [score(rl, s) for s in SEEDS]
            diffs = [s - bs for s in seed_scores]
            u = None
            p_w = wilcoxon(diffs, alternative="two-sided", method="exact", zero_method="wilcox").pvalue
            ci = bootstrap_ci(diffs)
            n_above = sum(1 for d in diffs if d > 0)
            mean_diff = float(np.mean(diffs))
            rows.append((cid, "P2", rl, bl.split(" ")[0], 10, 1, mean_diff, ci[0], ci[1], p_w, None, n_above, None, mean_diff > 0))
            pvals[cid] = p_w
        else:
            x = [score(a, s) for s in SEEDS]
            y = [score(b, s) for s in SEEDS]
            res = mannwhitneyu(x, y, alternative="two-sided", method="exact")
            mean_diff = float(np.mean(x) - np.mean(y))
            ci = bootstrap_ci([a_i - b_i for a_i, b_i in zip(x, y)])
            cd = cliff(x, y)
            rows.append((cid, "P2", a, b, 10, 10, mean_diff, ci[0], ci[1], float(res.pvalue), cd, None, None, mean_diff > 0))
            pvals[cid] = float(res.pvalue)
    h = holm([pvals[c] for c in ["C-1","C-2","C-3","C-4","C-5","C-6","C-7","C-8"]])
    holm_map = dict(zip(["C-1","C-2","C-3","C-4","C-5","C-6","C-7","C-8"], h))
    for i, r in enumerate(rows):
        r = list(r); r[12] = holm_map[r[0]]; rows[i] = tuple(r)
    # DB storage: add status column, supersede old n=5 rows, insert FINAL n=10 rows
    try:
        con.execute("ALTER TABLE contrasts ADD COLUMN status TEXT")
    except sqlite3.OperationalError:
        pass
    con.execute("UPDATE contrasts SET status='SUPERSEDED' WHERE status IS NULL OR status=''")
    now = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    for r in rows:
        cid, endpoint, ma, mb, na, nb, md, cl, ch, pr, eff, above, hp, dok = r
        con.execute("INSERT INTO contrasts (contrast_id, endpoint, method_a, method_b, n_a, n_b, statistic, p_value, effect_size, ci_low, ci_high, holm_p, exploratory, recorded_at, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,'FINAL')", (cid, endpoint, ma, mb, na, nb, md, pr, eff if eff is not None else (float(above) if above is not None else None), cl, ch, holm_map[cid], now))
    con.commit()
    return rows, holm_map

def p1p2():
    out = []
    for cfg in CONFIGS:
        probe = probe_of(cfg)
        cells = cells_of(probe)
        p1 = [cell_value(method_of(cfg, s), "nom") for s in SEEDS]
        p2 = [score(cfg, s) for s in SEEDS]
        out.append((cfg, float(np.mean(p1)), float(np.std(p1, ddof=1)), float(np.mean(p2)), float(np.std(p2, ddof=1))))
    return out

if __name__ == "__main__":
    rows, hm = main()
    print("cid   n_a n_b  meandiff    ci_low ci_high   p_raw     p_holm    effect/above  direction_ok")
    for r in rows:
        eff = r[10] if r[10] is not None else r[11]
        print("%s %d %d  %+.4f  %+.4f %+.4f  %.4g  %.4g  %s  %s" % (r[0], r[4], r[5], r[6], r[7], r[8], r[9], r[12], eff, r[13]))
    print("== EXPLORATORY P1/P2 per config (mean, sd over 10 seeds) ==")
    print("cfg                    P1_mean P1_sd   P2_mean P2_sd")
    for cfg, m1, s1, m2, s2 in p1p2():
        print("%-24s %+.4f %.4f  %+.4f %.4f" % (cfg, m1, s1, m2, s2))
    print("== direction flags (expect method_a > method_b; False = contradicts prereg order) ==")
    for r in rows:
        if not r[13]:
            print("CONTRADICTS", r[0], r[2], "vs", r[3], "meandiff %+.4f" % r[6])

