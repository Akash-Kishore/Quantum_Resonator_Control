"""Stage 11.8 (as amended by O2): ranking tables + DRAFT configs\\frozen\\selection.json.

Uses ONLY validation scores (split=val, seed_base 500000). Prints compactly:
(a) invariant checks, (b) per-config validation score (mean/sd over seeds), the tuning
table, the draft selection C1-C6. No FREEZE ADDENDUM, no PREREG.sha256 (O2)."""
import glob, json, math, os, re, sqlite3, sys

import numpy as np

S = r"C:\Projects\MiniProject_Sem4\qstudy"
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
VAL_CONDS = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]

con = sqlite3.connect(os.path.join(S, "db", "qstudy.sqlite"))
rows = [dict(zip([c[0] for c in con.execute("SELECT * FROM evals LIMIT 0").description],
                 r)) for r in con.execute("SELECT * FROM evals")]
con.close()


def val_scores(prefix):
    """config -> {seed: mean lock_w1000 over the 4 val conditions}."""
    per = {}
    for r in rows:
        m = re.search(r"runs.(.+?).seed_(\d+)$", r["model_dir"] or "")
        if not m:
            continue
        cfg, seed = m.group(1), int(m.group(2))
        if cfg == prefix and r["split"] == "val" and r["cond_id"] in VAL_CONDS:
            per.setdefault(seed, []).append(r["lock_w1000_mean"])
    return {s: float(np.mean(v)) for s, v in per.items()}


def fam_of(cfg):
    if "__full" in cfg:
        return "full"
    if "__DRcurr" in cfg:
        return "DRcurr"
    if "__DR" in cfg:
        return "DR"
    if "__filter" in cfg:
        return "filter"
    if re.search(r"__K\d+$|__gru8$", cfg):
        return "memory"
    if "__plain" in cfg:
        return "plain"
    return None


print("=== INVARIANT CHECKS (O2 a/b) ===")
sealed = [r for r in rows if r["split"] == "sealed"]
print("(a) result files with split=sealed: %d" % len(sealed))
bad_runs = []
for d in glob.glob(os.path.join(S, "runs", "*", "seed_1??")):
    m = re.search(r"seed_(\d+)$", d)
    s = int(m.group(1))
    fam = os.path.basename(os.path.dirname(d))
    if 100 <= s <= 109 and not fam.startswith("tune__"):
        bad_runs.append(os.path.relpath(d, S))
print("(b) seed 100-109 run dirs except tuning: %d %s" % (len(bad_runs), bad_runs[:5]))

print()
print("=== VALIDATION SCORE PER CONFIG (mean over the 4 val cells, per seed) ===")
print("%-34s %-6s %-8s %-8s %-8s %s" % ("config", "n", "mean", "sd", "ci95", "family"))
scores = {}
for cfg in sorted({m.group(1) for r in rows for m in [re.search(r"runs.(.+?).seed_(\d+)$", r["model_dir"] or "")] if m}):
    per = val_scores(cfg)
    if len(per) < 3:
        continue
    a = np.asarray(sorted(per.values()))
    fam = fam_of(cfg)
    scores.setdefault(fam, {})[cfg] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                                           ci=float(1.96 * a.std(ddof=1) / math.sqrt(a.size)),
                                           n=int(a.size), seeds=per)
    print("%-34s %-6d %-8.4f %-8.4f %-8.4f %s" % (cfg, a.size, a.mean(), a.std(ddof=1),
                                                  1.96 * a.std(ddof=1) / math.sqrt(a.size), fam))

print()
print("=== TUNING TABLE (from docs, tune seed 100) ===")
tt = open(os.path.join(S, "results", "tables", "tuning_table.csv")).read().strip().splitlines()
for line in tt:
    print("  " + line)

sel = {}
sel["C1_legacy_analog"] = "rl__rabi__plain"
best_plain = max(scores.get("plain", {}), key=lambda c: scores["plain"][c]["mean"])
sel["C2_best_plain"] = best_plain
best_filter = max(scores.get("filter", {}), key=lambda c: scores["filter"][c]["mean"])
sel["C3_best_hybrid_filter"] = best_filter
best_mem = max(scores.get("memory", {}), key=lambda c: scores["memory"][c]["mean"])
sel["C4_best_memory"] = best_mem
best_dr = max({**scores.get("DR", {}), **scores.get("DRcurr", {})}, key=lambda c: ({**scores["DR"], **scores["DRcurr"]})[c]["mean"])
sel["C5_best_dr"] = best_dr
best_full = max(scores.get("full", {}), key=lambda c: scores["full"][c]["mean"]) if scores.get("full") else None
sel["C6_best_full"] = best_full

print()
print("=== DRAFT SELECTION (O2: DRAFT ONLY, no freeze yet) ===")
for k, v in sel.items():
    sc = None
    for fam in scores.values():
        if v in fam:
            sc = fam[v]
    print("%-22s %-34s %s" % (k, v, ("val_mean=%.4f sd=%.4f n=%d" % (sc["mean"], sc["sd"], sc["n"])) if sc else "n/a"))

notes = []
seen = set()
for k, v in sel.items():
    if v in seen:
        notes.append("%s coincides with an earlier pick" % k)
    seen.add(v)

out = dict(status="DRAFT (O2: awaiting controller approval; freeze addendum NOT written)",
           selection=sel,
           baselines_frozen=["ctrl:pid", "ctrl:hill", "ctrl:bayes:nominal", "ctrl:bayes:oracle",
                              "ctrl:noctrl", "ctrl:oracle"],
           hyperparameters=dict(learning_rate=3e-4, ent_coef=0.003, gamma=0.99,
                                source="tune seed 100, best_val 0.8298 (tune__lr3e4__ent0003__gam099)"),
           val_scores={fam: {cfg: {k: v for k, v in d.items() if k != "seeds"}
                             for cfg, d in cfgs.items()} for fam, cfgs in scores.items()},
           coincidences=notes,
           invariants=dict(sealed_files=len(sealed), bad_seed_runs=bad_runs))
os.makedirs(os.path.join(S, "configs", "frozen"), exist_ok=True)
with open(os.path.join(S, "configs", "frozen", "selection.json"), "w") as f:
    json.dump(out, f, indent=1)
print()
print("draft written: configs\\frozen\\selection.json")
