"""Generate all result tables (CSV + markdown) for the missing exploratory docs (W3).
Every number comes from the database or a result file; nothing is typed by hand."""
import csv, glob, json, math, os, re, sqlite3, sys

import numpy as np

S = r"C:\Projects\MiniProject_Sem4\qstudy"
T = os.path.join(S, "results", "tables")
os.makedirs(T, exist_ok=True)
MD = []


def ci95(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(1.96 * x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan")


def fmt(v, nd=4):
    return ("%." + str(nd) + "f") % v if v is not None and math.isfinite(v) else "n/a"


def write(name, header, rows):
    with open(os.path.join(T, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    MD.append("### %s\n```csv\n%s\n```\n" % (name, "\n".join(md)))
    print("wrote", name, "(%d rows)" % len(rows))


con = sqlite3.connect(os.path.join(S, "db", "qstudy.sqlite"))
con.row_factory = sqlite3.Row
EV = [dict(r) for r in con.execute("SELECT * FROM evals")]
con.close()

VAL_CONDS = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]


def config_of(md_):
    m = re.search(r"runs.(.+?).seed_(\d+)", md_ or "")
    return (m.group(1), int(m.group(2))) if m else (None, None)


def ctrl_rows(method, conds=VAL_CONDS):
    out = []
    for cond in conds:
        for r in EV:
            if r["method"] == method and r["cond_id"] == cond and r["split"] == "val":
                out.append((cond, r["lock_w1000_mean"], r["episodes"]))
    return out


def rl_rows(config, conds=VAL_CONDS, split="val"):
    per = {}
    for r in EV:
        cfg, seed = config_of(r["model_dir"])
        if cfg == config and r["cond_id"] in conds and r["split"] == split:
            per.setdefault((r["cond_id"], seed), r["lock_w1000_mean"])
    return per


def method_table(name, entries, conds=VAL_CONDS):
    """entries: list of (label, kind, key) where kind='ctrl' -> key=method,
    kind='rl' -> key=config. One row per (label, cond) with n/mean/sd/ci95 over seeds."""
    rows = []
    for label, kind, key in entries:
        for cond in conds:
            if kind == "ctrl":
                vals = [v for c, v, _ in ctrl_rows(key, [cond])]
                n_ep = [e for c, v, e in ctrl_rows(key, [cond])]
                ep = int(n_ep[0]) if n_ep else 0
                src = "db:evals method=%s" % key
            else:
                per = rl_rows(key, [cond])
                vals = [v for (c, s), v in sorted(per.items())]
                ep = 200
                src = "db:evals model_dir~%s" % key
            if not vals:
                continue
            a = np.asarray(vals, dtype=float)
            rows.append([label, cond, len(a), ep, fmt(float(a.mean())), fmt(float(a.std(ddof=1))) if a.size > 1 else "0",
                         fmt(ci95(a)), src])
    write(name, ["method", "cond", "n_seeds", "episodes", "lock_w1000_mean", "sd", "ci95", "source"], rows)


# ---------------- approach 1 (dither) ----------------
method_table("a1_dither_methods.csv",
             [("noctrl", "ctrl", "ctrl:noctrl"), ("oracle-ceiling", "ctrl", "ctrl:oracle"),
              ("pid", "ctrl", "ctrl:pid"), ("bayes:nominal", "ctrl", "ctrl:bayes:nominal"),
              ("rl__dither__plain", "rl", "rl__dither__plain")])
per = rl_rows("rl__dither__plain")
rows = [["seed_%d" % s, c, fmt(per[(c, s)])] for (c, s) in sorted(per, key=lambda k: (k[1], VAL_CONDS.index(k[0]))) if (c, s) in per]
write("a1_dither_seeds.csv", ["run", "cond", "lock_w1000"], rows)

# ---------------- approach 2 (ramsey) ----------------
method_table("a2_ramsey_methods.csv",
             [("noctrl", "ctrl", "ctrl:noctrl"), ("oracle-ceiling", "ctrl", "ctrl:oracle"),
              ("pid", "ctrl", "ctrl:pid"), ("bayes:nominal", "ctrl", "ctrl:bayes:nominal"),
              ("bayes:oracle", "ctrl", "ctrl:bayes:oracle"),
              ("rl__ramsey_iq__plain", "rl", "rl__ramsey_iq__plain")])
rows = []
for tau in (25, 50, 100):
    p = os.path.join(S, "results", "val", "ctrl_pid__ramsey_tau%d__tune.json" % tau)
    if os.path.exists(p):
        d = json.load(open(p))
        rows.append(["tau_%dus" % tau, fmt(d["lock_w1000_mean"]), d["episodes"], p])
write("a2_pid_tau_sweep.csv", ["condition", "lock_w1000_mean", "episodes", "source_file"], rows)

# ---------------- approach 3 (bayes + hybrid) ----------------
method_table("a3_bayes_nominal_vs_oracle.csv",
             [("bayes:nominal", "ctrl", "ctrl:bayes:nominal"),
              ("bayes:oracle", "ctrl", "ctrl:bayes:oracle")])
entries = []
for probe in ("rabi", "dither", "ramsey_iq"):
    entries.append(("rl__%s__plain" % probe, "rl", "rl__%s__plain" % probe))
    entries.append(("rl__%s__filter" % probe, "rl", "rl__%s__filter" % probe))
method_table("a3_hybrid_vs_plain.csv", entries)

# ---------------- approach 4 (DR / curriculum) ----------------
entries = []
for probe in ("rabi", "dither", "ramsey_iq"):
    for suffix in ("plain", "DR", "DRcurr"):
        entries.append(("rl__%s__%s" % (probe, suffix), "rl", "rl__%s__%s" % (probe, suffix)))
method_table("a4_dr_families.csv", entries)
rows = []
for probe in ("rabi", "dither", "ramsey_iq"):
    for fam in ("DR", "DRcurr"):
        p = os.path.join(S, "runs", "rl__%s__%s" % (probe, fam))
        for sd in sorted(glob.glob(os.path.join(p, "seed_*"))):
            ts = os.path.join(sd, "train_summary.json")
            if not os.path.exists(ts):
                continue
            j = json.load(open(ts))
            vh = j.get("val_history") or []
            if len(vh) >= 2:
                rows.append([os.path.relpath(sd, S), probe, fam, j.get("seed"),
                             fmt(vh[0]["score"]), fmt(vh[len(vh) // 2]["score"]), fmt(vh[-1]["score"]),
                             fmt(j.get("best_val"))])
write("a4_curriculum_ramp.csv",
      ["run", "probe", "family", "seed", "val_first", "val_mid", "val_last", "best_val"], rows)
sys.path.insert(0, S)
import stress
n_in = sum(1 for c in stress.list_conds("ofat") if stress.dr_support(c) == "IN")
n_out = sum(1 for c in stress.list_conds("ofat") if stress.dr_support(c) == "OUT")
write("a4_dr_support.csv", ["cond_id", "support"], [[c, stress.dr_support(c)] for c in stress.list_conds("ofat")]
      + [["TOTAL_IN", n_in], ["TOTAL_OUT", n_out]])

# ---------------- approach 6 (memory) ----------------
entries = []
for probe in ("rabi", "ramsey_iq"):
    for suffix in ("plain", "K4", "K8", "K16", "gru8"):
        entries.append(("rl__%s__%s" % (probe, suffix), "rl", "rl__%s__%s" % (probe, suffix)))
method_table("a6_memory_families.csv", entries)

# ---------------- approach 7 (noise) ----------------
def ofat_table(name, method, factors):
    rows = []
    for factor in factors:
        for cond in sorted({r["cond_id"] for r in EV if r["method"] == method and r["cond_id"].startswith("ofat_%s_" % factor)}):
            vals = [r["lock_w1000_mean"] for r in EV if r["method"] == method and r["cond_id"] == cond and r["split"] == "val"]
            if vals:
                rows.append([factor, cond, fmt(float(np.mean(vals))), len(vals),
                             "db:evals method=%s cond=%s" % (method, cond)])
    nom = [r["lock_w1000_mean"] for r in EV if r["method"] == method and r["cond_id"] == "nom" and r["split"] == "val"]
    if nom:
        rows.insert(0, ["-", "nom", fmt(float(np.mean(nom))), len(nom), "db:evals method=%s cond=nom" % method])
    write(name, ["factor", "cond", "lock_w1000_mean", "n_rows", "source"], rows)

ofat_table("a7_noise_pid_ofat.csv", "ctrl:pid",
           ["shots", "t2", "action_delay", "dac_bits", "drift_sigma"])
ofat_table("a7_noise_hill_ofat.csv", "ctrl:hill",
           ["shots", "t2", "action_delay", "dac_bits", "drift_sigma"])
rows = []
for p in sorted(glob.glob(os.path.join(S, "results", "val", "noise_*__ro*.json"))) + \
        sorted(glob.glob(os.path.join(S, "results", "val", "noise_*readout*.json"))):
    d = json.load(open(p))
    cfg = d.get("cfg") or {}
    ro = cfg.get("e01")
    rows.append([os.path.basename(p)[:-5], d.get("method"), fmt(ro, 3) if ro is not None else "n/a",
                 fmt(d["lock_w1000_mean"]), d["episodes"], os.path.relpath(p, S)])
write("a7_noise_readout.csv", ["file", "method", "readout_e01", "lock_w1000_mean", "episodes", "path"], rows)

# ---------------- tuning ----------------
rows = []
for p in sorted(glob.glob(os.path.join(S, "runs", "tune__*", "seed_100", "train_summary.json"))):
    j = json.load(open(p))
    m = re.match(r"tune__lr(\w+)__ent(\w+)__gam(\w+)", os.path.basename(os.path.dirname(os.path.dirname(p))))
    rows.append([os.path.relpath(os.path.dirname(p), S), m.group(0) if m else os.path.basename(p),
                 j.get("steps"), fmt(j.get("wall_seconds"), 1), fmt(j.get("best_val")),
                 fmt(j["steps"] / j["wall_seconds"], 1) if j.get("wall_seconds") else "n/a",
                 j.get("device")])
rows.sort(key=lambda r: -(float(r[4]) if r[4] != "n/a" else -1))
write("tuning_table.csv", ["run", "grid_point", "steps", "wall_s", "best_val", "steps_per_s", "device"], rows)

with open(os.path.join(T, "doc_tables.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(MD))
print("markdown snippets -> results\\tables\\doc_tables.md")
