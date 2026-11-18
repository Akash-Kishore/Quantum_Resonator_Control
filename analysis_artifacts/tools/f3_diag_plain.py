"""F3 diagnostic: dump all val rows for rl__dither__plain with row metadata."""
import json, os, re, sqlite3

S = r"C:\Projects\MiniProject_Sem4\qstudy"
con = sqlite3.connect(os.path.join(S, "db", "qstudy.sqlite"))
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute("SELECT * FROM evals")]
con.close()
VC = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]

per = {}
for r in rows:
    m = re.search(r"runs.(.+?).seed_(\d+)$", r["model_dir"] or "")
    if not m:
        continue
    cfg, seed = m.group(1), int(m.group(2))
    if cfg == "rl__dither__plain" and r["split"] == "val" and r["cond_id"] in VC:
        per.setdefault(seed, []).append(r)

for s in sorted(per):
    print("seed %d: %d rows" % (s, len(per[s])))
    for r in sorted(per[s], key=lambda r: r["cond_id"]):
        print("  %-22s | md=%-58s | ep=%s | lw1000=%.4f | npz=%s | fin=%s" % (
            r["cond_id"], r["model_dir"], r["episodes"], r["lock_w1000_mean"] or -1,
            r["npz_path"], r["finished"]))

print()
print("=== old (stage118_draft) computation: mean over ALL rows per seed ===")
import statistics
import numpy as np
allv = {}
for s in sorted(per):
    v = [r["lock_w1000_mean"] for r in per[s]]
    allv[s] = float(np.mean(v))
    print("seed %d: n_rows=%d mean=%.6f" % (s, len(v), allv[s]))
print("val_mean=%.6f val_sd=%.6f" % (float(np.mean(list(allv.values()))),
                                     float(np.std(list(allv.values()), ddof=1))))
print()
print("=== strict: one value per (seed, cond) from the CURRENT result file ===")
fdir = os.path.join(S, "results", "val")
for s in sorted(per):
    vals = {}
    for r in per[s]:
        p = os.path.join(S, (r["npz_path"] or "").replace(".npz", ".json"))
        if os.path.exists(p) and "_INVALID" not in p:
            j = json.load(open(p, encoding="utf-8"))
            if j.get("cond_id") == r["cond_id"] and j.get("episodes") == r["episodes"]:
                vals.setdefault(r["cond_id"], []).append((r["finished"], j["lock_w1000_mean"], os.path.basename(p)))
    print("seed %d:" % s, {k: v for k, v in sorted(vals.items())})
