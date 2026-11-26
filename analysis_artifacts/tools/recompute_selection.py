"""M2 F3: recompute every val score in configs\\frozen\\selection.json from the database
using ONLY valid runs (rows backed by an existing, non-INVALID result file whose
lock_w1000_mean matches the file). Prints old-vs-new val_mean/val_sd per config, the
mismatches, the explanation of the known rl__dither__plain stale input, and writes the
recomputed table to WORK\\results\\f3_recompute.json (used by the review pack v2 T1)."""
import glob, json, math, os, re, sqlite3, statistics

S = r"C:\Projects\MiniProject_Sem4\qstudy"
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
VAL_CONDS = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]

sel = json.load(open(os.path.join(S, "configs", "frozen", "selection.json"), encoding="utf-8"))
old = {}
for fam, cfgs in sel["val_scores"].items():
    for cfg, d in cfgs.items():
        old.setdefault(cfg, (d["mean"], d["sd"], d["n"]))

con = sqlite3.connect(os.path.join(S, "db", "qstudy.sqlite"))
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute("SELECT * FROM evals")]
con.close()


def file_ok(r):
    """Valid row: backing json exists (npz twin path), not INVALID, values current."""
    npz = r["npz_path"] or ""
    p = npz if os.path.isabs(npz) else os.path.join(S, npz)
    if "_invalid" in p.lower():
        return False, "INVALID tag"
    jp = p[:-4] + ".json" if p.endswith(".npz") else p
    if not os.path.exists(jp):
        return False, "file missing"
    try:
        j = json.load(open(jp, encoding="utf-8"))
    except Exception:
        return False, "unloadable"
    if abs((j.get("lock_w1000_mean") or -1) - (r["lock_w1000_mean"] or -2)) > 1e-9:
        return False, "stale (db != file)"
    if (j.get("episodes") or 0) != (r["episodes"] or 0):
        return False, "episodes mismatch"
    return True, ""


per = {}
stale_rows = []
for r in rows:
    if r["split"] != "val" or r["cond_id"] not in VAL_CONDS:
        continue
    m = re.search(r"runs.(.+?).seed_(\d+)$", r["model_dir"] or "")
    if not m:
        continue
    cfg, seed = m.group(1), int(m.group(2))
    ok, why = file_ok(r)
    if not ok:
        stale_rows.append((cfg, seed, r["cond_id"], why, r["eval_id"][:80]))
        continue
    per.setdefault(cfg, {}).setdefault(seed, {})[r["cond_id"]] = r["lock_w1000_mean"]

new, detail = {}, {}
for cfg, seeds in per.items():
    use = {s: statistics.mean(v.values()) for s, v in seeds.items() if len(v) == 4}
    if len(use) < 3:
        continue
    a = [use[s] for s in sorted(use)]
    new[cfg] = dict(mean=statistics.mean(a), sd=statistics.stdev(a) if len(a) > 1 else 0.0,
                    n=len(a), seeds={s: use[s] for s in sorted(use)},
                    partial={s: len(v) for s, v in seeds.items() if len(v) != 4})

json.dump(dict(new=new, stale_rows=stale_rows),
          open(os.path.join(W, "results", "f3_recompute.json"), "w"), indent=1)

print("%-32s %-8s %-8s | %-8s %-8s %-3s | d_mean    d_sd" % ("config", "old_mean", "old_sd", "new_mean", "new_sd", "n"))
mism = 0
for cfg in sorted(set(old) | set(new)):
    om, osd, on = old.get(cfg, (float("nan"),) * 2 + (0,))
    nm = new.get(cfg, {})
    nmean, nsd, nn = nm.get("mean", float("nan")), nm.get("sd", float("nan")), nm.get("n", 0)
    dm = abs(om - nmean)
    flag = ""
    if cfg not in new:
        flag = "DROPPED(no valid rows)"
    elif cfg not in old:
        flag = "NEW"
    elif dm > 1e-6 or abs(osd - nsd) > 1e-6:
        flag = "MISMATCH"
        mism += 1
    print("%-32s %-8.4f %-8.4f | %-8.4f %-8.4f %-3d | %.4f  %.4f  %s"
          % (cfg, om, osd, nmean, nsd, nn, dm, abs(osd - nsd), flag))
print("MISMATCHES %d; stale/excluded rows: %d" % (mism, len(stale_rows)))
for cfg, seed, cond, why, eid in stale_rows[:12]:
    print("  stale: %s seed_%d %s (%s) %s" % (cfg, seed, cond, why, eid))
