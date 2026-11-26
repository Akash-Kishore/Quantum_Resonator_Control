"""Generate mechanical evidence snapshots for the Part-2 B5 handover answers.
Writes understanding\\handover_evidence\\*.txt. Numbers come only from files on disk."""
import glob
import json
import os
import sqlite3
import subprocess
import sys

W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
S = r"C:\Projects\MiniProject_Sem4\qstudy"
OUT = os.path.join(W, "understanding", "handover_evidence")
os.makedirs(OUT, exist_ok=True)
PY = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"

# 1. state show
r = subprocess.run([PY, os.path.join(W, "tools", "state.py"), W, "show"],
                   capture_output=True, text=True, timeout=60)
lines = r.stdout.splitlines()
present = [l.split()[0] for l in lines if l.strip()]
absent = [x for x in ["S%d" % i for i in range(15)] if x not in present]
with open(os.path.join(OUT, "state_show.txt"), "w", encoding="utf-8") as f:
    f.write(r.stdout)
    f.write("STEPS PRESENT: %s\n" % ",".join(present))
    f.write("ABSENT (NOT STARTED, no state entry): %s\n" % ",".join(absent))

# 2. job counts per lane+state
rows = []
for p in sorted(glob.glob(os.path.join(W, "queue", "*.job.json"))):
    j = json.load(open(p, encoding="utf-8"))
    sp = p.replace(".job.json", ".status.json")
    st = json.load(open(sp, encoding="utf-8"))["state"] if os.path.exists(sp) else "PENDING"
    rows.append((j["id"], j["lane"], st))
counts = {}
for jid, lane, st in rows:
    counts[(lane, st)] = counts.get((lane, st), 0) + 1
with open(os.path.join(OUT, "job_counts.txt"), "w", encoding="utf-8") as f:
    for (lane, st) in sorted(counts):
        f.write("lane=%s state=%s count=%d\n" % (lane, st, counts[(lane, st)]))
    tot = {}
    for jid, lane, st in rows:
        tot[st] = tot.get(st, 0) + 1
    f.write("TOTALS %s\n" % tot)
    for jid, lane, st in rows:
        if st == "RUNNING":
            f.write("RUNNING now: %s\n" % jid)
    failed = [(j, l) for j, l, s in rows if s == "FAILED"]
    blocked = [(j, l) for j, l, s in rows if s == "BLOCKED"]
    f.write("FAILED jobs now: %s\n" % (failed if failed else "none"))
    f.write("BLOCKED jobs now: %s\n" % (blocked if blocked else "none"))

# 3+4+9. trainings inventory
with open(os.path.join(OUT, "trainings_inventory.txt"), "w", encoding="utf-8") as f:
    for cd in sorted(os.listdir(os.path.join(S, "runs"))):
        cdir = os.path.join(S, "runs", cd)
        if not os.path.isdir(cdir):
            continue
        vals, seeds_ok, seeds_dir = [], [], []
        for sd in sorted(os.listdir(cdir)):
            if not sd.startswith("seed_"):
                continue
            seeds_dir.append(sd.replace("seed_", ""))
            ts = os.path.join(cdir, sd, "train_summary.json")
            if os.path.exists(ts):
                j = json.load(open(ts, encoding="utf-8"))
                vals.append(float(j["best_val"]))
                seeds_ok.append(sd.replace("seed_", ""))
        mean = sum(vals) / len(vals) if vals else float("nan")
        f.write("config=%s seeds_with_dirs=%s seeds_with_summary=%d best_val_mean=%.4f vals=%s\n"
                % (cd, ",".join(seeds_dir), len(seeds_ok), mean,
                   ",".join("%.4f" % v for v in vals)))

# 5+8. splits + addenda scan
with open(os.path.join(OUT, "splits_and_addenda.txt"), "w", encoding="utf-8") as f:
    val_n = len(glob.glob(os.path.join(S, "results", "val", "*.json")))
    ev_n = len(glob.glob(os.path.join(S, "results", "eval", "*.json")))
    sealed = [p for p in glob.glob(os.path.join(S, "results", "**", "*.json"), recursive=True)
              if "sealed" in p.lower()]
    f.write("results\\val json count: %d\n" % val_n)
    f.write("results\\eval json count (split=sealed would land here): %d\n" % ev_n)
    f.write("sealed-marked result files: %d\n" % len(sealed))
    f.write("PREREG.sha256 exists: %s\n" % os.path.exists(os.path.join(S, "PREREG.sha256")))
    f.write("PREREG2.sha256 exists: %s\n" % os.path.exists(os.path.join(S, "PREREG2.sha256")))
    f.write("selection.json exists: %s\n" % os.path.exists(os.path.join(S, "configs", "frozen", "selection.json")))
    f.write("qphysics_ext.py exists: %s\n" % os.path.exists(os.path.join(S, "qphysics_ext.py")))
    f.write("unchain.py exists: %s\n" % os.path.exists(os.path.join(W, "tools", "unchain.py")))
    for name, path in (("evaluate.py", os.path.join(S, "evaluate.py")),
                       ("train_ppo.py", os.path.join(S, "train_ppo.py")),
                       ("stress.py", os.path.join(S, "stress.py"))):
        t = open(path, encoding="utf-8").read()
        for key in ("CVaR95", "store_failures", "RSS_MB", "snapshot", "init_from",
                    "dr_support", '"algo"'):
            if key in ("CVaR95", "store_failures") and name != "evaluate.py":
                continue
            if key in ("RSS_MB", "snapshot", "init_from", '"algo"') and name != "train_ppo.py":
                continue
            if key == "dr_support" and name != "stress.py":
                continue
            f.write("%s contains %r: %s\n" % (name, key, key in t))

# 7. disk free
import shutil
with open(os.path.join(OUT, "disk_free.txt"), "w", encoding="utf-8") as f:
    f.write("free_disk_GB=%.1f (minimum 40 GB per Stage 0.4)\n"
            % (shutil.disk_usage("C:\\").free / 1e9))

# 6. db row counts
con = sqlite3.connect(os.path.join(S, "db", "qstudy.sqlite"))
with open(os.path.join(OUT, "db_rows.txt"), "w", encoding="utf-8") as f:
    for t in ["evals", "validation", "runs", "contrasts", "anomalies"]:
        n = con.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0]
        f.write("table %s rows=%d\n" % (t, n))
con.close()
print("evidence written to", OUT)
