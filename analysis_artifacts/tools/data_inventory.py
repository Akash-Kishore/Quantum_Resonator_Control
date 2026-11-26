import json, os, sys
import numpy as np

ROOT = r"C:\Projects\MiniProject_Sem4"
OUT = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\understanding\data_inventory"
os.makedirs(OUT, exist_ok=True)
SUMMARY = os.path.join(OUT, "data_summary.md")
SKIP = {".git", "__pycache__", ".opencode", "qstudy"}
lines = ["# Data inventory (Stage 1.5) — structure, first records, sizes", ""]

def describe_json(p, lines):
    rel = os.path.relpath(p, ROOT)
    try:
        j = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        lines.append("## %s\n- PARSE ERROR: %s\n" % (rel, type(e).__name__))
        return
    size = os.path.getsize(p)
    lines.append("## %s (%d B)" % (rel, size))
    if isinstance(j, dict):
        lines.append("- keys: %s" % sorted(j.keys())[:40])
        for k in sorted(j.keys())[:12]:
            v = j[k]
            if isinstance(v, (int, float, str, bool)) or v is None:
                lines.append("  - %s = %s" % (k, str(v)[:120]))
            elif isinstance(v, list):
                lines.append("  - %s = list len %d, first 3: %s" % (k, len(v), str(v[:3])[:200]))
            elif isinstance(v, dict):
                lines.append("  - %s = dict keys %s" % (k, sorted(v.keys())[:15]))
    elif isinstance(j, list):
        lines.append("- list len %d, first 3: %s" % (len(j), str(j[:3])[:300]))
    lines.append("")

def describe_npz(p, lines):
    rel = os.path.relpath(p, ROOT)
    lines.append("## %s (%d B)" % (rel, os.path.getsize(p)))
    try:
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            a = z[k]
            flat = np.ravel(a)[:3]
            lines.append("  - %s: shape %s dtype %s first3 %s" % (k, a.shape, a.dtype, str(flat)[:120]))
    except Exception as e:
        lines.append("  - LOAD ERROR: %s" % type(e).__name__)
    lines.append("")

def describe_csv(p, lines):
    rel = os.path.relpath(p, ROOT)
    lines.append("## %s (%d B)" % (rel, os.path.getsize(p)))
    try:
        import csv
        with open(p, encoding="utf-8", errors="replace", newline="") as f:
            r = list(csv.reader(f))
        lines.append("  - rows %d; header: %s" % (len(r), r[0][:12] if r else "EMPTY"))
        for row in r[1:4]:
            lines.append("  - row: %s" % str(row[:8])[:180])
    except Exception as e:
        lines.append("  - READ ERROR: %s" % type(e).__name__)
    lines.append("")

def describe_pkl(p, lines):
    rel = os.path.relpath(p, ROOT)
    lines.append("## %s (%d B) [pickle - type recorded]" % (rel, os.path.getsize(p)))
    try:
        import pickle
        o = pickle.load(open(p, "rb"))
        lines.append("  - type: %s" % type(o).__name__)
        if hasattr(o, "shape"):
            lines.append("  - shape/attrs: %s" % str(getattr(o, "shape", ""))[:100])
    except Exception as e:
        lines.append("  - LOAD ERROR: %s" % type(e).__name__)
    lines.append("")

n = {"json": 0, "npz": 0, "csv": 0, "pkl": 0, "zip": 0}
for d, ds, fs in os.walk(ROOT):
    ds[:] = [x for x in ds if x not in SKIP]
    for f in fs:
        ext = os.path.splitext(f)[1].lower()
        p = os.path.join(d, f)
        if ext == ".json":
            describe_json(p, lines); n["json"] += 1
        elif ext == ".npz":
            describe_npz(p, lines); n["npz"] += 1
        elif ext == ".csv":
            describe_csv(p, lines); n["csv"] += 1
        elif ext == ".pkl":
            describe_pkl(p, lines); n["pkl"] += 1
        elif ext == ".zip":
            n["zip"] += 1

open(SUMMARY, "w", encoding="utf-8").write("\n".join(lines))
print("SUMMARY counts:", n)

# checkpoint folders listing
ck = ["# Checkpoint folders (trained_models + qiskit)\n"]
for base in (os.path.join(ROOT, "rl_training", "trained_models"), os.path.join(ROOT, "data_logs", "qiskit_aer_simulation_2")):
    ck.append("## %s\n" % base)
    for fam in sorted(os.listdir(base)):
        fp = os.path.join(base, fam)
        if not os.path.isdir(fp):
            continue
        ck.append("- %s: %s" % (fam, sorted(os.listdir(fp))[:10]))
        for sub in sorted(os.listdir(fp)):
            sp = os.path.join(fp, sub)
            if os.path.isdir(sp):
                files = sorted(os.listdir(sp))
                total = sum(os.path.getsize(os.path.join(sp, x)) for x in files)
                ck.append("  - %s: %d files, %d B: %s" % (sub, len(files), total, files[:8]))
open(os.path.join(OUT, "checkpoints.md"), "w", encoding="utf-8").write("\n".join(ck))
print("checkpoints.md written")
