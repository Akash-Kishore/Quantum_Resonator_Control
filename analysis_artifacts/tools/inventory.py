import csv, os, sys
ROOT = r"C:\Projects\MiniProject_Sem4"
OUT = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\understanding\inventory.csv"
SKIP = {".git", "__pycache__", "node_modules", ".opencode"}
CODE = {".py", ".ipynb"}
DOC = {".md", ".txt", ".docx", ".pdf", ".yml", ".yaml", ".cfg", ".ini", ".json"}
DATA = {".csv", ".npz", ".npy", ".xlsx"}
CKPT = {".zip", ".pkl"}
FIG = {".png", ".jpg", ".svg", ".jpeg"}
CIRC = {".asc", ".cir", ".net", ".spice"}
FW = {".ino", ".cpp", ".h"}
rows = []
for d, ds, fs in os.walk(ROOT):
    ds[:] = [x for x in ds if x not in SKIP and not os.path.exists(os.path.join(d, x, "pyvenv.cfg")) and not os.path.exists(os.path.join(d, x, "conda-meta"))]
    for f in fs:
        p = os.path.join(d, f)
        rel = os.path.relpath(p, ROOT)
        ext = os.path.splitext(f)[1].lower()
        try: st = os.stat(p)
        except OSError: continue
        if ext in CODE: cat = "code"
        elif ext in DOC: cat = "doc"
        elif ext in DATA: cat = "data"
        elif "trained_models" in rel or ext in CKPT: cat = "checkpoint"
        elif ext in FIG: cat = "figure"
        elif ext in CIRC: cat = "circuit"
        elif ext in FW: cat = "firmware"
        else: cat = "other"
        rows.append([rel, st.st_size, st.st_mtime, ext or "none", cat])
rows.sort()
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["rel", "size", "mtime", "ext", "category"]); w.writerows(rows)
from collections import Counter
print("FILES", len(rows)); print(Counter(r[4] for r in rows)); print(Counter(r[3] for r in rows).most_common(20))
