#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs8_snapshot.py (v8) - READ-ONLY on the project. Makes one zip with the project's CODE, CONFIGS and DOCUMENTS only
(no runs, results, databases, queue, logs) plus a manifest CSV with the sha256 of every file.
Use: (1) as the immutable 'as-run' record of what produced the paper's numbers, (2) as the bundle the controller reviews.
Usage: python qs8_snapshot.py --roots R1 R2 --out AS_RUN_v1.zip"""
import os, re, sys, csv, zipfile, hashlib, argparse, time

SKIP = {"__pycache__", ".git", ".pytest_cache", "site-packages", "venv", ".venv", "graveyard", "quarantine", "backup",
        "_LEGACY_MiniProject_Sem4", "node_modules", "runs", "runs_smoke", "results", "results_smoke", "queue", "jobs_json",
        "logs", "smoke", "smoke_cfgs", "data", "db"}
CODE = {".py", ".ps1", ".bat", ".cmd", ".toml", ".yaml", ".yml", ".cfg", ".ini", ".md", ".txt", ".sha256"}

def keep(p):
    ext = os.path.splitext(p)[1].lower()
    pl = p.replace("\\", "/").lower()
    try:
        sz = os.path.getsize(p)
    except OSError:
        return False
    if ext == ".json":
        return sz < 2 * 1024 * 1024 and bool(re.search(r"/configs?/|pid_gains", pl))
    return ext in CODE and sz < 5 * 1024 * 1024

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n, tot = 0, 0
    man = os.path.splitext(a.out)[0] + ".manifest.csv"
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z, open(man, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["path_in_zip", "bytes", "sha256"])
        for root in a.roots:
            base = os.path.basename(os.path.normpath(root))
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if d not in SKIP]
                for name in sorted(fn):
                    p = os.path.join(dp, name)
                    if not keep(p):
                        continue
                    arc = base + "/" + os.path.relpath(p, root).replace("\\", "/")
                    h = hashlib.sha256()
                    with open(p, "rb") as f:
                        data = f.read()
                    h.update(data)
                    z.writestr(zipfile.ZipInfo.from_file(p, arc), data, zipfile.ZIP_DEFLATED)
                    w.writerow([arc, len(data), h.hexdigest()])
                    n += 1
                    tot += len(data)
    zh = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print("SNAPSHOT files=%d bytes=%d zip=%s" % (n, tot, a.out))
    print("ZIP_SHA256 " + zh)
    print("MANIFEST " + man)
    print("QS8-SNAPSHOT-DONE")

if __name__ == "__main__":
    main()
