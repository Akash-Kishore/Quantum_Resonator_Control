"""F2: Stage FTB last-step models into runs_snap\s15__FTB_<budget>_last\seed_<n>\.
Copy only (never move/edit runs\). Files per staged dir:
  best_model.zip      = copy of source best_model.zip (last-step model)
  vec_normalize.pkl   = copy of source vec_normalize.pkl (matching norm)
  progress.log        = copy from source
  train_summary.json  = copy from source
Write sha256 manifest of source vs copy for staged files only.
New files only under qstudy\runs_snap\ and WORK\. Never touch runs\.
"""
import hashlib, json, os, shutil, sys

BASE = r"C:\Projects\MiniProject_Sem4\qstudy"
RUNS = os.path.join(BASE, "runs")
RUNS_SNAP = os.path.join(BASE, "runs_snap")
WORK_AUDIT = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\audit"
MANIFEST = os.path.join(WORK_AUDIT, "s2f_ftb_last_stage_manifest.json")

budgets = [50000, 200000, 500000, 1000000]
seeds = [200, 201, 202, 203, 204]
STAGED_FILES = ["best_model.zip", "vec_normalize.pkl", "progress.log", "train_summary.json"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    manifest = {"n_staged_dirs": 0, "n_files": 0, "all_sha256_match": True, "entries": []}
    n_dirs = 0
    n_files = 0
    for b in budgets:
        for s in seeds:
            src_dir = os.path.join(RUNS, "s15__FTB_%d" % b, "seed_%d" % s)
            dst_dir = os.path.join(RUNS_SNAP, "s15__FTB_%d_last" % b, "seed_%d" % s)
            if not os.path.isdir(src_dir):
                print("SOURCE MISSING:", src_dir)
                sys.exit(1)
            if os.path.exists(dst_dir):
                print("DEST EXISTS (refusing to overwrite):", dst_dir)
                sys.exit(1)
            os.makedirs(dst_dir, exist_ok=True)
            n_dirs += 1
            for fn in STAGED_FILES:
                src = os.path.join(src_dir, fn)
                dst = os.path.join(dst_dir, fn)
                if not os.path.isfile(src):
                    print("SOURCE FILE MISSING:", src)
                    sys.exit(1)
                shutil.copy2(src, dst)
                src_sha = sha256_file(src)
                dst_sha = sha256_file(dst)
                match = src_sha == dst_sha
                if not match:
                    manifest["all_sha256_match"] = False
                manifest["entries"].append({
                    "budget": b, "seed": s, "file": fn,
                    "src": src, "dst": dst,
                    "src_sha256": src_sha, "dst_sha256": dst_sha, "match": match,
                })
                n_files += 1
    manifest["n_staged_dirs"] = n_dirs
    manifest["n_files"] = n_files
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("staged dirs:", n_dirs, "files:", n_files)
    print("all_sha256_match:", manifest["all_sha256_match"])
    print("manifest:", MANIFEST)
    # print counts per budget
    from collections import Counter
    c = Counter(e["budget"] for e in manifest["entries"])
    for b in budgets:
        print("  FTB %d_last: %d files" % (b, c[b]))


if __name__ == "__main__":
    main()
