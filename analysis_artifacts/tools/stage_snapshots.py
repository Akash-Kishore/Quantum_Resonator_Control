"""S2b: stage E4 snapshot model dirs by COPY (never move, never touch runs\).

For each step in {50000,200000,500000,1000000} x seed in {200..204}:
  dest  qstudy\runs_snap\s15__E4_s<step>\seed_<n>\
  best_model.zip   = copy of runs\s15__E4\seed_<n>\snapshot_<step>.zip
  vec_normalize.pkl= copy of runs\s15__E4\seed_<n>\vecnorm_<step>.pkl
  progress.log / train_summary.json copied from runs\s15__E4\seed_<n>\
Writes WORK\audit\s2b_snapshot_stage_manifest.json proving sha256(copy)==sha256(source).
"""
import hashlib, json, os, shutil, sys

STUDY = r"C:\Projects\MiniProject_Sem4\qstudy"
WORK = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
RUNS = os.path.join(STUDY, "runs", "s15__E4")
SNAP_ROOT = os.path.join(STUDY, "runs_snap")
STEPS = [50000, 200000, 500000, 1000000]
SEEDS = [200, 201, 202, 203, 204]
META_FILES = ["progress.log", "train_summary.json"]
MANIFEST = os.path.join(WORK, "audit", "s2b_snapshot_stage_manifest.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    entries = []
    ok = True
    for step in STEPS:
        for seed in SEEDS:
            sdir = os.path.join(RUNS, "seed_%d" % seed)
            ddir = os.path.join(SNAP_ROOT, "s15__E4_s%d" % step, "seed_%d" % seed)
            os.makedirs(ddir, exist_ok=True)
            pairs = [
                ("snapshot_%d.zip" % step, "best_model.zip"),
                ("vecnorm_%d.pkl" % step, "vec_normalize.pkl"),
            ]
            for src_name, dst_name in pairs:
                src = os.path.join(sdir, src_name)
                dst = os.path.join(ddir, dst_name)
                if not os.path.exists(src):
                    print("MISSING SOURCE %s" % src); ok = False; continue
                shutil.copyfile(src, dst)
                entries.append({
                    "step": step, "seed": seed,
                    "dst": dst, "src": src,
                    "sha_dst": sha256_file(dst), "sha_src": sha256_file(src),
                    "match": sha256_file(dst) == sha256_file(src),
                })
            for mf in META_FILES:
                src = os.path.join(sdir, mf)
                dst = os.path.join(ddir, mf)
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
                    entries.append({
                        "step": step, "seed": seed, "meta": mf,
                        "dst": dst, "src": src,
                        "sha_dst": sha256_file(dst), "sha_src": sha256_file(src),
                        "match": sha256_file(dst) == sha256_file(src),
                    })
    all_match = all(e.get("match", False) for e in entries)
    manifest = {"steps": STEPS, "seeds": SEEDS, "n_entries": len(entries),
                "all_sha256_match": bool(all_match and ok), "entries": entries}
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("staged entries=%d all_match=%s ok=%s" % (len(entries), all_match, ok))
    print("manifest -> %s" % MANIFEST)
    sys.exit(0 if (all_match and ok) else 1)


if __name__ == "__main__":
    main()
