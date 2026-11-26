"""F1 read-only: verify last-step model + matching vecnorm exist for all FTB dirs."""
import os
BASE = r"C:\Projects\MiniProject_Sem4\qstudy\runs"
budgets = [50000, 200000, 500000, 1000000]
seeds = [200, 201, 202, 203, 204]
missing = []
total = 0
for b in budgets:
    for s in seeds:
        d = os.path.join(BASE, "s15__FTB_%d" % b, "seed_%d" % s)
        total += 1
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        has_best = "best_model.zip" in files
        has_vn = "vec_normalize.pkl" in files
        step_snaps = [f for f in files if f.startswith("snapshot_") and f.endswith(".zip")]
        step_vns = [f for f in files if f.startswith("vecnorm_") and f.endswith(".pkl")]
        status = "OK" if (has_best and has_vn) else "MISSING"
        print("FTB %6d seed %-3d %-3s files=%s step_snaps=%s step_vns=%s" % (b, s, status, files, step_snaps, step_vns))
        if not (has_best and has_vn):
            missing.append((b, s, files))
print("\ntotal dirs checked:", total)
print("missing last-step model or vecnorm:", len(missing))
if missing:
    print("FT SNAPSHOTS NOT EVALUABLE")
    for m in missing:
        print("  ", m)
else:
    print("FT SNAPSHOTS EVALUABLE: all 20 dirs have best_model.zip + vec_normalize.pkl")
