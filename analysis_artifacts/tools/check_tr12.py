#!/usr/bin/env python
"""Check FTB_200k P2T values and do TR-2 sanity check."""
import json, os, numpy as np
from scipy import stats as sps

SEALED = r"C:\Projects\MiniProject_Sem4\qstudy\results\sealed"
SEEDS = [200, 201, 202, 203, 204]

def p2t_for_arm_seed(prefix, seed, step_suffix=""):
    """step_suffix: e.g., 's200000__' for E4 snapshots, '' for final models"""
    full_prefix = "%s%d__RB__%sofat_" % (prefix, seed, step_suffix)
    files = []
    for fn in os.listdir(SEALED):
        if fn.startswith(full_prefix) and fn.endswith(".json") and not fn.endswith(".DONE.json"):
            files.append(fn)
    files.sort()
    vals = []
    for fn in files:
        fp = os.path.join(SEALED, fn)
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("status") != "ok":
            continue
        vals.append(float(d["lock_w1000_mean"]))
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(files)

# FTB_200k uses z2__ prefix for final model evals
ftb200_p2t = {}
for s in SEEDS:
    v, n = p2t_for_arm_seed("z2__s15__FTB_200000_s", s, "")
    ftb200_p2t[s] = v
    print("Seed %d: FTB_200k P2T=%.4f (%d cells)" % (s, v, n))

# E4 200k snapshot uses z2s__ prefix with step suffix
e4_200k_p2t = {}
for s in SEEDS:
    v, n = p2t_for_arm_seed("z2s__s15__E4_s", s, "s200000__")
    e4_200k_p2t[s] = v
    print("Seed %d: E4_200k P2T=%.4f (%d cells)" % (s, v, n))

# TR-2: FTB_200k vs E4 200k snapshot
a = [ftb200_p2t[s] for s in SEEDS]
b = [e4_200k_p2t[s] for s in SEEDS]

def mannwhitney_exact(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        u, p = sps.mannwhitneyu(a, b, alternative="two-sided", method="exact")
    except Exception:
        u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)

u, p = mannwhitney_exact(a, b)
diff = float(np.mean(a) - np.mean(b))
print("\nTR-2 (FTB_200k vs E4 200k snapshot):")
print("FTB_200k mean=%.4f sd=%.4f" % (np.mean(a), np.std(a, ddof=1)))
print("E4_200k  mean=%.4f sd=%.4f" % (np.mean(b), np.std(b, ddof=1)))
print("diff=%.4f U=%s p=%.6g" % (diff, u, p))
print("Expected from S7: meanA=0.4104 meanB=0.2658 diff=0.1446 U=25 p=0.00794")

# Also check TR-1
# E1 uses z2__ prefix
e1_p2t = {}
e4_2m_p2t = {}
for s in SEEDS:
    v1, _ = p2t_for_arm_seed("z2__s15__E1_s", s, "")
    v4, _ = p2t_for_arm_seed("z2__s15__E4_s", s, "")
    e1_p2t[s] = v1
    e4_2m_p2t[s] = v4
    print("Seed %d: E1 P2T=%.4f, E4_2M P2T=%.4f" % (s, v1, v4))

a1 = [e1_p2t[s] for s in SEEDS]
b1 = [e4_2m_p2t[s] for s in SEEDS]
u1, p1 = mannwhitney_exact(a1, b1)
diff1 = float(np.mean(a1) - np.mean(b1))
print("\nTR-1 (E1 vs E4 2M):")
print("E1 mean=%.4f sd=%.4f" % (np.mean(a1), np.std(a1, ddof=1)))
print("E4_2M mean=%.4f sd=%.4f" % (np.mean(b1), np.std(b1, ddof=1)))
print("diff=%.4f U=%s p=%.6g (expected diff=-0.3136 p=0.00794)" % (diff1, u1, p1))