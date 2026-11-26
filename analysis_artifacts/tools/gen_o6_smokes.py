"""Submit the O6/A8 smoke trainings (<= 20000 steps, cpu lanes) that also recover the wiped
Stage 10.2/11.2 smokes (A9). Jobs: ppo+init_from+snapshots, sac, td3, DRcurr curriculum smoke,
rabi K4 stack smoke, ramsey_iq gru8 smoke."""
import json
import os
import sys

w = sys.argv[1]
py = sys.argv[2]
s = r"C:\Projects\MiniProject_Sem4\qstudy"
sys.path.insert(0, os.path.join(w, "tools"))
import jobtools

SMOKES = [
    ("o6smoke_ppo_initfrom_snap", "smoke__ppo__initfrom_snapshots.json", "smoke__ppo__initfrom_snap"),
    ("o6smoke_sac", "smoke__sac__dither.json", "smoke__sac__dither"),
    ("o6smoke_td3", "smoke__td3__dither.json", "smoke__td3__dither"),
    ("o6smoke_drcurr_curriculum", "rl__dither__DRcurr__smoke.json", "smoke__drcurr_curriculum"),
    ("o6smoke_k4_stack", "rl__rabi__K4__smoke.json", "smoke__k4_stack"),
    ("o6smoke_gru8", "rl__ramsey_iq__gru8__smoke.json", "smoke__gru8"),
]
jobs = []
for jid, cfg, out in SMOKES:
    outdir = os.path.join(s, "runs", out, "seed_0")
    cmd = '"%s" train_ppo.py --config configs\\%s --seed 0 --out "%s"' % (py, cfg, outdir)
    exp = [os.path.join(outdir, "best_model.zip"), os.path.join(outdir, "train_summary.json")]
    if "initfrom" in jid:
        exp.append(os.path.join(outdir, "snapshot_10000.zip"))
        exp.append(os.path.join(outdir, "vecnorm_10000.pkl"))
    jobs.append(dict(id=jid, lane="cpu2", cwd=s, cmd=cmd, depends_on=[],
                     expected_outputs=exp, env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                     max_retries=1, stall_minutes=45, max_wall_hours=2))
with open(os.path.join(w, "jobs_json", "o6_smokes.json"), "w") as f:
    json.dump(jobs, f, indent=1)
print("wrote", len(jobs), "smoke jobs")
