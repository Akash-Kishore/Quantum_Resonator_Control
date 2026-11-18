import os
import sys

w = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
sys.path.insert(0, os.path.join(w, "tools"))
import jobtools

s = r"C:\Projects\MiniProject_Sem4\qstudy"
py = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"
outdir = os.path.join(s, "runs", "smoke__td3__dither", "seed_0")
cmd = '"%s" train_ppo.py --config configs\\smoke__td3__dither.json --seed 0 --out "%s"' % (py, outdir)
print(jobtools.submit(w, "o6smoke_td3", "cpu2", s, cmd,
                      expected_outputs=[os.path.join(outdir, "best_model.zip"),
                                        os.path.join(outdir, "train_summary.json")],
                      env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                      max_retries=1, stall_minutes=45, max_wall_hours=2))
