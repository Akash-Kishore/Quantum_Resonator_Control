"""P3.2/P3.3/P3.4 (CONTINUATION5): smoke eval, Aer smoke, unit-suite jobs.

Outputs under WORK\smoke (never STUDY\results). RL smoke evals depend on the
p3_smoke_rl__dither__plain training job. Suite job reuses the m2_suite cmd shape.
"""
import json, os

S = r"C:\Projects\MiniProject_Sem4\qstudy"
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
PY = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"
EV = os.path.join(W, "smoke", "p3_ev")
AE = os.path.join(W, "smoke", "p3_aer")
CPUS = ["cpu2", "cpu3", "cpu4"]
jobs = []


def evset(jid, lane, method, cfg, st, probe, eps, outdir, tag, deps=(), engine=None):
    cmd = ('"%s" tools\\eval_set.py --method %s --cfg configs\\%s --set %s --probe %s '
           '--split val --episodes %d --seed_base 500000 --outdir "%s" --tag %s %s> "%s" 2>&1'
           % (PY, method, cfg, st, probe, eps, outdir, tag,
              ("--engine %s " % engine) if engine else "",
              os.path.join(W, "results", jid + ".log")))
    return dict(id=jid, lane=lane, cwd=S, cmd=cmd, depends_on=list(deps),
                expected_outputs=[os.path.join(outdir, "%s__%s.DONE.json" % (tag, st))],
                env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                max_retries=1, stall_minutes=60, max_wall_hours=4, watch=[])


jobs.append(evset("p3_ev_rl_nom", CPUS[0], "rl:runs_smoke\\rl__dither__plain",
                  "evalcfg__rl__dither__plain.json", "nominal", "dither", 5, EV, "p3rl",
                  deps=("p3_smoke_rl__dither__plain",)))
jobs.append(evset("p3_ev_rl_ofat", CPUS[1], "rl:runs_smoke\\rl__dither__plain",
                  "evalcfg__rl__dither__plain.json", "ofat", "dither", 5, EV, "p3rl",
                  deps=("p3_smoke_rl__dither__plain",)))
jobs.append(evset("p3_ev_pid_ofat", CPUS[2], "ctrl:pid", "eval__dither.json",
                  "ofat", "dither", 5, EV, "p3pid"))
jobs.append(evset("p3_ev_bayes_grids", CPUS[0], "ctrl:bayes:nominal", "eval__ramsey_iq.json",
                  "grids", "ramsey_iq", 5, EV, "p3bayes"))
jobs.append(evset("p3_aer_pid_aer6", "aer", "ctrl:pid", "eval__dither.json",
                  "aer6", "dither", 2, AE, "p3aer", engine="aer"))
suite_log = os.path.join(W, "results", "p3_suite.txt")
jobs.append(dict(id="p3_suite", lane="cpu1", cwd=r"C:\Projects\MiniProject_Sem4",
                 cmd='"%s" -m unittest discover qstudy\\tests -p "test_*.py" -v > "%s" 2>&1'
                     % (PY, suite_log),
                 depends_on=[], expected_outputs=[suite_log],
                 env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                 max_retries=1, stall_minutes=120, max_wall_hours=6, watch=[]))
with open(os.path.join(W, "jobs_json", "p3_smoke_eval.json"), "w", encoding="utf-8") as f:
    json.dump(jobs, f, indent=1)
print("p3_smoke_eval.json", len(jobs))
