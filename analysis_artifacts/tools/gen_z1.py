"""M5 z1: Part 1 Stage 12 confirmatory queue (post-freeze).
- configs\\frozen\\<name>.json for the nine confirmatory configs (frozen hyperparams:
  lr 3e-4, ent_coef 0.0, gamma 0.99, 2M steps; env/memory identical to the exploratory
  definitions).
- z1_train.json : 9 configs x seeds 101-110, gpu lane, no depends_on.
- z1_eval.json  : per model 3 sealed set-jobs (nominal 1000 eps, ofat 300, grids 100,
  seed base 900000), cpu lanes, each depends ONLY on its own training job.
- z1_base.json  : baselines x probe x 3 sets, cpu lanes, no deps.
- z1_aer.json   : Part 1 12.3 Aer subset (seeds 101-103, C1-C6, 7 cells, 100 eps,
  aer lane) + baseline Aer rows.
"""
import json, os

S = r"C:\Projects\MiniProject_Sem4\qstudy"
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
PY = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"
CPUS = ["cpu1", "cpu2", "cpu3", "cpu4"]
NINE = ["rl__rabi__plain", "rl__rabi__gru8", "rl__dither__plain", "rl__dither__filter",
        "rl__dither__DR", "rl__dither__DRcurr", "rl__dither__full",
        "rl__ramsey_iq__plain", "rl__ramsey_iq__gru8"]
SEEDS = list(range(101, 111))
PROBE_OF = {"rl__rabi__plain": "rabi", "rl__rabi__gru8": "rabi",
            "rl__dither__plain": "dither", "rl__dither__filter": "dither",
            "rl__dither__DR": "dither", "rl__dither__DRcurr": "dither",
            "rl__dither__full": "dither", "rl__ramsey_iq__plain": "ramsey_iq",
            "rl__ramsey_iq__gru8": "ramsey_iq"}
BASE_EVAL = [("ctrl:pid", "dither", "eval__dither.json"),
             ("ctrl:pid", "ramsey_iq", "eval__ramsey_iq.json"),
             ("ctrl:hill", "rabi", "eval__rabi.json"),
             ("ctrl:bayes:nominal", "rabi", "eval__rabi.json"),
             ("ctrl:bayes:nominal", "dither", "eval__dither.json"),
             ("ctrl:bayes:nominal", "ramsey_iq", "eval__ramsey_iq.json"),
             ("ctrl:bayes:oracle", "rabi", "eval__rabi.json"),
             ("ctrl:bayes:oracle", "dither", "eval__dither.json"),
             ("ctrl:bayes:oracle", "ramsey_iq", "eval__ramsey_iq.json"),
             ("ctrl:noctrl", "rabi", "eval__rabi.json"),
             ("ctrl:noctrl", "dither", "eval__dither.json"),
             ("ctrl:noctrl", "ramsey_iq", "eval__ramsey_iq.json"),
             ("ctrl:oracle", "rabi", "eval__rabi.json"),
             ("ctrl:oracle", "dither", "eval__dither.json"),
             ("ctrl:oracle", "ramsey_iq", "eval__ramsey_iq.json")]
AER_BASE = [("ctrl:pid", "dither", "eval__dither.json"),
            ("ctrl:hill", "rabi", "eval__rabi.json"),
            ("ctrl:bayes:nominal", "rabi", "eval__rabi.json"),
            ("ctrl:bayes:nominal", "dither", "eval__dither.json"),
            ("ctrl:bayes:nominal", "ramsey_iq", "eval__ramsey_iq.json"),
            ("ctrl:bayes:oracle", "dither", "eval__dither.json"),
            ("ctrl:oracle", "dither", "eval__dither.json"),
            ("ctrl:noctrl", "rabi", "eval__rabi.json")]
C1_C6 = ["rl__rabi__plain", "rl__dither__plain", "rl__dither__filter",
         "rl__ramsey_iq__gru8", "rl__dither__DRcurr", "rl__dither__full"]

# ---------------- frozen configs ----------------
os.makedirs(os.path.join(S, "configs", "frozen"), exist_ok=True)
for name in NINE:
    spec = json.load(open(os.path.join(S, "configs", name + ".json"), encoding="utf-8"))
    spec.setdefault("ppo", {})["learning_rate"] = 0.0003
    spec["ppo"]["ent_coef"] = 0.0
    spec["ppo"]["gamma"] = 0.99
    spec["total_steps"] = 2000000
    with open(os.path.join(S, "configs", "frozen", name + ".json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)

train, ev, base, aer = [], [], [], []
i = 0

def train_id(s, name):
    """D2: wave id, seed-major (z1a seeds 101-105, z1b seeds 106-110)."""
    return ("z1a" if s <= 105 else "z1b") + "_%d_%s" % (s, name)

for s in SEEDS:
    for name in NINE:
        jid = train_id(s, name)
        outdir = os.path.join(S, "runs", name, "seed_%d" % s)
        train.append(dict(id=jid, lane="gpu", cwd=S,
                         cmd='"%s" train_ppo.py --config configs\\frozen\\%s.json --seed %d --out "%s" > "%s" 2>&1'
                             % (PY, name, s, outdir, os.path.join(W, "results", jid + ".log")),
                         depends_on=[],
                         expected_outputs=[os.path.join(outdir, "best_model.zip"),
                                           os.path.join(outdir, "vec_normalize.pkl"),
                                           os.path.join(outdir, "train_summary.json")],
                         env={}, max_retries=1, stall_minutes=90, max_wall_hours=24,
                         watch=[os.path.join(outdir, "progress.log")]))
        for st, eps in (("nominal", 1000), ("ofat", 300), ("grids", 100)):
            ejid = "z1e_%s_%s_s%d" % (st, name, s)
            tag = "z1__%s_s%d" % (name, s)
            ev.append(dict(id=ejid, lane=CPUS[i % 4], cwd=S,
                          cmd='"%s" tools\\eval_set.py --method rl:runs\\%s\\seed_%d '
                              '--cfg configs\\evalcfg__%s.json --set %s --probe %s --split sealed '
                              '--episodes %d --seed_base 900000 --outdir results\\sealed --tag %s '
                              '> "%s" 2>&1'
                              % (PY, name, s, name, st, PROBE_OF[name], eps, tag,
                                 os.path.join(W, "results", ejid + ".log")),
                          depends_on=[jid],
                          expected_outputs=[os.path.join(S, "results", "sealed", tag + "__" + st + ".DONE.json")],
                          env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                          max_retries=1, stall_minutes=240, max_wall_hours=48, watch=[]))
        i += 1

for meth, probe, cfgf in BASE_EVAL:
    for st, eps in (("nominal", 1000), ("ofat", 300), ("grids", 100)):
        bjid = "z1v_%s_%s_%s" % (st, meth.replace(":", "_"), probe)
        tag = "z1__base_%s_%s" % (meth.replace(":", "_"), probe)
        base.append(dict(id=bjid, lane=CPUS[i % 4], cwd=S,
                        cmd='"%s" tools\\eval_set.py --method %s --cfg configs\\%s --set %s '
                            '--probe %s --split sealed --episodes %d --seed_base 900000 '
                            '--outdir results\\sealed --tag %s > "%s" 2>&1'
                            % (PY, meth, cfgf, st, probe, eps, tag,
                               os.path.join(W, "results", bjid + ".log")),
                         depends_on=[],
                         expected_outputs=[os.path.join(S, "results", "sealed", tag + "__" + st + ".DONE.json")],
                        env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                        max_retries=1, stall_minutes=240, max_wall_hours=48, watch=[]))
        i += 1

for name in C1_C6:
    for s in (101, 102, 103):
        ajid = "z1x_%s_s%d" % (name, s)
        tag = "z1aer__%s_s%d" % (name, s)
        aer.append(dict(id=ajid, lane="aer", cwd=S,
                        cmd='"%s" tools\\eval_set.py --method rl:runs\\%s\\seed_%d '
                            '--cfg configs\\evalcfg__%s.json --set aer6 --probe %s --split sealed '
                            '--episodes 100 --seed_base 900000 --outdir results\\sealed --tag %s '
                            '--engine aer > "%s" 2>&1'
                            % (PY, name, s, name, PROBE_OF[name], tag,
                               os.path.join(W, "results", ajid + ".log")),
                        depends_on=[train_id(s, name)],
                        expected_outputs=[os.path.join(S, "results", "sealed", tag + "__aer6.DONE.json")],
                        env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                        max_retries=1, stall_minutes=240, max_wall_hours=48, watch=[]))
for meth, probe, cfgf in AER_BASE:
    ajid = "z1x_base_%s_%s" % (meth.replace(":", "_"), probe)
    tag = "z1aer__base_%s_%s" % (meth.replace(":", "_"), probe)
    aer.append(dict(id=ajid, lane="aer", cwd=S,
                    cmd='"%s" tools\\eval_set.py --method %s --cfg configs\\%s --set aer6 '
                        '--probe %s --split sealed --episodes 100 --seed_base 900000 '
                        '--outdir results\\sealed --tag %s --engine aer > "%s" 2>&1'
                        % (PY, meth, cfgf, probe, tag, os.path.join(W, "results", ajid + ".log")),
                    depends_on=[],
                    expected_outputs=[os.path.join(S, "results", "sealed", tag + "__aer6.DONE.json")],
                    env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    max_retries=1, stall_minutes=240, max_wall_hours=48, watch=[]))

counts = {}
for fn, jobs in (("z1_train.json", train), ("z1_eval.json", ev),
                 ("z1_base.json", base), ("z1_aer.json", aer)):
    with open(os.path.join(W, "jobs_json", fn), "w") as f:
        json.dump(jobs, f, indent=1)
    counts[fn] = len(jobs)
    print(fn, len(jobs))
with open(os.path.join(W, "jobs_json", "z1_counts.json"), "w") as f:
    json.dump(counts, f, indent=1)
