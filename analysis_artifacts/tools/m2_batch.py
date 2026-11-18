"""CONTINUATION4 M2 batch: F5/F6/F7 job generation + tainted-file move + stale-row delete.

- Moves the F1/F2-tainted val result files (ctrl:bayes:*, RL __filter/__full families)
  to results\\val_INVALID_f1f2leak\\ and deletes their db rows (so re-ingest is clean).
- Emits job batches: m2f6_tune (6 gpu), m2f5_oracle (12 cpu), m2f7_bayes (24 cpu),
  m2f7_rl (100 cpu), m2_suite (full unit-test suite, cpu1).
Idempotent: re-running skips already-moved files and existing jobs.
"""
import glob, json, os, re, sqlite3, sys

S = r"C:\Projects\MiniProject_Sem4\qstudy"
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
PY = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"
VAL = os.path.join(S, "results", "val")
INVD = os.path.join(S, "results", "val_INVALID_f1f2leak")
VAL_CONDS = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]
PROBES = ["rabi", "dither", "ramsey_iq"]
CPUS = ["cpu1", "cpu2", "cpu3", "cpu4"]


def is_tainted(j):
    m = j.get("method") or ""
    md = j.get("model_dir") or ""
    return m.startswith("ctrl:bayes") or ("__filter" in md) or ("__full" in md)


def main():
    os.makedirs(INVD, exist_ok=True)
    moved, eids = [], []
    for p in sorted(glob.glob(os.path.join(VAL, "*.json"))):
        if "_INVALID" in p:
            continue
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if "method" not in j or "lock_w1000_mean" not in j or not is_tainted(j):
            continue
        eid = "%s|%s|%s|%s|%s" % (j.get("method"), j.get("model_dir") or "", j.get("cond_id"),
                                  j.get("split"), j.get("seed_base"))
        eids.append(eid)
        npz = p[:-5] + ".npz"
        moved.append(dict(json=os.path.basename(p), npz=os.path.basename(npz), eid=eid,
                          cmd=j.get("cmd"), method=j.get("method"), cond=j.get("cond_id")))
        os.replace(p, os.path.join(INVD, os.path.basename(p)))
        if os.path.exists(npz):
            os.replace(npz, os.path.join(INVD, os.path.basename(npz)))
    with open(os.path.join(W, "results", "f1f2_moved_manifest.json"), "w") as f:
        json.dump(moved, f, indent=1)
    con = sqlite3.connect(os.path.join(S, "db", "qstudy.sqlite"))
    nd = 0
    for eid in eids:
        cur = con.execute("DELETE FROM evals WHERE eval_id=?", (eid,))
        nd += cur.rowcount
    con.commit()
    con.close()
    print("MOVED %d tainted files (json+npz); DELETED %d db rows" % (len(moved), nd))

    def mk_job(jid, lane, cwd, cmd_args, out_abs, log_name, stall=90, watch=()):
        return dict(id=jid, lane=lane, cwd=cwd,
                    cmd='"%s" %s > "%s" 2>&1' % (PY, cmd_args, os.path.join(W, "results", log_name)),
                    depends_on=[], expected_outputs=[out_abs],
                    env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    max_retries=2, stall_minutes=stall, max_wall_hours=24, watch=list(watch))

    # ---- F6 tuning replicates (gpu, 1M steps, seeds 90-92) ----
    tune = []
    for cfgname in ("tune__lr3e4__ent0003__gam099", "tune__lr3e4__ent00__gam099"):
        for seed in (90, 91, 92):
            jid = "m2f6_tune_%s_s%d" % (cfgname.replace("tune__", ""), seed)
            outdir = os.path.join(S, "runs", cfgname, "seed_%d" % seed)
            tune.append(dict(id=jid, lane="gpu", cwd=S,
                             cmd='"%s" train_ppo.py --config configs\\%s.json --seed %d --out "%s" > "%s" 2>&1'
                                 % (PY, cfgname, seed, outdir,
                                    os.path.join(W, "results", jid + ".log")),
                             depends_on=[],
                             expected_outputs=[os.path.join(outdir, "best_model.zip"),
                                               os.path.join(outdir, "train_summary.json")],
                             env={}, max_retries=1, stall_minutes=60, max_wall_hours=24,
                             watch=[os.path.join(outdir, "progress.log")]))
    json.dump(tune, open(os.path.join(W, "jobs_json", "m2f6_tune.json"), "w"), indent=1)

    # ---- F5 ctrl:oracle reference rows (12 cpu) ----
    cfg_probe = {}
    for m in moved:
        if m["method"] == "ctrl:bayes:nominal":
            mm = re.search(r"--cfg (\S+)", m["cmd"])
            cfg_probe[re.search(r"__([a-z_0-9]+)__", os.path.basename(m["json"])).group(1)] = mm.group(1)
    oracle = []
    i = 0
    for probe in PROBES:
        for cond in VAL_CONDS:
            jid = "m2f5_oracle_%s_%s" % (probe, cond)
            out = os.path.join(S, "results", "val", "ctrl_oracle__%s__%s.json" % (probe, cond))
            oracle.append(mk_job(jid, CPUS[i % 4], S,
                                 "evaluate.py --method ctrl:oracle --cfg %s --cond %s --split val "
                                 "--episodes 200 --seed_base 500000 --out results\\val\\ctrl_oracle__%s__%s.json"
                                 % (cfg_probe[probe], cond, probe, cond), out, jid + ".log"))
            i += 1
    json.dump(oracle, open(os.path.join(W, "jobs_json", "m2f5_oracle.json"), "w"), indent=1)

    # ---- F7 re-runs (bayes 24 + RL filter/full 100) ----
    bayes, rl = [], []
    i = 0
    for m in moved:
        out_abs = os.path.join(S, "results", "val", m["json"])
        if m["method"].startswith("ctrl:bayes"):
            jid = "m2f7_bayes_%s__%s" % (m["method"].replace("ctrl:", "").replace(":", "_"), m["cond"])
            bayes.append(mk_job(jid, CPUS[i % 4], S, m["cmd"], out_abs, jid + ".log"))
            i += 1
        else:
            fam = "filter" if "__filter" in m["method"] else "full"
            base = re.sub(r"[^A-Za-z0-9_]", "_", m["json"])[:-5]
            jid = ("m2f7_rl%s_%s" % (fam, base))[:100]
            rl.append(mk_job(jid, CPUS[i % 4], S, m["cmd"], out_abs, jid + ".log"))
            i += 1
    json.dump(bayes, open(os.path.join(W, "jobs_json", "m2f7_bayes.json"), "w"), indent=1)
    json.dump(rl, open(os.path.join(W, "jobs_json", "m2f7_rl.json"), "w"), indent=1)

    # ---- A6 full unit-test suite ----
    suite = [dict(id="m2_suite_unittests", lane="cpu1", cwd=r"C:\Projects\MiniProject_Sem4",
                  cmd='"%s" -m unittest discover qstudy\\tests -p "test_*.py" -v > "%s" 2>&1'
                      % (PY, os.path.join(W, "results", "m2_suite_unittests.txt")),
                  depends_on=[], expected_outputs=[os.path.join(W, "results", "m2_suite_unittests.txt")],
                  env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                  max_retries=2, stall_minutes=45, max_wall_hours=24, watch=[])]
    json.dump(suite, open(os.path.join(W, "jobs_json", "m2_suite.json"), "w"), indent=1)
    print("JOB BATCHES: tune=%d oracle=%d bayes=%d rl=%d suite=1"
          % (len(tune), len(oracle), len(bayes), len(rl)))


if __name__ == "__main__":
    main()
