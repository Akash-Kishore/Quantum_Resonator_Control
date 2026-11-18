"""Stage 11.4 rerun + 11.6 combination runs: configs, evalcfgs, job batch."""
import json, os

S = r"C:\Projects\MiniProject_Sem4\qstudy"
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
PY = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"

# ---- 11.6 combination ("full") configs: best probe (dither, ramsey_iq by 4-cell val mean)
# + filter features + best memory (gru8) + best DR variant (DRcurr) + tuned PPO params
ppo = dict(n_envs=8, n_steps=256, batch_size=256, n_epochs=10, gamma=0.99,
           gae_lambda=0.95, clip_range=0.2, learning_rate=3e-4, ent_coef=0.003,
           vf_coef=0.5, max_grad_norm=0.5)
for probe in ("dither", "ramsey_iq"):
    spec = dict(ppo=dict(ppo), total_steps=2000000, val_every=100000,
                env=dict(probe=probe, stack=8, filter_features=True,
                         dr=dict(scale=1.0, curr=True)),
                memory=dict(type="gru", K=8))
    with open(os.path.join(S, "configs", "rl__%s__full.json" % probe), "w") as f:
        json.dump(spec, f, indent=1)
    ecfg = dict(probe=probe, stack=8, filter_features=True)
    with open(os.path.join(S, "configs", "evalcfg__rl__%s__full.json" % probe), "w") as f:
        json.dump(ecfg, f)
print("full configs written")

VAL_CONDS = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]
jobs = []

# ---- 11.4 aer fine-tune requeue (same job ids; the bug in finetune_aer.py is fixed)
for s in (0, 1, 2):
    jid = "s114_aerft_s%d" % s
    sp = os.path.join(W, "queue", jid + ".status.json")
    st = json.load(open(sp))
    st["state"] = "PENDING"
    st["note"] = "requeued after finetune_aer.py VecNormalize.load fix (X2)"
    json.dump(st, open(sp, "w"), indent=1)

# ---- 11.4 aer evals: zero-shot vs fine-tuned (100 episodes, nominal, engine aer, lane aer)
for s in (0, 1, 2):
    res = r"results\val\aerft__zeroshot_s%d__nom.json" % s
    jobs.append(dict(
        id="s114_eval_zeroshot_s%d" % s, lane="aer", cwd=S,
        cmd='"%s" evaluate.py --method rl:runs\\rl__ramsey_iq__DRcurr\\seed_%d --cfg configs\\evalcfg__rl__ramsey_iq__DRcurr.json '
            '--cond nom --split val --episodes 100 --seed_base 500000 --engine aer --out %s '
            '> "%s\\results\\s114_eval_zeroshot_s%d.log" 2>&1' % (PY, s, res, W, s),
        depends_on=[], expected_outputs=[os.path.join(S, res)],
        env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=120))
    res2 = r"results\val\aerft__finetuned_s%d__nom.json" % s
    jobs.append(dict(
        id="s114_eval_finetuned_s%d" % s, lane="aer", cwd=S,
        cmd='"%s" evaluate.py --method rl:runs\\rl__ramsey_iq__DRcurr_aerft\\seed_%d --cfg configs\\evalcfg__rl__ramsey_iq__DRcurr.json '
            '--cond nom --split val --episodes 100 --seed_base 500000 --engine aer --out %s '
            '> "%s\\results\\s114_eval_finetuned_s%d.log" 2>&1' % (PY, s, res2, W, s),
        depends_on=["s114_aerft_s%d" % s], expected_outputs=[os.path.join(S, res2)],
        env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=120))

# ---- 11.6 combo trainings (gpu lane, NO gpu->gpu deps) + val evals (cpu)
for probe in ("dither", "ramsey_iq"):
    for s in range(5):
        jid = "s116_tr_rl__%s__full_s%d" % (probe, s)
        jobs.append(dict(
            id=jid, lane="gpu", cwd=S,
            cmd='"%s" train_ppo.py --config configs\\rl__%s__full.json --seed %d --out runs\\rl__%s__full\\seed_%d '
                '> "%s\\results\\%s.log" 2>&1' % (PY, probe, s, probe, s, W, jid),
            depends_on=[], expected_outputs=[r"runs\rl__%s__full\seed_%d\best_model.zip" % (probe, s)],
            env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=120))
    for s in range(5):
        for cond in VAL_CONDS:
            res = r"results\val\rlval__rl__%s__full_s%d__%s.json" % (probe, s, cond)
            jobs.append(dict(
                id="s116_eval_rl__%s__full_s%d__%s" % (probe, s, cond),
                lane="cpu%d" % (1 + (s % 4)), cwd=S,
                cmd='"%s" evaluate.py --method rl:runs\\rl__%s__full\\seed_%d --cfg configs\\evalcfg__rl__%s__full.json '
                    '--cond %s --split val --episodes 200 --seed_base 500000 --out %s '
                    '> "%s\\results\\s116_eval_%s_s%d_%s.log" 2>&1' % (PY, probe, s, probe, cond, res, W, probe, s, cond),
                depends_on=["s116_tr_rl__%s__full_s%d" % (probe, s)],
                expected_outputs=[os.path.join(S, res)],
                env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=90))

with open(os.path.join(W, "jobs_json", "s114_116_batch.json"), "w") as f:
    json.dump(jobs, f, indent=1)
print("wrote %d jobs" % len(jobs))
