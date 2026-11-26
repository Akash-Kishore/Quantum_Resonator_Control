"""Generate qstudy config files for stages 7-11 experiments."""
import json, os

S = r"C:\Projects\MiniProject_Sem4\qstudy"
CFG = os.path.join(S, "configs")
os.makedirs(CFG, exist_ok=True)

BASE_PPO = {"n_envs": 8, "n_steps": 256, "batch_size": 256, "n_epochs": 10, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_range": 0.2, "learning_rate": 3e-4, "ent_coef": 0.0,
            "vf_coef": 0.5, "max_grad_norm": 0.5}
BASE = {"ppo": BASE_PPO, "total_steps": 2000000, "val_every": 100000}
SMOKE = {"ppo": dict(BASE_PPO, n_envs=4), "total_steps": 20000, "val_every": 10000}


def w(name, spec):
    with open(os.path.join(CFG, name + ".json"), "w") as f:
        json.dump(spec, f, indent=1)
    print("wrote", name)


# Stage 7: ramsey_iq plain
w("rl__ramsey_iq__plain", dict(BASE, env={"probe": "ramsey_iq"}))

# Stage 9: hybrid (filter features)
for probe in ("rabi", "dither", "ramsey_iq"):
    w("rl__%s__filter" % probe, dict(BASE, env={"probe": probe, "filter_features": True}))

# Stage 10: memory variants
for probe in ("rabi", "ramsey_iq"):
    for K in (4, 8, 16):
        w("rl__%s__K%d" % (probe, K), dict(BASE, env={"probe": probe, "stack": K},
                                           memory={"type": "stack", "K": K}))
    w("rl__%s__gru8" % probe, dict(BASE, env={"probe": probe, "stack": 8},
                                   memory={"type": "gru", "K": 8}))
w("rl__rabi__K4__smoke", dict(SMOKE, env={"probe": "rabi", "stack": 4}, memory={"type": "stack", "K": 4}))
w("rl__ramsey_iq__gru8__smoke", dict(SMOKE, env={"probe": "ramsey_iq", "stack": 8}, memory={"type": "gru", "K": 8}))

# Stage 11: DR and DRcurr
for probe in ("rabi", "dither", "ramsey_iq"):
    w("rl__%s__DR" % probe, dict(BASE, env={"probe": probe, "dr": {"scale": 1.0}}))
    w("rl__%s__DRcurr" % probe, dict(BASE, env={"probe": probe, "dr": {"scale": 1.0, "curr": True}}))
w("rl__dither__DRcurr__smoke", dict(SMOKE, env={"probe": "dither", "dr": {"scale": 1.0, "curr": True}}))

# Stage 11.5 tuning grid: probe dither plain, seed 100, 1M steps, lr x ent x gamma
for lr in ("3e-4", "1e-4"):
    for ent in ("0.0", "0.003"):
        for gam in ("0.99", "0.995"):
            name = "tune__lr%s__ent%s__gam%s" % (lr.replace("-", ""), ent.replace(".", ""), gam.replace(".", ""))
            w(name, dict(BASE, env={"probe": "dither"}, total_steps=1000000,
                         ppo=dict(BASE_PPO, learning_rate=float(lr), ent_coef=float(ent), gamma=float(gam))))
print("done")
