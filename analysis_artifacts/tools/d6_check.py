"""D6 (P2) check: freeze_obs_norm keeps obs_rms identical; lr equals configured."""
import json, os, pickle, sys

import numpy as np

S = r"C:\Projects\MiniProject_Sem4\qstudy"
SRC = os.path.join(S, "runs", "rl__dither__plain", "seed_0")
FT = os.path.join(S, "runs_smoke", "d6_ft_freeze")
CFG = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\smoke_cfgs\d6_ft_freeze.json"


def obs_rms(path):
    with open(path, "rb") as f:
        vn = pickle.load(f)
    return np.asarray(vn.obs_rms.mean), np.asarray(vn.obs_rms.var), vn.training


m1, v1, tr1 = obs_rms(os.path.join(SRC, "vec_normalize.pkl"))
m2, v2, tr2 = obs_rms(os.path.join(FT, "vec_normalize.pkl"))
print("obs_rms mean equal:", bool(np.array_equal(m1, m2)))
print("obs_rms var equal: ", bool(np.array_equal(v1, v2)))
print("freeze left venv.training =", tr2, "(VecNormalize.training flag, expect False)")

cfg = json.load(open(CFG, encoding="utf-8"))
want_lr = float(cfg["ppo"]["learning_rate"])
from stable_baselines3 import PPO
model = PPO.load(os.path.join(FT, "best_model.zip"), device="cpu")
got_lr = float(model.learning_rate) if not callable(model.learning_rate) else float(model.lr_schedule(1))
print("lr configured:", want_lr, "lr in saved model:", got_lr, "match:", abs(got_lr - want_lr) < 1e-12)
