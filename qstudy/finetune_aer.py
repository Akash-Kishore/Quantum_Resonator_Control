"""qstudy finetune_aer — Stage 11.4: fine-tune a surrogate-trained policy on the Aer engine.
CLI: python finetune_aer.py --config <base config json> --src runs\\<src>\\seed_<k> --out runs\\<out>\\seed_<k> --steps 300000"""
import argparse, datetime, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg, f=None):
    line = "[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if f:
        f.write(line + "\n")
        f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--n_envs", type=int, default=4)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "progress.log"), "a", encoding="utf-8")
    spec = json.load(open(a.config, encoding="utf-8"))
    env_cfg = dict(spec.get("env", {}))
    env_cfg["engine"] = "aer"  # nominal noise; AerEngine built from the same cfg
    ppo_cfg = dict(spec.get("ppo", {}))

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from envs import QuantumCalibGym

    def make(i):
        def _init():
            torch.set_num_threads(1)
            c = dict(env_cfg)
            c["env_seed"] = 90000 + i
            return QuantumCalibGym(c)
        return _init

    venv = SubprocVecEnv([make(i) for i in range(a.n_envs)])
    venv = VecNormalize.load(os.path.join(a.src, "vec_normalize.pkl"), venv)  # warm-start statistics
    venv.training = False  # locked statistics during the fine-tune (matches the val-time behaviour)
    model = PPO.load(os.path.join(a.src, "best_model.zip"), env=venv, device="cpu")
    model.learning_rate = 5e-5

    t0 = time.time()
    n0 = 0
    log("AERFT start src=%s out=%s steps=%d" % (a.src, a.out, a.steps), logf)
    model.learn(total_timesteps=a.steps, reset_num_timesteps=False)
    model.save(os.path.join(a.out, "best_model.zip"))
    venv.save(os.path.join(a.out, "vec_normalize.pkl"))
    wall = time.time() - t0
    summary = dict(status="ok", src=a.src, out=a.out, steps=a.steps, wall_seconds=wall,
                   engine="aer", finished=now(), cmd=" ".join(sys.argv))
    with open(os.path.join(a.out, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("AERFT DONE wall=%.0fs" % wall, logf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
