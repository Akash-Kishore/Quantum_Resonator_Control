"""qstudy tune_pid — Stage 6.2: exhaustive PID gain grid on TUNE seeds (base 100000, 200 episodes).
Winner per (probe, drift kind) saved to configs\\pid_gains.json with its tune score."""
import itertools, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KP = [0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
KI = [0.0, 0.02, 0.05]
KD = [0.0, 0.1]
EPISODES = 200
TUNE = 100000


def score(probe, drift, kp, ki, kd):
    from controllers import PIDController
    from envs import BatchEnv
    cfg = {"probe": probe, "drift": drift}
    env = BatchEnv(cfg, EPISODES, TUNE)
    env.reset()
    ctl = PIDController(cfg, env, gains=dict(kp=kp, ki=ki, kd=kd))
    e_tr = np.zeros((env.T, env.B))
    for t in range(env.T):
        a = ctl.act(None).reshape(-1, 1)
        obs, rew, done, info = env.step(a)
        e_tr[t] = info["e"]
    m = env.metrics(e_tr)
    return float(m["lock_w1000"].mean())


def main():
    out = {}
    for probe, drift in itertools.product(("dither", "ramsey_iq"), ("gaussian", "ou")):
        best, best_g = -1, None
        for kp, ki, kd in itertools.product(KP, KI, KD):
            s = score(probe, drift, kp, ki, kd)
            if s > best:
                best, best_g = s, (kp, ki, kd)
            print("PID %s %s kp=%.2f ki=%.3f kd=%.2f -> %.4f (best %.4f %s)" %
                  (probe, drift, kp, ki, kd, s, best, best_g), flush=True)
        out.setdefault(probe, {})[drift] = dict(kp=best_g[0], ki=best_g[1], kd=best_g[2], tune_score=best)
        print("WINNER %s %s: kp=%.2f ki=%.3f kd=%.2f score=%.4f" % (probe, drift, *best_g, best), flush=True)
    path = os.path.join(HERE, "configs", "pid_gains.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("SAVED %s" % path)


if __name__ == "__main__":
    main()
