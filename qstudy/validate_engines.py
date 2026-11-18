"""qstudy validate_engines — V1 expected-probability, V2 KS distribution, V3 closed-loop, V4 speed.
Results -> results\\validation\\*.json + csv; ingested into db table 'validation'."""
import datetime, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results", "validation")
os.makedirs(OUT, exist_ok=True)

DETUNINGS = list(range(-12000, 12001, 2000))
NOISE_SETS = {
    "none": dict(t2=None, e01=0.0, e10=0.0),
    "t2_200us_e02": dict(t2=200e-6, e01=0.02, e10=0.02),
    "t2_50us_e05_10": dict(t2=50e-6, e01=0.05, e10=0.10),
}
V1_SHOTS = 200000
V2_CELLS = [("rabi", 0.0), ("rabi", 6000.0), ("rabi", -8000.0),
            ("ramsey_I", 0.0), ("ramsey_I", 1500.0), ("ramsey_I", -9000.0),
            ("ramsey_Q", 1000.0), ("ramsey_Q", 5000.0), ("ramsey_Q", -6000.0),
            ("ramsey_I", 4000.0), ("ramsey_Q", -12000.0), ("rabi", 12000.0)]


def log(msg):
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def surrogate_prob(kind, delta, noise):
    import qphysics as qp
    t2 = noise["t2"] if noise["t2"] is not None else np.inf
    if kind == "rabi":
        p = qp.rabi_p(delta, omega=5000.0)
    elif kind == "ramsey_I":
        p = qp.ramsey_p(delta, tau=50e-6, t2=t2, quad="I")
    elif kind == "ramsey_Q":
        p = qp.ramsey_p(delta, tau=50e-6, t2=t2, quad="Q")
    else:
        raise ValueError(kind)
    return float(qp.readout(p, noise["e01"], noise["e10"]))


def aer_probs(kind, deltas, noise, shots, reps=1):
    """Run `reps` copies of each delta through AerEngine in ONE batched run; returns (len(deltas), reps) counts."""
    from engines import AerEngine
    cfg = dict(omega=5000.0, tau=50e-6, t2=noise["t2"] if noise["t2"] is not None else np.inf,
               e01=noise["e01"], e10=noise["e10"])
    eng = AerEngine(cfg)
    B = len(deltas)
    delta_arr = np.tile(np.asarray(deltas, dtype=float), reps)
    rng = np.random.default_rng(777)
    rngs = [np.random.default_rng(777 + i) for i in range(B * reps)]
    counts = eng.counts(kind if kind != "rabi" else "rabi", delta_arr,
                        np.full(B * reps, shots), np.full(B * reps, cfg["t2"]),
                        np.full(B * reps, cfg["e01"]), np.full(B * reps, cfg["e10"]),
                        rngs, omega=5000.0, tau=50e-6)
    return counts.reshape(reps, B).T  # (B, reps)


def v1():
    log("V1 start")
    rows = []
    all_pass = True
    for ns_name, noise in NOISE_SETS.items():
        for kind in ("rabi", "ramsey_I", "ramsey_Q"):
            counts = aer_probs(kind, DETUNINGS, noise, V1_SHOTS)[ :, 0]
            for d, c in zip(DETUNINGS, counts):
                p_sur = surrogate_prob(kind, float(d), noise)
                p_aer = float(c) / V1_SHOTS
                tol = 4 * np.sqrt(p_sur * (1 - p_sur) / V1_SHOTS) + 0.002
                ok = abs(p_aer - p_sur) <= tol
                all_pass = all_pass and ok
                rows.append(dict(test="V1", kind=kind, noise=ns_name, delta=d,
                                 p_aer=p_aer, p_sur=p_sur, tol=tol, pass_=bool(ok)))
            log("V1 %s %s done (%d cells)" % (ns_name, kind, len(DETUNINGS)))
    with open(os.path.join(OUT, "v1_expected_probability.json"), "w") as f:
        json.dump({"pass": bool(all_pass), "n_cells": len(rows),
                   "n_pass": sum(r["pass_"] for r in rows), "rows": rows}, f, indent=1)
    log("V1 %s (%d/%d cells pass)" % ("PASS" if all_pass else "FAIL", sum(r["pass_"] for r in rows), len(rows)))
    return all_pass


def v2():
    from scipy.stats import ks_2samp
    log("V2 start")
    rows = []
    n_pass = 0
    for kind, d in V2_CELLS:
        for shots in (64, 1024):
            reps = 300
            c_aer = aer_probs(kind, [d], NOISE_SETS["t2_200us_e02"], shots, reps=reps)[:, 0]
            rng = np.random.default_rng(1234)
            p = surrogate_prob(kind, float(d), NOISE_SETS["t2_200us_e02"])
            c_sur = rng.binomial(shots, p, reps)
            stat, pval = ks_2samp(c_aer, c_sur)
            ok = pval > 0.01
            n_pass += int(ok)
            rows.append(dict(test="V2", kind=kind, delta=d, shots=shots,
                             ks_stat=float(stat), p_value=float(pval), pass_=bool(ok)))
            log("V2 cell %s d=%.0f shots=%d p=%.4f %s" % (kind, d, shots, pval, "PASS" if ok else "FAIL"))
    frac = n_pass / len(rows)
    overall = frac >= 0.90
    with open(os.path.join(OUT, "v2_ks_distribution.json"), "w") as f:
        json.dump({"pass": bool(overall), "n_cells": len(rows), "n_pass": n_pass, "rows": rows}, f, indent=1)
    log("V2 %s (%d/%d cells p>0.01)" % ("PASS" if overall else "FAIL", n_pass, len(rows)))
    return overall


def v3():
    from envs import BatchEnv
    import controllers as C
    log("V3 start (HillClimb closed loop)")
    res = {}
    for engine_name in ("surrogate", "aer"):
        engine = None
        if engine_name == "aer":
            from engines import AerEngine
            engine = AerEngine(dict(omega=5000.0, tau=50e-6, t2=200e-6, e01=0.02, e10=0.02))
        env = BatchEnv({"probe": "rabi"}, 100, 500000, engine=engine)
        env.reset()
        ctl = C.HillClimb({"probe": "rabi"}, env, step_hz=1000.0, thr=0.02)
        e_tr = np.zeros((env.T, env.B))
        t0 = time.time()
        for t in range(env.T):
            a = ctl.act(None).reshape(-1, 1)
            obs, rew, done, info = env.step(a)
            e_tr[t] = info["e"]
            if time.time() - t0 > 60:
                log("V3 %s t=%d/%d" % (engine_name, t + 1, env.T))
                t0 = time.time()
        m = env.metrics(e_tr)
        res[engine_name] = float(m["lock_w1000"].mean())
        log("V3 %s lock_w1000=%.4f" % (engine_name, res[engine_name]))
    diff = abs(res["surrogate"] - res["aer"])
    overall = diff <= 0.04
    with open(os.path.join(OUT, "v3_closed_loop.json"), "w") as f:
        json.dump({"pass": bool(overall), "lock_w1000": res, "abs_difference": diff,
                   "note": "HillClimb (Bayes tracker comparison re-run in Stage 8/9)"}, f, indent=1)
    log("V3 %s (|diff|=%.4f)" % ("PASS" if overall else "FAIL", diff))
    return overall


def v3b():
    """Stage 9.3: V3 re-run with the Bayes tracker on both engines (|lock diff| <= 0.04)."""
    from envs import BatchEnv
    from controllers import BayesTracker
    log("V3b start (Bayes tracker closed loop)")
    res = {}
    for engine_name in ("surrogate", "aer"):
        engine = None
        if engine_name == "aer":
            from engines import AerEngine
            engine = AerEngine(dict(omega=5000.0, tau=50e-6, t2=200e-6, e01=0.02, e10=0.02))
        cfg = {"probe": "ramsey_iq"}
        env = BatchEnv(cfg, 100, 500000, engine=engine)
        env.reset()
        ctl = BayesTracker(cfg, assumed="nominal", env_view=env)
        e_tr = np.zeros((env.T, env.B))
        t0 = time.time()
        for t in range(env.T):
            a = ctl.act(None).reshape(-1, 1)
            obs, rew, done, info = env.step(a)
            e_tr[t] = info["e"]
            if time.time() - t0 > 60:
                log("V3b %s t=%d/%d" % (engine_name, t + 1, env.T))
                t0 = time.time()
        m = env.metrics(e_tr)
        res[engine_name] = float(m["lock_w1000"].mean())
        log("V3b %s lock_w1000=%.4f" % (engine_name, res[engine_name]))
    diff = abs(res["surrogate"] - res["aer"])
    overall = diff <= 0.04
    with open(os.path.join(OUT, "v3b_bayes_closed_loop.json"), "w") as f:
        json.dump({"pass": bool(overall), "lock_w1000": res, "abs_difference": diff}, f, indent=1)
    log("V3b %s (|diff|=%.4f)" % ("PASS" if overall else "FAIL", diff))
    return overall


def v4():
    from engines import AerEngine
    import qphysics as qp
    log("V4 speed")
    B = 64
    deltas = np.linspace(-2000, 2000, B)
    rngs = [np.random.default_rng(i) for i in range(B)]
    shots = np.full(B, 1024)
    t2 = np.full(B, 200e-6); e01 = np.full(B, 0.02); e10 = np.full(B, 0.02)
    from engines import SurrogateEngine
    sur = SurrogateEngine()
    t0 = time.time(); n = 0
    while time.time() - t0 < 5:
        sur.counts("ramsey_I", deltas, shots, t2, e01, e10, rngs, tau=50e-6)
        n += B
    sur_rate = n / (time.time() - t0)
    aer = AerEngine(dict(omega=5000.0, tau=50e-6, t2=200e-6, e01=0.02, e10=0.02))
    t0 = time.time(); n = 0
    while time.time() - t0 < 20:
        aer.counts("ramsey_I", deltas, shots, t2, e01, e10, rngs, tau=50e-6)
        n += B
        if n % (B * 10) == 0:
            log("V4 aer %d steps" % n)
    aer_rate = n / (time.time() - t0)
    with open(os.path.join(OUT, "v4_speed.json"), "w") as f:
        json.dump({"surrogate_steps_per_s": sur_rate, "aer_steps_per_s": aer_rate}, f, indent=1)
    log("V4 surrogate=%.0f steps/s aer=%.0f steps/s (report only)" % (sur_rate, aer_rate))
    return True


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = {}
    if which in ("all", "v1"):
        ok["V1"] = v1()
    if which in ("all", "v2"):
        ok["V2"] = v2()
    if which in ("all", "v3"):
        ok["V3"] = v3()
    if which in ("all", "v3b"):
        ok["V3b"] = v3b()
    if which in ("all", "v4"):
        ok["V4"] = v4()
    with open(os.path.join(OUT, "validation_summary.json"), "w") as f:
        json.dump({k: bool(v) for k, v in ok.items()}, f, indent=1)
    log("VALIDATION SUMMARY: %s" % ok)
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

