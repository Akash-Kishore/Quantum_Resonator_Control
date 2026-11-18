"""qstudy complexity — Part-2 S19.2/S19.3: time and space complexity measurements.

MEASUREMENTS MUST RUN ON A QUIET MACHINE: schedule via a job on ONE dedicated cpu lane
while no other job runs on that lane, with OMP_NUM_THREADS=1 (torch single thread).
Smoke mode: --quick (tiny sizes; verifies the wiring, NOT the numbers).

Parts:
  latencies : per-decision latency percentiles (us) over N decisions after W warm-up for
              PID, Hill, Bayes (G in 101/201/401/601/1201), MLP forward, stacked MLP
              (K in 4/8/16), GRU (K in 4/8/16); peak RSS per section; parameter counts.
  envs      : environment throughput (steps/s) for surrogate, classical (lorentz+gauss),
              hybrid (rabi+binom), Aer (B in 1/10/100/1000); peak RSS.
  exponents : log-log least-squares scaling exponents with 95% CI (latency vs G, K, B)
              compared with the S19.1 theory.
  feasibility: ratio of measured decision latency to the physical step time T_step of
              every platform profile at shots 64/1024.
  trainstats : training wall-clock, steps/s, peak RSS per config family from train
              summaries (+ progress logs for RSS).
  all       : all of the above into one JSON.
"""
import argparse, glob, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def peak_rss_mb():
    import psutil
    p = psutil.Process(os.getpid())
    return float(p.memory_info().rss / 2 ** 20)


def percentiles_us(times_s):
    t = np.asarray(times_s, dtype=float) * 1e6
    return dict(p50=float(np.percentile(t, 50)), p95=float(np.percentile(t, 95)),
                p99=float(np.percentile(t, 99)), n=int(t.size))


def time_calls(fn, n, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    pc = time.perf_counter
    for _ in range(n):
        t0 = pc()
        fn()
        ts.append(pc() - t0)
    return percentiles_us(ts)


def latencies(decisions, warmup):
    import qphysics as qp
    from envs import BatchEnv, DEFAULT_CFG
    from controllers import PIDController, HillClimb, BayesTracker
    out = {}

    # Hill on rabi
    cfg = dict(DEFAULT_CFG); cfg.update(probe="rabi", horizon=3, burn_in=0)
    env = BatchEnv(cfg, 1, 490001); env.reset()
    hill = HillClimb(cfg, env)
    out["hill"] = time_calls(lambda: hill.act(None), decisions, warmup)
    out["hill"]["peak_rss_mb"] = peak_rss_mb()

    # PID on ramsey_iq
    cfgp = dict(DEFAULT_CFG); cfgp.update(probe="ramsey_iq", horizon=3, burn_in=0)
    envp = BatchEnv(cfgp, 1, 490002); envp.reset()
    pid = PIDController(cfgp, envp)
    out["pid"] = time_calls(lambda: pid.act(None), decisions, warmup)
    out["pid"]["peak_rss_mb"] = peak_rss_mb()

    # Bayes filter, G sweep (dense transition predict+update with a fixed measurement)
    cfgb = dict(DEFAULT_CFG); cfgb.update(probe="ramsey_iq", horizon=3, burn_in=0)
    envb = BatchEnv(cfgb, 1, 490003); envb.reset()
    flt = BayesTracker(cfgb, assumed="nominal", env_view=envb)
    counts = np.array([[300, 300]], dtype=np.int64)
    shots_arr = np.array([1024], dtype=np.int64)
    for G in (101, 201, 401, 601, 1201):
        flt.gb = qp.GridBayes(half_range=30000.0, n=G, sigma=flt.f_sigma, theta=flt.f_theta)
        flt.post = flt.gb.init(1)
        r = time_calls(lambda: flt.step_filter(counts, shots_arr, np.zeros(1), "ramsey_iq"),
                       decisions, warmup)
        r["G"] = G
        r["peak_rss_mb"] = peak_rss_mb()
        r["transition_matrix_mb"] = float(flt.gb.T.nbytes / 2 ** 20)
        out["bayes_G%d" % G] = r

    # Torch policies (single thread)
    import torch
    torch.set_num_threads(1)
    D = 5  # ramsey_iq observation size
    with torch.no_grad():
        def mlp_forward(model, x):
            model(x)

        mlp = torch.nn.Sequential(torch.nn.Linear(D, 64), torch.nn.Tanh(),
                                  torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                  torch.nn.Linear(64, 1))
        x1 = torch.zeros(1, D)
        out["mlp"] = time_calls(lambda: mlp_forward(mlp, x1), decisions, warmup)
        out["mlp"]["n_params"] = int(sum(p.numel() for p in mlp.parameters()))
        out["mlp"]["peak_rss_mb"] = peak_rss_mb()
        for K in (4, 8, 16):
            smlp = torch.nn.Sequential(torch.nn.Linear(K * D, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, 1))
            xk = torch.zeros(1, K * D)
            r = time_calls(lambda: mlp_forward(smlp, xk), decisions, warmup)
            r["K"] = K
            r["n_params"] = int(sum(p.numel() for p in smlp.parameters()))
            r["peak_rss_mb"] = peak_rss_mb()
            out["stacked_mlp_K%d" % K] = r
        for K in (4, 8, 16):
            gru = torch.nn.GRU(D, 64, batch_first=True)
            xg = torch.zeros(1, K, D)
            r = time_calls(lambda: mlp_forward(gru, xg), decisions, warmup)
            r["K"] = K
            r["n_params"] = int(sum(p.numel() for p in gru.parameters()))
            r["peak_rss_mb"] = peak_rss_mb()
            out["gru_K%d" % K] = r
    return out


def envs(steps_per_b, batches, use_aer=True):
    from envs import BatchEnv, DEFAULT_CFG
    out = {}
    variants = {
        "surrogate": dict(engine="surrogate"),
        "classical": dict(engine="hybrid", lineshape="lorentz", noise="gauss"),
        "hybrid": dict(engine="hybrid", lineshape="rabi", noise="binom"),
    }
    for name, v in variants.items():
        for B in batches:
            cfg = dict(DEFAULT_CFG); cfg.update(probe="rabi", horizon=steps_per_b + 5,
                                                burn_in=0, **v)
            env = BatchEnv(cfg, B, 490100)
            env.reset()
            pc = time.perf_counter
            t0 = pc()
            n = 0
            done = False
            while not done:
                obs, rew, done, info = env.step(np.zeros((B, 1)))
                n += 1
            dt = pc() - t0
            out["%s_B%d" % (name, B)] = dict(B=B, steps=float(n * B), seconds=dt,
                                              steps_per_s=float(n * B / dt),
                                              peak_rss_mb=peak_rss_mb())
    if use_aer:
        from engines import AerEngine
        for B in batches:
            cfg = dict(DEFAULT_CFG); cfg.update(probe="rabi", horizon=steps_per_b + 5,
                                                burn_in=0, engine="aer")
            env = BatchEnv(cfg, B, 490200)
            env.reset()
            pc = time.perf_counter
            t0 = pc()
            n = 0
            done = False
            while not done:
                obs, rew, done, info = env.step(np.zeros((B, 1)))
                n += 1
            dt = pc() - t0
            out["aer_B%d" % B] = dict(B=B, steps=float(n * B), seconds=dt,
                                      steps_per_s=float(n * B / dt),
                                      peak_rss_mb=peak_rss_mb())
    return out


def _fit_exponent(xs, ys):
    """Log-log least squares slope with 95% CI (t approximation, small-n honest)."""
    x = np.log(np.asarray(xs, dtype=float))
    y = np.log(np.asarray(ys, dtype=float))
    n = len(x)
    if n < 3:
        return dict(slope=float(np.nan), ci_lo=float("nan"), ci_hi=float("nan"), n=n)
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        return dict(slope=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), n=n)
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    yhat = ym + slope * (x - xm)
    resid = y - yhat
    s2 = float((resid ** 2).sum() / max(n - 2, 1))
    se = float(math.sqrt(s2 / sxx))
    # t_{0.975, n-2}
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(0.975, max(n - 2, 1)))
    except Exception:
        tcrit = 4.303 if n - 2 >= 1 else 12.706
    return dict(slope=slope, ci_lo=slope - tcrit * se, ci_hi=slope + tcrit * se, n=n)


def exponents(lat, envd):
    out = {}
    gs = [101, 201, 401, 601, 1201]
    out["bayes_vs_G"] = dict(fit=_fit_exponent(gs, [lat["bayes_G%d" % g]["p50"] for g in gs]),
                             theory=2.0)   # dense G x G transition matrix
    ks = [4, 8, 16]
    out["stacked_mlp_vs_K"] = dict(fit=_fit_exponent(ks, [lat["stacked_mlp_K%d" % k]["p50"] for k in ks]),
                                   theory=1.0)
    out["gru_vs_K"] = dict(fit=_fit_exponent(ks, [lat["gru_K%d" % k]["p50"] for k in ks]),
                           theory=1.0)     # K sequential GRU cells, hidden size fixed
    if envd:
        bs = sorted({v["B"] for k, v in envd.items() if k.startswith("surrogate")})
        if len(bs) >= 3:
            per_step_ms = [envd["surrogate_B%d" % b]["seconds"] / envd["surrogate_B%d" % b]["steps"] * 1e3
                           for b in bs]
            out["env_step_vs_B"] = dict(fit=_fit_exponent(bs, per_step_ms), theory=1.0)
    return out


def feasibility(lat):
    from qphysics_ext import PROFILES, profile_to_cfg
    rows = []
    lat_us = {
        "pid": lat["pid"]["p50"], "hill": lat["hill"]["p50"],
        "bayes_G601": lat["bayes_G601"]["p50"], "mlp": lat["mlp"]["p50"],
        "gru_K8": lat["gru_K8"]["p50"],
    }
    for name in PROFILES:
        for shots in (64, 1024):
            _, info = profile_to_cfg(name, shots=shots)
            t_step_s = float(info["T_step_s"])
            row = dict(profile=name, shots=shots, T_step_s=t_step_s)
            for k, v in lat_us.items():
                row["ratio_" + k] = float(v * 1e-6 / t_step_s)
            rows.append(row)
    return rows


def trainstats():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", "seed_*", "train_summary.json"))):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        d = os.path.dirname(p)
        rss = j.get("peak_rss_mb")
        if rss is None:
            rss = None
            prog = os.path.join(d, "progress.log")
            if os.path.exists(prog):
                best = 0.0
                for line in open(prog, errors="replace"):
                    if "RSS_MB=" in line:
                        try:
                            best = max(best, float(line.split("RSS_MB=")[1].split()[0]))
                        except Exception:
                            pass
                rss = best if best > 0 else None
        # model file size
        z = os.path.join(d, "best_model.zip")
        rows.append(dict(run=os.path.relpath(d, HERE), config=j.get("config"),
                         seed=j.get("seed"), steps=j.get("steps"),
                         wall_seconds=j.get("wall_seconds"),
                         steps_per_s=(float(j["steps"]) / j["wall_seconds"]) if j.get("wall_seconds") else None,
                         best_val=j.get("best_val"), peak_rss_mb=rss,
                         model_zip_mb=float(os.path.getsize(z) / 2 ** 20) if os.path.exists(z) else None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["latencies", "envs", "exponents", "feasibility", "trainstats", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--decisions", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--env-steps", type=int, default=2000)
    ap.add_argument("--batches", default="1,10,100,1000")
    ap.add_argument("--lat-file", default=None,
                    help="latencies JSON (required for standalone exponents/feasibility)")
    ap.add_argument("--no-aer", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke sizes; wiring check only")
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("SKIP (exists): %s" % a.out)
        return 0
    if a.quick:
        a.decisions, a.warmup, a.env_steps = 60, 10, 30
    batches = [int(b) for b in a.batches.split(",")]
    res = {}
    if a.part in ("latencies", "all"):
        res["latencies"] = latencies(a.decisions, a.warmup)
        print("PROGRESS latencies done")
    if a.part in ("envs", "all"):
        res["envs"] = envs(a.env_steps, batches, use_aer=not a.no_aer)
        print("PROGRESS envs done")
    if a.part in ("exponents", "all"):
        lat = res.get("latencies")
        if lat is None:
            if not a.lat_file:
                print("REFUSED: standalone exponents needs --lat-file", flush=True)
                return 3
            lat = json.load(open(a.lat_file, encoding="utf-8"))["latencies"]
        envd = res.get("envs")
        res["exponents"] = exponents(lat, envd)
        print("PROGRESS exponents done")
    if a.part in ("feasibility", "all"):
        lat = res.get("latencies")
        if lat is None:
            if not a.lat_file:
                print("REFUSED: standalone feasibility needs --lat-file", flush=True)
                return 3
            lat = json.load(open(a.lat_file, encoding="utf-8"))["latencies"]
        res["feasibility"] = feasibility(lat)
        print("PROGRESS feasibility done")
    if a.part in ("trainstats", "all"):
        res["trainstats"] = trainstats()
        print("PROGRESS trainstats done")
    res["status"] = "ok"
    res["part"] = a.part
    res["quick"] = bool(a.quick)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("DONE complexity part=%s -> %s" % (a.part, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
