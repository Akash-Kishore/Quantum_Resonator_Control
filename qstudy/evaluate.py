"""qstudy evaluate — CLI evaluation of any method under any pre-registered condition.
Part-2 O5 extensions: CVaR95 of per-episode rms and of per-episode (1 - lock_w1000), the 95th
percentile of per-episode rms, lock-loss probability, per-episode first-lock time, and
--store_failures K (full traces of up to K lock-lost episodes plus K matched non-lost episodes).
Existing result files are NOT rewritten; new files carry the new fields."""
import argparse, datetime, hashlib, json, os, pickle, re, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import stress
from envs import BatchEnv, DEFAULT_CFG, mean_reward

METRIC_KEYS = ["lock_w500", "lock_w1000", "lock_w2000", "lock_w3333", "lock_w5000",
               "rms", "med_abs", "p95_abs", "max_abs", "lock_lost"]


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def cvar95(x):
    """Mean of the worst 5% of episodes (empty -> nan)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(0.05 * x.size)))
    return float(np.sort(x)[-k:].mean())


def first_lock_time(e, W, consec=3):
    """First step with `consec` consecutive |e|<=W; -1 if never. e: (T,) trace."""
    ok = np.abs(np.asarray(e, dtype=float)) <= W
    run = 0
    for t, o in enumerate(ok):
        run = run + 1 if o else 0
        if run >= consec:
            return t - consec + 1
    return -1


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "-C", os.path.dirname(HERE), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "UNKNOWN"


def versions():
    import importlib
    out = {"python": sys.version.split()[0]}
    for m in ("numpy", "scipy", "torch", "stable_baselines3", "gymnasium", "qiskit", "qiskit_aer", "matplotlib"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            out[m] = "absent"
    return out


def load_rl(model_dir):
    from stable_baselines3 import PPO
    model = PPO.load(os.path.join(model_dir, "best_model.zip"), device="cpu")
    with open(os.path.join(model_dir, "vec_normalize.pkl"), "rb") as f:
        vn = pickle.load(f)
    obs_rms = vn.obs_rms
    clip = getattr(vn, "clip_obs", 10.0)
    eps = getattr(vn, "epsilon", 1e-8)

    def norm(obs):
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + eps), -clip, clip)

    return model, norm


def make_controller(method, cfg, env):
    import controllers as C
    kind = method.split(":", 1)
    if method == "ctrl:noctrl":
        return C.NoControl(cfg, env)
    if method == "ctrl:oracle":
        return C.Oracle(cfg, env)
    if method == "ctrl:hill":
        return C.HillClimb(cfg, env)
    if method == "ctrl:pid":
        return C.PIDController(cfg, env)
    if method.startswith("ctrl:bayes"):
        assumed = method.split(":")[-1] if ":" in method[10:] else "nominal"
        return C.BayesTracker(cfg, assumed=assumed if assumed in ("nominal", "oracle") else "nominal", env_view=env)
    raise SystemExit("unknown controller method: %s" % method)


# ---- M2 F4: configuration-consistency pre-flight/post-flight ------------------
def _cfg_val_eq(a, b):
    try:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
            return bool(np.isclose(float(a), float(b), rtol=0, atol=0, equal_nan=True)) or float(a) == float(b)
    except Exception:
        pass
    return a == b


def check_cfg_consistent(cfg, cond_id, model_dir=None):
    """F4: the effective cfg must be consistent with its cond_id (every override of the
    condition must be present in the effective cfg), and for RL models the recorded
    probe/stack/filter type must equal the model's training config. Returns a list of
    problems (empty = consistent). An inconsistent cfg aborts the job with exit code 4."""
    bad = []
    try:
        if cond_id.strip().startswith("{"):
            want = json.loads(cond_id)
        else:
            want = stress.cond_cfg(cond_id)
    except Exception as ex:
        return ["cond %r not parseable: %s" % (cond_id, ex)]
    for k, v in want.items():
        if k not in cfg:
            bad.append("cfg missing %s (cond wants %r)" % (k, v))
        elif not _cfg_val_eq(cfg[k], v):
            bad.append("cfg[%s]=%r != cond %r" % (k, cfg[k], v))
    if model_dir is not None:
        base = os.path.basename(os.path.normpath(model_dir))
        m = re.match(r"^(rl__.+?)_s\d+$", base)
        name = m.group(1) if m else base
        tpath = os.path.join(HERE, "configs", name + ".json")
        if os.path.exists(tpath):
            tcfg = json.load(open(tpath, encoding="utf-8"))
            tenv = tcfg.get("env", tcfg)
            for k in ("probe", "stack", "filter_features", "dr"):
                if k in tenv and k in cfg and not _cfg_val_eq(cfg[k], tenv[k]):
                    bad.append("cfg[%s]=%r != training config %r" % (k, cfg[k], tenv[k]))
    return bad


def abort_cfg_inconsistent(problems, tag):
    """F4 abort: exit code 4, logged (re-run of a sealed evaluation is allowed ONLY
    after such an infrastructure abort, never because of its results)."""
    line = "%s CFG-AUDIT ABORT (%s): %s" % (now(), tag, "; ".join(problems))
    print(line, flush=True)
    try:
        os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
        with open(os.path.join(HERE, "logs", "cfg_audit_aborts.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    sys.exit(4)


def run_batch(cfg, B, seed_base, method, model_dir, traces, progress_every=60.0, store_failures=0):
    env = BatchEnv(cfg, B, seed_base)
    obs = env.reset()
    T = env.T
    e_tr = np.zeros((T, B))
    fc_tr = np.zeros((T, B))
    x_tr = np.zeros((T, B))
    r_tr = np.zeros((T, B))
    a_tr = np.zeros((T, B))
    c_tr = np.zeros((T, B))       # counts (first probe's count; -1 when a probe emits several)
    shots_tr = np.zeros((T, B))
    mode_tr = np.zeros((T, B))    # probe mode index (rabi=0, dither=1, ramsey_iq=2); -1 unknown
    sig_tr = np.full((T, B), np.nan)  # filter posterior sigma, when available
    rl = None
    ctrl = None
    if method.startswith("rl:"):
        rl, norm = load_rl(model_dir)
        obs = norm(obs)
    else:
        ctrl = make_controller(method, cfg, env)
    t0 = time.time()
    for t in range(T):
        if rl is not None:
            a, _ = rl.predict(obs, deterministic=True)
            a = np.asarray(a, dtype=float).reshape(-1, 1)
        else:
            a = ctrl.act(None).reshape(-1, 1)
        obs, rew, done, info = env.step(a)
        e_tr[t] = info["e"]
        fc_tr[t] = info["fc"]
        x_tr[t] = info["x"]
        r_tr[t] = rew
        a_tr[t] = a.reshape(-1)
        counts = info.get("counts")
        c_tr[t] = -1.0 if getattr(counts, "ndim", 1) > 1 else (float(np.asarray(counts).reshape(-1)[0])
                                                               if counts is not None else -1.0)
        shots_tr[t] = np.asarray(info.get("shots_used", np.zeros(B)), dtype=float).reshape(-1)[0] \
            if info.get("shots_used") is not None else 0.0
        mode_tr[t] = {"rabi": 0, "rabi_legacy": 0, "dither": 1, "ramsey_iq": 2}.get(str(cfg.get("probe")), -1)
        if hasattr(env, "_flt") and env._flt is not None and getattr(env._flt, "last_sigma", None) is not None:
            sig_tr[t] = env._flt.last_sigma
        if rl is not None:
            obs = norm(obs)
        if time.time() - t0 > progress_every:
            print("PROGRESS t=%d/%d elapsed=%.0fs" % (t + 1, T, time.time() - t0), flush=True)
            t0 = time.time()
    m = env.metrics(e_tr)
    m["mean_reward"] = mean_reward(r_tr, int(cfg.get("burn_in", 20)))
    # M2 F1: record the parameter dict actually used by a ctrl:bayes:* filter
    if ctrl is not None and hasattr(ctrl, "assumed_params"):
        try:
            m["_assumed_params"] = ctrl.assumed_params()
        except Exception as ex:
            m["_assumed_params"] = {"error": str(ex)}
    # ---- Part-2 O5 per-episode risk metrics ----
    W = 1000.0
    m["cvar95_rms"] = cvar95(m["rms"])
    m["cvar95_lockmiss"] = cvar95(1.0 - m["lock_w1000"])
    m["p95_rms"] = float(np.percentile(np.asarray(m["rms"], dtype=float), 95))
    m["lock_loss_prob"] = float(np.asarray(m["lock_lost"]).mean())
    flt = np.array([first_lock_time(e_tr[:, b], W) for b in range(B)], dtype=float)
    m["first_lock_time"] = flt
    if store_failures > 0:
        lost_idx = [b for b in range(B) if m["lock_lost"][b]]
        keep_lost = lost_idx[:store_failures]
        ok_idx = [b for b in range(B) if not m["lock_lost"][b]]
        rng = np.random.default_rng(0)
        keep_ok = list(rng.choice(ok_idx, size=min(store_failures, len(ok_idx)), replace=False)) if ok_idx else []
        m["_failure_episodes"] = {"lost": keep_lost, "matched_ok": [int(i) for i in keep_ok]}
        m["_failure_traces"] = {b: dict(e=e_tr[:, b], fc=fc_tr[:, b], x=x_tr[:, b], actions=a_tr[:, b],
                                        counts=c_tr[:, b], shots=shots_tr[:, b], mode=mode_tr[:, b],
                                        posterior_sigma=sig_tr[:, b])
                                for b in keep_lost + [int(i) for i in keep_ok]}
    return m, e_tr, fc_tr, x_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--cfg", default=None, help="json file (or inline json) of cfg overrides")
    ap.add_argument("--cond", default="nom", help="cond_id or inline json")
    ap.add_argument("--split", required=True, choices=["tune", "val", "sealed"])
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed_base", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", type=int, default=0)
    ap.add_argument("--engine", default="surrogate", choices=["surrogate", "aer"])
    ap.add_argument("--store_failures", type=int, default=20,
                    help="Part-2 O5: store full traces of up to K lock-lost episodes plus K matched non-lost")
    a = ap.parse_args()

    started = now()
    if os.path.exists(a.out):
        if a.split == "sealed":
            try:
                with open(a.out, encoding="utf-8") as f:
                    json.load(f)
                print("SKIP sealed exists: %s" % a.out, flush=True)
                return 0
            except Exception as ex:
                print("SEALED REFUSED: existing sealed file is not valid JSON (%s); "
                      "not overwriting: %s" % (ex, a.out), flush=True)
                return 4
        print("SKIP (exists): %s" % a.out)
        return 0

    # sealed gate (Part 1 base 900000 -> PREREG; Part-2 bases 920000/950000/960000 -> PREREG2)
    if a.split == "sealed":
        base_prereg = {900000: ("PREREG.md", "PREREG.sha256"),
                       920000: ("PREREG2.md", "PREREG2.sha256"),
                       950000: ("PREREG2.md", "PREREG2.sha256"),
                       960000: ("PREREG2.md", "PREREG2.sha256")}.get(a.seed_base)
        if base_prereg is None:
            print("SEALED REFUSED: seed_base %d is not a registered sealed base" % a.seed_base, flush=True)
            return 3
        pre = os.path.join(HERE, base_prereg[0])
        sha = os.path.join(HERE, base_prereg[1])
        if not os.path.exists(sha):
            print("SEALED REFUSED: %s does not exist" % base_prereg[1], flush=True)
            return 3
        want = open(sha, encoding="utf-8").read().strip().split()[0]
        got = sha256_file(pre)
        if want != got:
            print("SEALED REFUSED: %s mismatch (want %s got %s)" % (base_prereg[1], want, got), flush=True)
            return 3

    # seed-base discipline
    bases = {"tune": 100000, "val": 500000, "sealed": 900000}
    sealed_ok = (900000, 920000, 950000, 960000)
    if a.split == "sealed":
        if a.seed_base not in sealed_ok:
            print("REFUSED: seed_base %d not valid for split sealed" % a.seed_base, flush=True)
            return 3
    elif a.seed_base != bases[a.split]:
        if not (a.seed_base >= bases[a.split]):
            print("REFUSED: seed_base %d not valid for split %s" % (a.seed_base, a.split), flush=True)
            return 3

    cfg = dict(DEFAULT_CFG)
    if a.cfg:
        cfg.update(json.loads(a.cfg) if a.cfg.strip().startswith("{") else json.load(open(a.cfg, encoding="utf-8")))
    if a.cond.strip().startswith("{"):
        cfg.update(json.loads(a.cond))
    else:
        cfg.update(stress.cond_cfg(a.cond))
    engine = None
    if a.engine == "aer":
        from engines import AerEngine
        engine = AerEngine(cfg)

    method = a.method
    model_dir = None
    if method.startswith("rl:"):
        model_dir = method[3:]
        if not os.path.isdir(model_dir):
            print("REFUSED: model_dir not found: %s" % model_dir, flush=True)
            return 3

    # M2 F4 pre-flight: an inconsistent effective cfg aborts the job with exit code 4
    # (infrastructure fault; a sealed evaluation may be re-run ONLY after such an abort)
    pre_bad = check_cfg_consistent(cfg, a.cond, model_dir)
    if pre_bad:
        abort_cfg_inconsistent(pre_bad, "pre-flight")

    m, e_tr, fc_tr, x_tr = run_batch(cfg, a.episodes, a.seed_base, method, model_dir, a.traces,
                                      store_failures=a.store_failures)
    assumed_params = m.pop("_assumed_params", None)

    # M2 F4 post-flight: re-verify the cfg dict that will be recorded in the result JSON
    post_bad = check_cfg_consistent(cfg, a.cond, model_dir)
    if post_bad:
        abort_cfg_inconsistent(post_bad, "post-flight")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    npz_path = a.out[:-5] + ".npz" if a.out.endswith(".json") else a.out + ".npz"
    arrays = {k: np.asarray(m[k]) for k in m if not k.startswith("_")}
    fail_eps = m.pop("_failure_episodes", None)
    fail_traces = m.pop("_failure_traces", None)
    if a.traces > 0:
        n = min(a.traces, e_tr.shape[1])
        arrays["trace_e"] = e_tr[:, :n]
        arrays["trace_fc"] = fc_tr[:, :n]
        arrays["trace_x"] = x_tr[:, :n]
    if fail_eps is not None:
        for b, tr in fail_traces.items():
            for key in ("e", "fc", "x", "actions", "counts", "shots", "mode", "posterior_sigma"):
                arrays["fail_ep%d_%s" % (b, key)] = tr[key]
        arrays["fail_lost_episodes"] = np.asarray(fail_eps["lost"], dtype=int)
        arrays["fail_matched_ok_episodes"] = np.asarray(fail_eps["matched_ok"], dtype=int)
    np.savez_compressed(npz_path, **arrays)

    out = {
        "status": "ok",
        "method": method,
        "model_dir": model_dir,
        "cfg": {k: (None if v is None or (isinstance(v, float) and np.isinf(v)) else v) for k, v in cfg.items()
                if isinstance(v, (int, float, str, bool, type(None)))},
        "cond_id": a.cond,
        "split": a.split,
        "episodes": a.episodes,
        "seed_base": a.seed_base,
        "git_head": git_head(),
        "versions": versions(),
        "started": started,
        "finished": now(),
        "cmd": " ".join(sys.argv),
        "npz_path": npz_path,
        "npz_sha256": sha256_file(npz_path),
    }
    if assumed_params is not None:
        out["assumed_params"] = assumed_params   # M2 F1: ctrl:bayes:* filter assumptions
    for k in METRIC_KEYS:
        arr = np.asarray(m[k], dtype=float)
        out[k + "_mean"] = float(arr.mean())
        out[k + "_sd"] = float(arr.std())
    out["lock_lost_rate"] = float(np.asarray(m["lock_lost"]).mean())
    out["mean_reward_mean"] = float(np.asarray(m["mean_reward"]).mean())
    out["mean_reward_sd"] = float(np.asarray(m["mean_reward"]).std())
    # ---- Part-2 O5 fields (new files only; the database columns are nullable) ----
    out["cvar95_rms"] = float(m["cvar95_rms"])
    out["cvar95_lockmiss"] = float(m["cvar95_lockmiss"])
    out["p95_rms"] = float(m["p95_rms"])
    out["lock_loss_prob"] = float(m["lock_loss_prob"])
    flt = np.asarray(m["first_lock_time"], dtype=float)
    locked = flt[flt >= 0]
    out["first_lock_time_mean"] = float(locked.mean()) if locked.size else None
    out["first_lock_time_never_frac"] = float((flt < 0).mean())
    if fail_eps is not None:
        out["fail_lost_episodes"] = fail_eps["lost"]
        out["fail_matched_ok_episodes"] = fail_eps["matched_ok"]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("DONE %s %s episodes=%d lock_w1000=%.3f" % (method, a.cond, a.episodes, out["lock_w1000_mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
