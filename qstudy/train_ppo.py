"""qstudy train_ppo — training recipe used by all stages (Stage 6.6, amended by Part-2 O6).
CLI: python train_ppo.py --config configs/<name>.json --seed <k> --out runs/<name>/seed_<k>
Config keys beyond Stage 6.6 (Part-2 O6): "algo": "ppo"|"sac"|"td3"; "snapshots": [steps] ->
snapshot_<steps>.zip + vecnorm_<steps>.pkl at those total steps; "init_from": <model dir> ->
continue training (fine-tune) from a saved model and its VecNormalize statistics.
PPO configs without these keys behave exactly as before (O6 backward-compatibility requirement)."""
import argparse, datetime, hashlib, json, os, sys, time

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


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def rss_mb():
    """Resident memory of THIS process in MB (psutil if present, else ctypes GetProcessMemoryInfo)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import ctypes
        import ctypes.wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def normalize_obs(venv, obs):
    """O6c: VecNormalize running stats for PPO; fixed scale-free clip to [-1,1] for SAC/TD3
    (the observation is already in about [-1,1] by construction, so no running statistics)."""
    if hasattr(venv, "obs_rms") and getattr(venv, "obs_rms", None) is not None:
        return np.clip((obs - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var + 1e-8), -10, 10)
    return np.clip(obs, -1.0, 1.0)


class ValCallback:
    """Every val_every steps: evaluate the CURRENT policy on the val condition set
    (seed base 500000, 4 cells x 100 episodes) with BatchEnv; score = mean lock_w1000.
    Keeps best_model.zip + vec_normalize.pkl at the best score."""

    def __init__(self, model, venv, cfg, out_dir, val_every, n_episodes=100, seed_base=500000, logf=None):
        self.model = model
        self.venv = venv
        self.cfg = cfg
        self.out_dir = out_dir
        self.val_every = val_every
        self.n_eps = n_episodes
        self.seed_base = seed_base
        self.best = -1e9
        self.history = []
        self.next_at = val_every
        self.logf = logf
        self.calls = 0

    def _val_score(self):
        import stress
        from envs import BatchEnv
        cells = stress.list_conds("val")
        scores = []
        for cond in cells:
            c = dict(self.cfg)
            c.update(stress.cond_cfg(cond))
            env = BatchEnv(c, self.n_eps, self.seed_base)
            obs = env.reset()
            e_tr = np.zeros((env.T, env.B))
            for t in range(env.T):
                nobs = normalize_obs(self.venv, obs)
                a, _ = self.model.predict(nobs, deterministic=True)
                obs, rew, done, info = env.step(a)
                e_tr[t] = info["e"]
            m = env.metrics(e_tr)
            scores.append(float(m["lock_w1000"].mean()))
        return float(np.mean(scores)), scores

    def __call__(self, locals_, globals_):
        self.model = locals_["self"]
        n = self.model.num_timesteps
        if n < self.next_at:
            return True
        self.next_at += self.val_every
        self.calls += 1
        t0 = time.time()
        score, per_cell = self._val_score()
        self.history.append(dict(step=n, score=score, per_cell=per_cell, wall_s=time.time() - t0))
        if score > self.best:
            self.best = score
            self.model.save(os.path.join(self.out_dir, "best_model.zip"))
            if hasattr(self.venv, "save"):
                self.venv.save(os.path.join(self.out_dir, "vec_normalize.pkl"))
        log("VAL steps=%d score=%.4f best=%.4f cells=%s (%.1fs)" % (n, score, self.best,
            ["%.3f" % s for s in per_cell], time.time() - t0), self.logf)
        return True


class ProgressCallback:
    def __init__(self, logf, every_s=30.0):
        self.t0 = time.time()
        self.every = every_s
        self.next = time.time() + every_s
        self.logf = logf
        self.start_step = 0

    def __call__(self, locals_, globals_):
        model = locals_.get("self")
        n = model.num_timesteps if model else 0
        t = time.time()
        if t >= self.next:
            fps = (n - self.start_step) / max(1e-9, t - self.t0)
            log("PROGRESS steps=%d fps=%.0f RSS_MB=%.0f" % (n, fps, rss_mb()), self.logf)
            self.next = t + self.every
            self.start_step = n
            self.t0 = t
        return True


class SnapshotCallback:
    """Part-2 O6b: at each step threshold in cfg "snapshots" (TOTAL steps of THIS run),
    save the CURRENT policy (snapshot_<steps>.zip) and the VecNormalize statistics
    (vecnorm_<steps>.pkl; only when a VecNormalize wrapper is active)."""

    def __init__(self, model, venv, out_dir, steps_list, logf=None):
        self.model = model
        self.venv = venv
        self.out = out_dir
        self.pending = sorted(int(s) for s in steps_list)
        self.logf = logf

    def __call__(self, locals_, globals_):
        m = locals_.get("self")
        n = m.num_timesteps if m else 0
        while self.pending and n >= self.pending[0]:
            s = self.pending.pop(0)
            self.model.save(os.path.join(self.out, "snapshot_%d.zip" % s))
            if hasattr(self.venv, "save"):
                self.venv.save(os.path.join(self.out, "vecnorm_%d.pkl" % s))
            log("SNAPSHOT steps=%d files=snapshot_%d.zip RSS_MB=%.0f" % (s, s, rss_mb()), self.logf)
        return True


class CurriculumCallback:
    """Stage 11.1: every 10000 steps env_method("set_curriculum", s) with
    s = min(1, steps/(0.5*total_steps)) — only when the config enables DR curriculum."""

    def __init__(self, venv, total_steps, every=10000, logf=None):
        self.venv = venv
        self.total = total_steps
        self.every = every
        self.next_at = every
        self.logf = logf

    def __call__(self, locals_, globals_):
        model = locals_.get("self")
        n = model.num_timesteps if model else 0
        if n >= self.next_at:
            s = min(1.0, n / (0.5 * self.total))
            self.venv.env_method("set_curriculum", s)
            log("CURRICULUM steps=%d s=%.3f" % (n, s), self.logf)
            self.next_at += self.every
        return True


def make_envs(cfg, n_envs, seed, memory=None):
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from envs import QuantumCalibGym

    def make(i):
        def _init():
            torch.set_num_threads(1)
            c = dict(cfg)
            c["env_seed"] = seed * 10000 + i
            return QuantumCalibGym(c)
        return _init

    return SubprocVecEnv([make(i) for i in range(n_envs)])


def make_fixed_clip(venv):
    """Part-2 O6c: scale-free observation handling for SAC/TD3 — no VecNormalize;
    the observation is already in about [-1,1], clip to [-1,1]."""
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class FixedObsClip(VecEnvWrapper):
        def reset(self):
            return np.clip(self.venv.reset(), -1.0, 1.0)

        def step_wait(self):
            obs, rews, dones, infos = self.venv.step_wait()
            return np.clip(obs, -1.0, 1.0), rews, dones, infos

    return FixedObsClip(venv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = json.load(open(a.config, encoding="utf-8"))
    env_cfg = dict(spec.get("env", {}))
    ppo_cfg = dict(spec.get("ppo", {}))
    total_steps = int(spec.get("total_steps", 2_000_000))
    val_every = int(spec.get("val_every", 100_000))
    memory = spec.get("memory", None)
    algo = str(spec.get("algo", "ppo")).lower()
    snapshots = [int(s) for s in spec.get("snapshots", [])]
    init_from = spec.get("init_from", None)
    if init_from and "{seed}" in str(init_from):
        # M5: per-seed fine-tune sources - "{seed}" resolves to this run's --seed so one
        # config file serves all seeds (init_from = "runs\\s15__envLG\\seed_{seed}").
        init_from = str(init_from).replace("{seed}", str(a.seed))
    if algo not in ("ppo", "sac", "td3"):
        raise SystemExit("unknown algo %r (ppo|sac|td3)" % algo)

    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "progress.log"), "a", encoding="utf-8")
    log("TRAIN start config=%s seed=%d out=%s cfg_hash=%s algo=%s init_from=%s" %
        (a.config, a.seed, a.out, cfg_hash(spec), algo, init_from), logf)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    n_envs = int(ppo_cfg.get("n_envs", 8))
    raw_venv = make_envs(env_cfg, n_envs, a.seed, memory)
    device = "cpu"

    if algo == "ppo":
        if init_from:
            # Part-2 O6d: fine-tune from a saved model and its VecNormalize statistics
            # (continue from the loaded running stats; training=True).
            venv = VecNormalize.load(os.path.join(init_from, "vec_normalize.pkl"), raw_venv)
            venv.training = True
            # PREREG2 amendment B (FT-B): frozen observation-normalisation statistics
            # (no running-stat updates) during fine-tuning.
            if bool(spec.get("freeze_obs_norm", False)):
                venv.training = False
        else:
            venv = VecNormalize(raw_venv, norm_obs=True, norm_reward=True, clip_obs=10.0,
                                gamma=float(ppo_cfg.get("gamma", 0.99)))
    else:
        venv = make_fixed_clip(raw_venv)  # SAC/TD3: no VecNormalize (O6c)

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    if memory and memory.get("type") == "gru":
        from policies import WindowGRUExtractor
        K = int(memory.get("K", 8))
        d = venv.observation_space.shape[0] // K
        policy_kwargs = dict(features_extractor_class=WindowGRUExtractor,
                              features_extractor_kwargs=dict(K=K, D=d),
                              net_arch=dict(vf=[128, 128]))
        if torch.cuda.is_available():
            device = "cuda"

    if algo == "ppo":
        if init_from:
            model = PPO.load(os.path.join(init_from, "best_model.zip"), env=venv, device=device,
                             # D6 (P2): fine-tune uses the CONFIG lr (FT-A frozen lr, FT-B frozen/10),
                             # not the source model's saved lr.
                             custom_objects={"learning_rate": float(ppo_cfg.get("learning_rate", 3e-4))})
            if hasattr(model, "set_random_seed"):
                model.set_random_seed(int(a.seed))
        else:
            model = PPO(
                "MlpPolicy", venv, device=device, verbose=0,
                learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
                n_steps=int(ppo_cfg.get("n_steps", 256)),
                batch_size=int(ppo_cfg.get("batch_size", 256)),
                n_epochs=int(ppo_cfg.get("n_epochs", 10)),
                gamma=float(ppo_cfg.get("gamma", 0.99)),
                gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
                clip_range=float(ppo_cfg.get("clip_range", 0.2)),
                ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
                vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
                max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
                policy_kwargs=policy_kwargs,
                seed=int(a.seed),
            )
    elif algo == "sac":
        from stable_baselines3 import SAC
        if init_from:
            model = SAC.load(os.path.join(init_from, "best_model.zip"), env=venv, device=device)
            if hasattr(model, "set_random_seed"):
                model.set_random_seed(int(a.seed))
        else:
            # O6c: SB3 defaults except net_arch [128,128], learning_starts 5000,
            # batch_size 256, gradient_steps 1.
            model = SAC("MlpPolicy", venv, device=device, verbose=0, seed=int(a.seed),
                        learning_starts=5000, batch_size=256, gradient_steps=1,
                        policy_kwargs=dict(net_arch=[128, 128]))
    else:
        from stable_baselines3 import TD3
        if init_from:
            model = TD3.load(os.path.join(init_from, "best_model.zip"), env=venv, device=device)
            if hasattr(model, "set_random_seed"):
                model.set_random_seed(int(a.seed))
        else:
            # train_freq=1 (step): TD3's SB3 default (1, "episode") requires a single env
            # and asserts with SubprocVecEnv n_envs>1 (documented deviation from SB3 defaults).
            model = TD3("MlpPolicy", venv, device=device, verbose=0, seed=int(a.seed),
                        learning_starts=5000, batch_size=256, gradient_steps=1, train_freq=1,
                        policy_kwargs=dict(net_arch=[128, 128]))

    val_cb = ValCallback(model, venv, env_cfg, a.out, val_every, logf=logf)
    prog_cb = ProgressCallback(logf)
    curr_cb = CurriculumCallback(venv, total_steps, logf=logf) if (spec.get("env", {}).get("dr", {}).get("curr")) else None
    snap_cb = SnapshotCallback(model, venv, a.out, snapshots, logf=logf) if snapshots else None
    t0 = time.time()

    def _cb(loc, glo):
        ok = val_cb(loc, glo)
        if not ok:
            return False
        ok = prog_cb(loc, glo)
        if curr_cb is not None:
            ok = ok and curr_cb(loc, glo)
        if snap_cb is not None:
            ok = ok and snap_cb(loc, glo)
        return ok

    model.learn(total_timesteps=total_steps, callback=_cb, reset_num_timesteps=True)
    wall = time.time() - t0

    # ensure at least one checkpoint exists even if no validation beat the initial -inf
    if not os.path.exists(os.path.join(a.out, "best_model.zip")):
        model.save(os.path.join(a.out, "best_model.zip"))
        if hasattr(venv, "save"):
            venv.save(os.path.join(a.out, "vec_normalize.pkl"))

    summary = dict(
        config=a.config, config_hash=cfg_hash(spec), seed=int(a.seed), steps=total_steps,
        wall_seconds=wall, best_val=val_cb.best, val_history=val_cb.history,
        algo=algo, init_from=init_from, snapshots=snapshots,
        device=device, finished=now(), cmd=" ".join(sys.argv),
        versions={m: __import__(m).__version__ for m in ("numpy", "torch", "stable_baselines3")},
    )
    with open(os.path.join(a.out, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("TRAIN DONE best_val=%.4f wall=%.0fs RSS_MB=%.0f" % (val_cb.best, wall, rss_mb()), logf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
