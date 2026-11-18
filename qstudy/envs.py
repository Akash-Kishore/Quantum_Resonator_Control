"""qstudy envs — ONE physics implementation used for training, evaluation and everything else."""
import numpy as np
import gymnasium as gym
import qphysics as qp

F_NOM = qp.F_NOM
CLIP = qp.CLIP

DEFAULT_CFG = dict(
    probe="rabi",            # "rabi" | "dither" | "ramsey_iq" | "rabi_legacy"
    engine="surrogate",       # "surrogate"; an AerEngine instance may be injected via engine=
    drift="gaussian",         # gaussian | ou | onef | telegraph | sinusoid
    drift_sigma=500.0,
    shots=1024,               # TOTAL per control step; two-circuit probes split it in halves
    omega=5000.0,
    dither_delta=2500.0,
    tau=50e-6,
    t2=200e-6,
    e01=0.02,
    e10=0.02,
    contrast_power=1.0,       # Part-2 S16.2: Ramsey contrast decay exponent (1=exp, 2=Gaussian)
    contrast_override=None,   # Part-2 S16.2: fixed visibility (photonic); None = use t2/power
    drift_params=None,        # Part-2 S16.2: dict of kind-specific DriftExt parameters
    profile=None,             # Part-2 S16.2: informational platform tag
    action_scale=3000.0,
    action_delay=0,
    dac_bits=12,
    obs_bits=None,
    horizon=200,
    burn_in=20,
    init_offset=5000.0,
    reward_width=4000.0,
    reward_kind="base",        # Part-2 S17.8: "base" | "threshold" (+1 if |e|<=1000 Hz else 0, minus 0.01|a|)
    action_penalty=0.01,
    filter_features=False,
    filter_sigma=None,
    filter_theta=None,
    filter_window=5000.0,
    stack=1,
    dr=None,
)

LOCK_WIDTHS = (500, 1000, 2000, 3333, 5000)


class BatchEnv:
    """Vectorised batch of independent tracking episodes.

    State per episode b: x = true offset of the frequency from F_NOM (Hz), fc = control offset.
    Drift RNG per episode = default_rng([seed_base, ep, 1]); measurement RNG = default_rng([seed_base, ep, 2]),
    so every method sees identical drift realisations (paired design).
    """

    def __init__(self, cfg, B, seed_base, engine=None):
        self.cfg = dict(DEFAULT_CFG)
        self.cfg.update(cfg or {})
        self.B = int(B)
        self.seed_base = int(seed_base)
        if engine is None:
            eng_kind = str(self.cfg.get("engine", "surrogate"))
            if eng_kind == "aer":
                from engines import AerEngine
                engine = AerEngine(cfg)
            elif eng_kind == "hybrid":
                from hybrid_engines import HybridEngine
                engine = HybridEngine(cfg)
            else:
                from engines import SurrogateEngine
                engine = SurrogateEngine(self.cfg)
        self.engine = engine
        self.probe = str(self.cfg["probe"])
        self.T = int(self.cfg["horizon"])
        self.stack = int(self.cfg["stack"])
        self.filter_on = bool(self.cfg["filter_features"])
        self._flt = None
        self.t = 0
        self._reset_done = False

    # ---------------- per-episode parameter arrays (Stage 8/12 support)
    def _ep_arr(self, key, default, dtype=float):
        v = self.cfg.get("ep_" + key, None)
        if v is None:
            return np.full(self.B, default, dtype=dtype)
        v = np.asarray(v, dtype=dtype)
        assert v.shape == (self.B,), "ep_%s must be a (B,) array" % key
        return v

    # ---------------- domain randomisation (Stage 11; surrogate-only)
    def _dr_sample(self):
        """Per-episode DR draws, interpolated toward nominal by self.dr_scale in [0,1]:
        value = nominal + s*(sample - nominal) for numeric parameters; categorical parameters
        take the sampled value with probability s else nominal. Dedicated RNG stream
        [seed_base, 999] keeps the paired drift/measure streams untouched."""
        c = self.cfg
        rng = np.random.default_rng([self.seed_base, 999])
        B = self.B
        s = self.dr_scale
        nom = dict(shots=float(c["shots"]), e01=float(c["e01"]), e10=float(c["e10"]),
                   t2=float(c["t2"]), sigma=float(c["drift_sigma"]),
                   delay=float(c["action_delay"]), dac_bits=float(c["dac_bits"]))
        logu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), B))
        sam = dict(shots=np.round(logu(64, 4096)).astype(float),
                   e01=rng.uniform(0, 0.05, B), e10=rng.uniform(0, 0.08, B),
                   t2=logu(40e-6, 400e-6), sigma=logu(250, 1500),
                   delay=rng.integers(0, 3, B).astype(float),
                   dac_bits=rng.choice([8, 10, 12], B).astype(float))
        out = {k: nom[k] + s * (sam[k] - nom[k]) for k in nom}
        # Part-2 S17.7: optional restriction of the DR drift-kind pool (cfg dr["kinds"],
        # e.g. ["gaussian","ou"] for DR-restricted training). Default path unchanged.
        kinds = list(c.get("dr", {}).get("kinds") or ["gaussian", "ou", "onef", "telegraph", "sinusoid"])
        k_idx = rng.integers(0, len(kinds), B)
        take = rng.uniform(0, 1, B) < s
        base_kind = str(c["drift"])
        out["kind"] = [kinds[k_idx[b]] if take[b] else base_kind for b in range(B)]
        out["shots"] = np.clip(np.round(out["shots"]), 16, 1000000).astype(np.int64)
        out["dac_bits"] = int(np.clip(np.round(out["dac_bits"][0]), 6, 16))
        out["delay"] = np.clip(np.round(out["delay"]), 0, 3).astype(int)
        return out

    def set_curriculum(self, s):
        """Called by the training callback: interpolation scale toward nominal (Stage 11.1)."""
        self.dr_scale = float(s)

    # ---------------- lifecycle
    def reset(self):
        c = self.cfg
        self.ep_shots = self._ep_arr("shots", c["shots"], np.int64)
        self.ep_e01 = self._ep_arr("e01", c["e01"], float)
        self.ep_e10 = self._ep_arr("e10", c["e10"], float)
        self.ep_t2 = self._ep_arr("t2", c["t2"], float)
        self.ep_sigma = self._ep_arr("sigma", c["drift_sigma"], float)
        self.ep_delay = self._ep_arr("delay", c["action_delay"], int)
        ep_kind = None
        if c.get("dr"):
            if not hasattr(self, "dr_scale"):
                self.dr_scale = float(c["dr"].get("scale", 1.0))
            drs = self._dr_sample()
            self.ep_shots = drs["shots"]
            self.ep_e01 = drs["e01"]
            self.ep_e10 = drs["e10"]
            self.ep_t2 = drs["t2"]
            self.ep_sigma = drs["sigma"]
            self.ep_delay = drs["delay"]
            self.cfg["dac_bits"] = drs["dac_bits"]  # per-batch DAC bits; per-episode would need per-ep quantisation (documented)
            ep_kind = drs["kind"]
        self.t = 0
        self.fc = np.zeros(self.B)
        self.prev_action = np.zeros(self.B)
        self.phat_prev = None
        self.frames = None
        self._act_hist = [np.zeros(self.B)]
        self.drifts, self.mrng, self.x = [], [], np.zeros(self.B)
        # Part-2 S16.2: DriftExt for every drift kind (identical to Drift at defaults);
        # cfg drift_params dict carries kind-specific parameters (flip, period_range,
        # theta_list, std_total); profile is informational only.
        from qphysics_ext import DriftExt
        dp = dict(c.get("drift_params") or {})
        for b in range(self.B):
            rng_d = np.random.default_rng([self.seed_base, b, 1])
            kind = ep_kind[b] if ep_kind is not None else str(c["drift"])
            d = DriftExt(kind, rng_d, sigma=float(self.ep_sigma[b]), **dp)
            self.x[b] = d.reset(rng_d.uniform(-float(c["init_offset"]), float(c["init_offset"])))
            self.drifts.append(d)
            self.mrng.append(np.random.default_rng([self.seed_base, b, 2]))
        if self.filter_on:
            from controllers import BayesTracker
            # M2 F2: filter features always assume the frozen N0 (nominal); an explicit
            # cfg filter_assumed="oracle" is allowed only for labelled diagnostics.
            assumed = str(c.get("filter_assumed", "nominal"))
            assumed = assumed if assumed in ("nominal", "oracle") else "nominal"
            if bool(c.get("cusum")):  # Part-2 S17.2: Bayes filter + CUSUM detector
                from changepoint import BayesCusum
                self._flt = BayesCusum(c, assumed=assumed, env_view=self)
            else:
                self._flt = BayesTracker(c, assumed=assumed, env_view=self)
        self._reset_done = True
        self._measure()  # one measurement at fc=0 (no drift step)
        if self.filter_on and self._flt is not None:
            self._flt.step_env_filter()  # filter also processes the reset measurement
        return self.obs()

    # ---------------- action delay + DAC
    def _apply_action(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        if a.shape[0] == 1 and self.B > 1:
            a = np.full(self.B, float(a[0]))
        self._act_hist.append(a.copy())
        hist = self._act_hist
        a_eff = np.empty(self.B)
        for b in range(self.B):
            k = int(self.ep_delay[b])
            a_eff[b] = hist[len(hist) - 1 - k][b] if k < len(hist) else 0.0
        fc_new = qp.quantize(self.fc + a_eff * float(self.cfg["action_scale"]), self.cfg["dac_bits"])
        self.fc = np.clip(fc_new, -CLIP, CLIP)
        self.prev_action = a

    # ---------------- measurement (sets self._last_extra and self.counts)
    def _measure(self):
        c = self.cfg
        delta = self.x - self.fc
        k = self.probe
        if k in ("rabi", "rabi_legacy"):
            cnt = self.engine.counts("rabi" if k == "rabi" else "rabi_legacy", delta,
                                     self.ep_shots, self.ep_t2, self.ep_e01, self.ep_e10, self.mrng,
                                     omega=float(c["omega"]))
            self.counts = cnt
            self.shots_used = self.ep_shots
            phat = cnt / self.ep_shots
            if c["obs_bits"] is not None:
                lv = 2 ** int(c["obs_bits"])
                phat = np.round(phat * lv) / lv
            prev = self.phat_prev if self.phat_prev is not None else phat
            extra = np.stack([2 * phat - 1, 2 * (phat - prev)], axis=1)
            self._cur_phat = phat
        elif k == "dither":
            n = (self.ep_shots // 2).astype(np.int64)
            cp = self.engine.counts("rabi", delta - float(c["dither_delta"]), n, self.ep_t2,
                                    self.ep_e01, self.ep_e10, self.mrng, omega=float(c["omega"]))
            cm = self.engine.counts("rabi", delta + float(c["dither_delta"]), n, self.ep_t2,
                                    self.ep_e01, self.ep_e10, self.mrng, omega=float(c["omega"]))
            self.counts = np.stack([cp, cm], axis=1)
            self.shots_used = self.ep_shots
            pp, pm = cp / n, cm / n
            if c["obs_bits"] is not None:
                lv = 2 ** int(c["obs_bits"])
                pp = np.round(pp * lv) / lv
                pm = np.round(pm * lv) / lv
            extra = np.stack([2 * pp - 1, 2 * pm - 1, 2 * (pp - pm)], axis=1)
            self._cur_phat = None
        elif k == "ramsey_iq":
            n = (self.ep_shots // 2).astype(np.int64)
            cI = self.engine.counts("ramsey_I", delta, n, self.ep_t2, self.ep_e01, self.ep_e10,
                                    self.mrng, tau=float(c["tau"]))
            cQ = self.engine.counts("ramsey_Q", delta, n, self.ep_t2, self.ep_e01, self.ep_e10,
                                    self.mrng, tau=float(c["tau"]))
            self.counts = np.stack([cI, cQ], axis=1)
            self.shots_used = self.ep_shots
            pI, pQ = cI / n, cQ / n
            if c["obs_bits"] is not None:
                lv = 2 ** int(c["obs_bits"])
                pI = np.round(pI * lv) / lv
                pQ = np.round(pQ * lv) / lv
            phase = qp.ramsey_phase(pI, pQ)
            extra = np.stack([2 * pI - 1, 2 * pQ - 1, phase / np.pi], axis=1)
            self._cur_phat = None
        else:
            raise ValueError(self.probe)
        self._last_extra = extra.astype(np.float32)

    # ---------------- observation assembly
    def obs(self):
        parts = [np.stack([self.fc / 25000.0, self.prev_action], axis=1).astype(np.float32),
                 self._last_extra]
        if self.filter_on and self._flt is not None and getattr(self._flt, "last_mu", None) is not None:
            parts.append(np.stack([(self._flt.last_mu - self.fc) / 5000.0,
                                   self._flt.last_std / 5000.0], axis=1).astype(np.float32))
        frame = np.concatenate(parts, axis=1).astype(np.float32)
        if self.stack > 1:
            if self.frames is None:
                self.frames = [frame.copy() for _ in range(self.stack)]
            else:
                self.frames = self.frames[1:] + [frame.copy()]
            frame = np.concatenate(self.frames, axis=1)
        return frame

    def obs_dim(self):
        d = 2 + (2 if self.probe in ("rabi", "rabi_legacy") else 3)
        if self.filter_on:
            d += 2
        return d * self.stack

    # ---------------- public state for controllers (NEVER x)
    def public_state(self):
        return dict(counts=self.counts, shots=self.shots_used, fc=self.fc.copy(), probe=self.probe)

    # ---------------- step
    def step(self, action):
        assert self._reset_done, "call reset() first"
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        if a.shape[0] == 1 and self.B > 1:
            a = np.full(self.B, float(a[0]))
        self._apply_action(a)
        for b in range(self.B):
            self.x[b] = self.drifts[b].step()
        self._measure()
        if self.filter_on and self._flt is not None:
            self._flt.step_env_filter()
        e = self.x - self.fc  # ERROR AT PROBE TIME
        if str(self.cfg.get("reward_kind", "base")) == "threshold":
            # S17.8 R2: +1 if |e|<=1000 Hz else 0, minus 0.01|a| (threshold fixed by spec)
            reward = (np.abs(e) <= 1000.0).astype(float) - float(self.cfg["action_penalty"]) * np.abs(a)
        else:
            reward = 1.0 - np.minimum(1.0, np.abs(e) / float(self.cfg["reward_width"])) - float(self.cfg["action_penalty"]) * np.abs(a)
        self.t += 1
        done = self.t >= self.T
        if self.probe in ("rabi", "rabi_legacy"):
            self.phat_prev = self._cur_phat
        info = dict(e=e.copy(), fc=self.fc.copy(), x=self.x.copy(),
                    counts=self.counts, shots_used=self.shots_used, reward=reward)
        return self.obs(), reward, bool(done), info

    # ---------------- metrics helper
    def metrics(self, e_traces):
        """e_traces: (T, B) errors at probe time -> dict of per-episode (B,) arrays."""
        e = np.abs(np.asarray(e_traces, dtype=float))
        T, B = e.shape
        bi = int(self.cfg["burn_in"])
        post = e[bi:, :]
        out = {}
        for w in LOCK_WIDTHS:
            out["lock_w%d" % w] = (post <= w).mean(axis=0)
        out["rms"] = np.sqrt((post ** 2).mean(axis=0))
        out["med_abs"] = np.median(post, axis=0)
        out["p95_abs"] = np.percentile(post, 95, axis=0)
        out["max_abs"] = post.max(axis=0)
        over = post > 10000
        lost = np.zeros(B)
        for b in range(B):
            run = 0
            for v in over[:, b]:
                run = run + 1 if v else 0
                if run >= 20:
                    lost[b] = 1.0
                    break
        out["lock_lost"] = lost
        return out


def mean_reward(r_traces, burn_in=20):
    r = np.asarray(r_traces, dtype=float)
    return r[burn_in:, :].mean(axis=0)


class QuantumCalibGym(gym.Env):
    """Gymnasium single-episode wrapper around BatchEnv(B=1); seeds from reset(seed)."""
    metadata = {"render_modes": []}

    def __init__(self, cfg=None):
        from gymnasium import spaces
        c = dict(DEFAULT_CFG)
        if cfg:
            c.update(cfg)
        self.cfg = c
        probe_dim = 2 if c["probe"] in ("rabi", "rabi_legacy") else 3
        d = 2 + probe_dim + (2 if c["filter_features"] else 0)
        self.observation_space = spaces.Box(-10.0, 10.0, (d * int(c["stack"]),), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self._env = None
        self._seed = None

    def _ensure(self, seed):
        if self._env is None or (seed is not None and seed != self._seed):
            self._env = BatchEnv(self.cfg, 1, int(seed) if seed is not None else 0)
            self._seed = seed

    def set_curriculum(self, s):
        if self._env is not None:
            self._env.set_curriculum(s)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None and self._env is None:
            seed = int(self.cfg.get("env_seed", 0))
        self._ensure(seed)
        obs = self._env.reset()
        return obs[0].astype(np.float32), {}

    def step(self, action):
        obs, rew, done, info = self._env.step(np.asarray(action, dtype=float).reshape(1))
        return (obs[0].astype(np.float32), float(rew[0]), bool(done), False,
                {"x": float(info["x"][0]), "fc": float(info["fc"][0]), "e": float(info["e"][0])})
