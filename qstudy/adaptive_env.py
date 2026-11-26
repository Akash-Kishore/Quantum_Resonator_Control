"""qstudy adaptive_env — Part-2 S17.1: adaptive-sensing environment.

Environment probe "adaptive" in BatchEnv with physical_time=True and drift gaussian
sigma_base 500 Hz at 1024 shots; step drift sigma = sigma_base*sqrt(N/1024) (more shots
cost time = more drift). Action Box(3): a0 control; a1 mode (rabi if a1<-1/3, dither if
-1/3<=a1<1/3, ramsey_iq if a1>=1/3); a2 shots (64 if a2<-1/3, 256 if -1/3<=a2<1/3, 1024 if
a2>=1/3). Hard bins, documented.

Observation (fixed layout): [fc/25000, prev a0, onehot(prev mode) (3), log2(N_prev)/10,
(mu-fc)/5000, sigma_f/5000, f1, f2, f3] where an always-on GridBayes filter (assumed nominal
parameters, probe model = chosen mode and shots) gives mu, sigma_f, and f1..f3 are the raw
features of the last measurement in a unified 3-slot layout (rabi: [2p-1,0,0]; dither:
[2p+ -1, 2p- -1, 2(p+ - p-)]; ramsey: [2pI-1, 2pQ-1, phase/pi]).

Reward: base reward - lambda*N/1024 (lambda in {0, 0.02, 0.05, 0.1, 0.2}).

Implementation: AdaptiveEnv composes a BatchEnv-like physics core from envs.py by wrapping
its measurement and stepping with per-step drift scaling. Used via QuantumCalibGym-compatible
gym wrapper for training (seeds 400-404 per lambda).
"""
import numpy as np

import qphysics as qp
from qphysics import CLIP, sample_counts, readout

MODES = ("rabi", "dither", "ramsey_iq")
SHOT_BINS = (64, 256, 1024)


def mode_of(a1):
    return MODES[0] if a1 < -1.0 / 3.0 else (MODES[1] if a1 < 1.0 / 3.0 else MODES[2])


def shots_of(a2):
    return SHOT_BINS[0] if a2 < -1.0 / 3.0 else (SHOT_BINS[1] if a2 < 1.0 / 3.0 else SHOT_BINS[2])


class AdaptiveFilter:
    """Always-on GridBayes filter with assumed NOMINAL parameters; probe model follows the
    chosen mode and shots (S17.1 observation layout)."""

    def __init__(self, cfg, B, seed_base):
        sigma = float(cfg.get("filter_sigma", cfg.get("drift_sigma", 500.0)))
        self.gb = qp.GridBayes(half_range=30000.0, n=601, sigma=sigma, theta=0.0)
        self.post = self.gb.init(B)
        self.window = float(cfg.get("filter_window", 5000.0))
        self.mu = np.zeros(B)
        self.sigma_f = np.full(B, 8000.0)
        self.omega = float(cfg.get("omega", 5000.0))
        self.dither_delta = float(cfg.get("dither_delta", 2500.0))
        self.tau = float(cfg.get("tau", 50e-6))
        self.t2 = float(cfg.get("t2", 200e-6))
        self.e01 = float(cfg.get("e01", 0.02))
        self.e10 = float(cfg.get("e10", 0.02))

    def _p(self, mode, D, n):
        if mode == "rabi":
            return qp.readout(qp.rabi_p(D, omega=self.omega), self.e01, self.e10)
        if mode == "dither":
            dd = self.dither_delta
            return (qp.readout(qp.rabi_p(D - dd, omega=self.omega), self.e01, self.e10),
                    qp.readout(qp.rabi_p(D + dd, omega=self.omega), self.e01, self.e10))
        return (qp.readout(qp.ramsey_p(D, self.tau, self.t2, "I"), self.e01, self.e10),
                qp.readout(qp.ramsey_p(D, self.tau, self.t2, "Q"), self.e01, self.e10))

    def update(self, mode, counts, n, fc):
        self.post = self.gb.predict(self.post)
        D = self.gb.g[None, :] - fc[:, None]
        if mode == "rabi":
            ll = qp.binom_loglik(counts[:, None], int(n), np.clip(self._p("rabi", D, n), 1e-6, 1 - 1e-6))
        elif mode == "dither":
            pp, pm = self._p("dither", D, n)
            ll = (qp.binom_loglik(counts[:, 0][:, None], int(n), np.clip(pp, 1e-6, 1 - 1e-6)) +
                  qp.binom_loglik(counts[:, 1][:, None], int(n), np.clip(pm, 1e-6, 1 - 1e-6)))
        else:
            pI, pQ = self._p("ramsey_iq", D, n)
            ll = (qp.binom_loglik(counts[:, 0][:, None], int(n), np.clip(pI, 1e-6, 1 - 1e-6)) +
                  qp.binom_loglik(counts[:, 1][:, None], int(n), np.clip(pQ, 1e-6, 1 - 1e-6)))
        self.post = self.gb.update(self.post, ll)
        self.mu, self.sigma_f = self.gb.mean_std(self.post, window=self.window)
        return self.mu, self.sigma_f


class AdaptiveEnv:
    """Batched adaptive-sensing environment (S17.1). One physics core: x drifts per step with
    sigma = sigma_base*sqrt(N/1024) (physical-time coupling); the measurement uses the chosen
    mode and shots; the reward is the BatchEnv base reward minus lambda*N/1024."""

    def __init__(self, cfg, B, seed_base):
        c = dict(cfg)
        c.setdefault("probe", "adaptive")
        self.cfg = c
        self.B = int(B)
        self.seed_base = int(seed_base)
        self.T = int(c.get("horizon", 200))
        self.burn_in = int(c.get("burn_in", 20))
        self.sigma_base = float(c.get("drift_sigma", 500.0))
        self.lam = float(c.get("adaptive_lambda", 0.05))
        self.action_scale = float(c.get("action_scale", 3000.0))
        self.reward_width = float(c.get("reward_width", 4000.0))
        self.action_penalty = float(c.get("action_penalty", 0.01))
        self.init_offset = float(c.get("init_offset", 5000.0))
        self.dac_bits = c.get("dac_bits", 12)
        self.omega = float(c.get("omega", 5000.0))
        self.dither_delta = float(c.get("dither_delta", 2500.0))
        self.tau = float(c.get("tau", 50e-6))
        self.t2 = float(c.get("t2", 200e-6))
        self.e01 = float(c.get("e01", 0.02))
        self.e10 = float(c.get("e10", 0.02))
        self._flt = AdaptiveFilter(c, self.B, seed_base)
        self.reset()

    def obs_dim(self):
        return 11  # 2 + 3 + 1 + 2 + 3

    def reset(self):
        self.t = 0
        self.fc = np.zeros(self.B)
        self.prev_a0 = np.zeros(self.B)
        self.mode = np.array(["rabi"] * self.B)
        self.N = np.full(self.B, 1024)
        rng = np.random.default_rng([self.seed_base, 999])
        self.x = rng.uniform(-self.init_offset, self.init_offset, self.B)
        self.mrng = [np.random.default_rng([self.seed_base, b, 2]) for b in range(self.B)]
        self._flt = AdaptiveFilter(self.cfg, self.B, self.seed_base)
        self._measure()
        return self._obs()

    def _measure(self):
        """Measure at detuning x - fc with each episode's current mode/shots; update filter."""
        B = self.B
        cnt = np.zeros((B, 2))
        for b in range(B):
            mode, n = self.mode[b], int(self.N[b])
            rng = self.mrng[b]
            d = float(self.x[b] - self.fc[b])
            if mode == "rabi":
                p = readout(qp.rabi_p(d, omega=self.omega), self.e01, self.e10)
                cnt[b, 0] = rng.binomial(n, float(np.clip(p, 0, 1)))
            elif mode == "dither":
                m = n // 2
                pp = readout(qp.rabi_p(d - self.dither_delta, omega=self.omega), self.e01, self.e10)
                pm = readout(qp.rabi_p(d + self.dither_delta, omega=self.omega), self.e01, self.e10)
                cnt[b, 0] = rng.binomial(m, float(np.clip(pp, 0, 1)))
                cnt[b, 1] = rng.binomial(m, float(np.clip(pm, 0, 1)))
            else:
                m = n // 2
                pI = readout(qp.ramsey_p(d, self.tau, self.t2, "I"), self.e01, self.e10)
                pQ = readout(qp.ramsey_p(d, self.tau, self.t2, "Q"), self.e01, self.e10)
                cnt[b, 0] = rng.binomial(m, float(np.clip(pI, 0, 1)))
                cnt[b, 1] = rng.binomial(m, float(np.clip(pQ, 0, 1)))
        self.counts = cnt
        self.shots_used = self.N.copy()
        # vectorised filter update, one group per mode (modes can differ per episode)
        for mode in MODES:
            idx = np.where(self.mode == mode)[0]
            if idx.size == 0:
                continue
            n = int(self.N[idx[0]])
            D = self._flt.gb.g[None, :] - self.fc[idx][:, None]
            if mode == "rabi":
                p = readout(qp.rabi_p(D, omega=self._flt.omega), self.e01, self.e10)
                ll = qp.binom_loglik(cnt[idx, 0][:, None], n, np.clip(p, 1e-6, 1 - 1e-6))
            elif mode == "dither":
                m = n // 2
                dd = self._flt.dither_delta
                pp = readout(qp.rabi_p(D - dd, omega=self._flt.omega), self.e01, self.e10)
                pm = readout(qp.rabi_p(D + dd, omega=self._flt.omega), self.e01, self.e10)
                ll = (qp.binom_loglik(cnt[idx, 0][:, None], m, np.clip(pp, 1e-6, 1 - 1e-6)) +
                      qp.binom_loglik(cnt[idx, 1][:, None], m, np.clip(pm, 1e-6, 1 - 1e-6)))
            else:
                m = n // 2
                pI = readout(qp.ramsey_p(D, self._flt.tau, self._flt.t2, "I"), self.e01, self.e10)
                pQ = readout(qp.ramsey_p(D, self._flt.tau, self._flt.t2, "Q"), self.e01, self.e10)
                ll = (qp.binom_loglik(cnt[idx, 0][:, None], m, np.clip(pI, 1e-6, 1 - 1e-6)) +
                      qp.binom_loglik(cnt[idx, 1][:, None], m, np.clip(pQ, 1e-6, 1 - 1e-6)))
            post = self._flt.gb.predict(self._flt.post[idx])
            self._flt.post[idx] = self._flt.gb.update(post, ll)
        mu, sd = self._flt.gb.mean_std(self._flt.post, window=self._flt.window)
        self._flt.mu, self._flt.sigma_f = mu, sd
        # raw unified features of the last measurement
        f = np.zeros((self.B, 3))
        for b in range(B):
            mode, n = self.mode[b], int(self.N[b])
            m = n // 2
            if mode == "rabi":
                p = cnt[b, 0] / n
                f[b] = [2 * p - 1, 0.0, 0.0]
            elif mode == "dither":
                pp, pm = cnt[b, 0] / m, cnt[b, 1] / m
                f[b] = [2 * pp - 1, 2 * pm - 1, 2 * (pp - pm)]
            else:
                pI, pQ = cnt[b, 0] / m, cnt[b, 1] / m
                ph = np.arctan2(2 * pQ - 1, 2 * pI - 1)
                f[b] = [2 * pI - 1, 2 * pQ - 1, ph / np.pi]
        self.f = f

    def _obs(self):
        onehot = np.stack([ (self.mode == m).astype(float) for m in MODES ], axis=1)
        o = [self.fc / 25000.0, self.prev_a0, onehot[:, 0], onehot[:, 1], onehot[:, 2],
             np.log2(self.N) / 10.0,
             (self._flt.mu - self.fc) / 5000.0, self._flt.sigma_f / 5000.0,
             self.f[:, 0], self.f[:, 1], self.f[:, 2]]
        return np.stack(o, axis=1).astype(np.float32)

    def step(self, action):
        """action layout: (B, 3) with columns [a0 control, a1 mode, a2 shots].
        A (3,) single action is tiled over the batch. Hard bins (documented): mode rabi if
        a1<-1/3, dither if -1/3<=a1<1/3, ramsey_iq if a1>=1/3; shots 64/256/1024 by the same
        thresholds on a2."""
        act = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        if act.size == 3:
            act = np.tile(act, self.B)
        assert act.size == 3 * self.B, "adaptive action must be (B,3) or (3,), got %r" % (action.shape
                                                                                            if hasattr(action, "shape") else action)
        a0 = act[0::3]
        a1 = act[1::3]
        a2 = act[2::3]
        new_mode = np.array([mode_of(v) for v in a1])
        new_N = np.array([shots_of(v) for v in a2], dtype=np.int64)
        # (1) apply control; (2) drift with physical-time coupling sigma*sqrt(N/1024);
        # (3) measure with the NEW mode and shots; (4) error at probe time; (5) reward.
        fc_new = qp.quantize(self.fc + a0 * self.action_scale, self.dac_bits)
        self.fc = np.clip(fc_new, -CLIP, CLIP)
        self.mode, self.N = new_mode, new_N
        for b in range(self.B):
            rng = np.random.default_rng([self.seed_base, b, 1, self.t])
            self.x[b] = float(np.clip(self.x[b] + self.sigma_base * np.sqrt(self.N[b] / 1024.0)
                                      * rng.normal(), -CLIP, CLIP))
        self._measure()
        e = self.x - self.fc
        rew = 1.0 - np.minimum(1.0, np.abs(e) / self.reward_width) - self.action_penalty * np.abs(a0)
        rew = rew - self.lam * self.N / 1024.0
        self.prev_a0 = a0
        self.t += 1
        info = dict(e=e, fc=self.fc.copy(), x=self.x.copy(), counts=self.counts,
                    shots_used=self.shots_used.copy(), mode=self.mode.copy(),
                    posterior_sigma=self._flt.sigma_f.copy())
        done = self.t >= self.T
        return self._obs(), rew, done, info

    def metrics(self, e_tr):
        """Same metric helper as BatchEnv (burn-in excluded)."""
        T, B = e_tr.shape
        bi = self.burn_in
        out = {}
        for W in (500, 1000, 2000, 3333, 5000):
            out["lock_w%d" % W] = (np.abs(e_tr[bi:]) <= W).mean(axis=0)
        out["rms"] = np.sqrt((e_tr[bi:] ** 2).mean(axis=0))
        out["med_abs"] = np.median(np.abs(e_tr[bi:]), axis=0)
        out["p95_abs"] = np.percentile(np.abs(e_tr[bi:]), 95, axis=0)
        out["max_abs"] = np.abs(e_tr[bi:]).max(axis=0)
        over = (np.abs(e_tr[bi:]) > 10000).astype(float)
        run20 = np.ones(B)
        c = 0
        for t in range(over.shape[0]):
            c = (c + 1) if over[t].any() else 0
            if c >= 20:
                run20 = 0.0
        out["lock_lost"] = run20
        return out
