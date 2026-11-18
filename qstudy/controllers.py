"""qstudy controllers — NoControl, Oracle, HillClimb now; PID (Stage 6) and BayesTracker (Stage 9) added later."""
import numpy as np

# ---------------------------------------------------------------- M2 F1: frozen nominal assumption N0
# ctrl:bayes:nominal and the hybrid filter features (cfg filter_features=True) may use
# ONLY this dict plus observables (counts, shots used this step, fc, probe type).
NOMINAL_ASSUMPTION_N0 = {
    "drift": "random_walk",      # random walk with theta 0
    "theta": 0.0,
    "sigma": 500.0,              # Hz
    "tau": 50e-6,
    "t2": 200e-6,                # exponential contrast (power 1)
    "contrast_power": 1.0,
    "contrast_override": None,
    "e01": 0.02,
    "e10": 0.02,
    "omega": 5000.0,
    "dither_delta": 2500.0,
}

_EMPIRICAL_DRIFT_CACHE = {}


def empirical_drift_std(cfg, n_steps=2000):
    """M2 F1: empirical per-step std of the drift defined by cfg (used by the oracle
    filter for onef/telegraph/sinusoid/regime drifts): simulate n_steps with the cell's
    own cfg and return std(x_t - x_{t-1}). Cached per cell parameters."""
    import json as _json
    dp = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in dict(cfg.get("drift_params") or {}).items()}
    key = (str(cfg.get("drift", "gaussian")), float(cfg.get("drift_sigma", 500.0) or 500.0),
           _json.dumps(dp, sort_keys=True), int(n_steps))
    if key in _EMPIRICAL_DRIFT_CACHE:
        return _EMPIRICAL_DRIFT_CACHE[key]
    from qphysics_ext import DriftExt
    rng = np.random.default_rng(20261015)
    d = DriftExt(key[0], rng, sigma=key[1], **{k: (tuple(v) if isinstance(v, list) else v) for k, v in dp.items()})
    xs = np.empty(n_steps + 1)
    xs[0] = d.reset(0.0)
    for i in range(1, n_steps + 1):
        xs[i] = d.step()
    s = float(np.std(np.diff(xs)))
    _EMPIRICAL_DRIFT_CACHE[key] = s
    return s




class NoControl:
    """Never moves."""
    def __init__(self, cfg, env=None):
        self.cfg = cfg
        self.env = env

    def act(self, obs_or_state):
        return np.zeros(self.env.B)


class Oracle:
    """Uses the previous true frequency x_prev: fc <- x_prev. Ceiling reference.
    Needs BatchEnv internal access; used only for reference rows and flagged ORACLE.

    act() is called BEFORE each step; at that moment env.x is the most recent drifted
    value x_{t-1}. Setting fc <- env.x makes the step-t error e_t = x_t - x_{t-1},
    i.e. exactly one drift increment -> the pre-registered ceiling 0.954 at sigma=500."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env

    def act(self, obs_or_state):
        env = self.env
        return np.clip((env.x - env.fc) / float(self.cfg.get("action_scale", 3000.0)), -1.0, 1.0)


class HillClimb:
    """Perturb-and-observe on the single rabi probe: keep direction; reverse when phat drops by more than thr."""

    def __init__(self, cfg, env, step_hz=1000.0, thr=0.02):
        self.cfg = cfg
        self.env = env
        self.step_hz = float(step_hz)
        self.thr = float(thr)
        self.dir = np.ones(env.B)
        self.phat_prev = None

    def _phat(self):
        st = self.env.public_state()
        counts = st["counts"]
        shots = st["shots"]
        if st["probe"] in ("rabi", "rabi_legacy"):
            return (counts / shots).astype(float)
        raise ValueError("HillClimb supports the single rabi probe only, got %s" % st["probe"])

    def act(self, obs_or_state):
        phat = self._phat()
        if self.phat_prev is not None:
            drop = self.phat_prev - phat
            self.dir = np.where(drop > self.thr, -self.dir, self.dir)
        self.phat_prev = phat
        return self.dir * self.step_hz / float(self.cfg.get("action_scale", 3000.0))



class PIDController:
    """PID on the signed probe error signal.
    dither:     err = obs entry 2*(p_plus - p_minus)  (obs index 4)
    ramsey_iq:  err = obs entry phase/pi             (obs index 4)
    a = Kp*err + Ki*sum(err) + Kd*(err - err_prev), integral anti-windup clipped to +-1.
    Gains are grid-tuned on TUNE seeds (Stage 6.2) and saved to configs\\pid_gains.json."""

    def __init__(self, cfg, env, gains=None):
        self.cfg = cfg
        self.env = env
        probe = env.probe if env is not None else cfg.get("probe")
        if probe not in ("dither", "ramsey_iq"):
            raise ValueError("PIDController supports probes dither and ramsey_iq (for the single "
                             "rabi probe use HillClimb); got %s" % probe)
        self.gains_source = "explicit" if gains is not None else "defaults"
        if gains is None:
            import json, os
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "pid_gains.json")
            g = {}
            if os.path.exists(path):
                data = json.load(open(path, encoding="utf-8"))
                drift = cfg.get("drift", "gaussian")
                pd = data.get(probe, {})
                if pd.get(drift):
                    g = pd[drift]
                    self.gains_source = "pid_gains.json[%s][%s]" % (probe, drift)
                elif pd:
                    g = next(iter(pd.values()))
                    self.gains_source = "pid_gains.json[%s] fallback-first-entry (drift %r absent)" % (probe, drift)
            gains = dict(kp=float(g.get("kp", 0.7)), ki=float(g.get("ki", 0.02)), kd=float(g.get("kd", 0.0)))
        self.kp, self.ki, self.kd = gains["kp"], gains["ki"], gains["kd"]
        self.integral = np.zeros(env.B)
        self.err_prev = None

    def assumed_params(self):
        """D3 (P2): the gains actually used, recorded in every ctrl:pid result JSON."""
        return {"pid_gains_used": {"kp": float(self.kp), "ki": float(self.ki), "kd": float(self.kd)},
                "pid_gains_source": self.gains_source}

    def act(self, obs_or_state):
        obs = obs_or_state if isinstance(obs_or_state, np.ndarray) else None
        if obs is None or obs.ndim != 2 or obs.shape[1] < 5:
            # fall back to recomputing the error from public state (evaluate.py passes None)
            st = self.env.public_state()
            counts, shots = st["counts"], st["shots"]
            if st["probe"] == "dither":
                pp = counts[:, 0] / (shots // 2)
                pm = counts[:, 1] / (shots // 2)
                err = 2 * (pp - pm)
            else:
                pI = counts[:, 0] / (shots // 2)
                pQ = counts[:, 1] / (shots // 2)
                err = np.arctan2(2 * pQ - 1, 2 * pI - 1) / np.pi
        else:
            err = obs[:, 4]
        self.integral = np.clip(self.integral + err, -1.0, 1.0)
        d = (err - self.err_prev) if self.err_prev is not None else np.zeros_like(err)
        self.err_prev = err
        a = self.kp * err + self.ki * self.integral + self.kd * d
        return np.clip(a, -1.0, 1.0)

class BayesTracker:
    """Vectorised grid-Bayes tracker (Stage 9.1).
    assumed="nominal": the filter uses ONLY the frozen nominal assumption N0 (M2 F1) for
    the drift model and the probe likelihood, plus observables (counts, shots used this
    step, fc, probe type); assumed="oracle": it uses the episode's TRUE parameters (from
    the env view; non-Gaussian drifts are moment-matched to a random walk whose sigma is
    the empirical per-step std of the cell's drift).
    Grid: half_range 30000 Hz, n=601, window=filter_window (Hz). Per step: predict -> log-likelihood
    from the probe model -> update -> estimate = mean_std(window) mode; action = clip((mu-fc)/action_scale)."""

    def __init__(self, cfg, assumed="nominal", env_view=None):
        import qphysics as qp
        self.qp = qp
        self.cfg = cfg
        self.assumed = assumed if assumed in ("nominal", "oracle") else "nominal"
        self.env = env_view
        self.B = env_view.B if env_view is not None else int(cfg.get("B", 1))
        self.probe = cfg.get("probe", env_view.probe if env_view is not None else "rabi")
        self._init_filter()

    def _nominal(self, key, default):
        v = self.cfg.get(key, default)
        return float(v) if v is not None else float(default)

    def _init_filter(self):
        qp = self.qp
        if self.assumed == "oracle" and self.env is not None:
            sigma, theta = self._oracle_drift_params()
        else:
            # M2 F1: nominal uses ONLY the frozen assumption N0 (random walk, theta 0,
            # sigma 500 Hz); it must never read the cell's true drift_sigma/drift kind
            # (the old filter_sigma=None -> "true" behaviour is removed).
            sigma = float(NOMINAL_ASSUMPTION_N0["sigma"])
            theta = float(NOMINAL_ASSUMPTION_N0["theta"])
        self.gb = qp.GridBayes(half_range=30000.0, n=601, sigma=sigma, theta=theta)
        self.f_sigma, self.f_theta = float(sigma), float(theta)   # exposed for the S17.2 CUSUM inflation
        self.post = self.gb.init(self.B)
        self.window = self._nominal("filter_window", 5000.0)
        self.last_mu = None
        self.last_std = None
        self._first = True

    def _oracle_drift_params(self):
        """(sigma, theta) of the TRUE drift of the cell (F1): ou -> theta 0.05;
        gaussian -> random walk at the true sigma; onef/telegraph/sinusoid/regime ->
        random walk at the empirical per-step std of that drift (2000 simulated steps)."""
        e = self.env
        c = getattr(e, "cfg", None) or self.cfg
        kind = str(c.get("drift", "gaussian"))
        sigma = float(np.mean(e.ep_sigma))
        if kind == "ou":
            return sigma, 0.05
        if kind in ("onef", "telegraph", "sinusoid", "regime"):
            return empirical_drift_std(c), 0.0
        return sigma, 0.0

    def _assumed_probe_params(self):
        if self.assumed == "oracle" and self.env is not None:
            e = self.env
            c = getattr(e, "cfg", None) or self.cfg
            return dict(e01=float(np.mean(e.ep_e01)), e10=float(np.mean(e.ep_e10)),
                        t2=float(np.mean(e.ep_t2)),
                        tau=float(c.get("tau", 50e-6) or 50e-6),
                        omega=float(c.get("omega", 5000.0)),
                        contrast_power=float(c.get("contrast_power", 1.0) or 1.0),
                        contrast_override=c.get("contrast_override", None),
                        dither_delta=float(c.get("dither_delta", 2500.0)))
        # M2 F1: nominal probe likelihood uses ONLY N0.
        return dict(e01=float(NOMINAL_ASSUMPTION_N0["e01"]), e10=float(NOMINAL_ASSUMPTION_N0["e10"]),
                    t2=float(NOMINAL_ASSUMPTION_N0["t2"]), tau=float(NOMINAL_ASSUMPTION_N0["tau"]),
                    omega=float(NOMINAL_ASSUMPTION_N0["omega"]),
                    contrast_power=float(NOMINAL_ASSUMPTION_N0["contrast_power"]),
                    contrast_override=NOMINAL_ASSUMPTION_N0["contrast_override"],
                    dither_delta=float(NOMINAL_ASSUMPTION_N0["dither_delta"]))

    def assumed_params(self):
        """M2 F1: the parameter dict the filter ACTUALLY uses (recorded into every
        ctrl:bayes:* result JSON). nominal -> exactly N0 in every cell; oracle -> the
        cell's true parameters (batch means for per-episode arrays)."""
        if self.assumed == "oracle" and self.env is not None:
            e = self.env
            c = getattr(e, "cfg", None) or self.cfg
            sigma, theta = self._oracle_drift_params()
            kind = str(c.get("drift", "gaussian"))
            return {"drift": ("ou" if kind == "ou" else "random_walk"),
                    "theta": float(theta), "sigma": float(sigma),
                    "tau": float(c.get("tau", 50e-6) or 50e-6),
                    "t2": float(np.mean(e.ep_t2)),
                    "contrast_power": float(c.get("contrast_power", 1.0) or 1.0),
                    "contrast_override": c.get("contrast_override", None),
                    "e01": float(np.mean(e.ep_e01)), "e10": float(np.mean(e.ep_e10)),
                    "omega": float(c.get("omega", 5000.0)),
                    "dither_delta": float(c.get("dither_delta", 2500.0))}
        return dict(NOMINAL_ASSUMPTION_N0)

    # ---- Part-2 S15.1c: hybrid likelihood support -------------------------------
    def _hybrid_noise(self):
        """(kind, sigma_a) when the env runs a gauss-noise HybridEngine, else None.
        For gauss noise the likelihood is Gaussian(a | lineshape(grid - fc), sigma_a^2)
        with the same sigma_a as the engine (the engine's noise_floor)."""
        eng = getattr(self.env, "engine", None) if self.env is not None else None
        if eng is None or not hasattr(eng, "noise"):
            return None
        if getattr(eng, "noise", "") == "gauss" and hasattr(eng, "lineshape"):
            return eng.lineshape, float(getattr(eng, "noise_floor", 0.02))
        return None

    def _gauss_loglik(self, amp, p_grid, sigma_a):
        """Gaussian likelihood of a continuous amplitude reading given probe probabilities on the
        grid: log N(amp | p_grid, sigma_a^2). amp: (B,), p_grid: (B, G)."""
        amp = np.asarray(amp, dtype=float)[:, None]
        return -0.5 * ((amp - p_grid) / sigma_a) ** 2 - np.log(sigma_a) - 0.5 * np.log(2 * np.pi)

    def _loglik(self, Dg, counts, shots, p):
        qp = self.qp
        hn = self._hybrid_noise()
        if hn is not None and not np.issubdtype(np.asarray(counts).dtype, np.integer):
            # gauss-noise hybrid engine: counts/shots is the continuous amplitude
            amp = np.asarray(counts, dtype=float) / float(shots)
            return self._gauss_loglik(amp, p, hn[1])
        return qp.binom_loglik(counts[:, None], int(shots), np.clip(p, 1e-6, 1 - 1e-6))

    def _probe_p(self, Dg, ap):
        """Probe probabilities on the grid using the ASSUMED parameters; uses the hybrid engine's
        lineshape when the env runs a gauss-noise HybridEngine (S15.1c)."""
        qp = self.qp
        hn = self._hybrid_noise()
        if hn is not None and hn[0] == "lorentz":
            from hybrid_engines import lorentzian_p
            base = lorentzian_p(Dg)
            return base, None
        if hn is not None and hn[0] == "rabi":
            # gauss noise on the exact-Rabi lineshape: no readout error in the model
            return qp.rabi_p(Dg, omega=ap["omega"]), None
        return None, None

    def step_filter(self, counts, shots, fc, probe):
        qp = self.qp
        ap = self._assumed_probe_params()
        self.post = self.gb.predict(self.post)
        Dg = self.gb.g[None, :] - fc[:, None]
        hn = self._hybrid_noise()
        if hn is not None:
            # S15.1c: gauss-noise hybrid engine — Gaussian likelihood on the continuous
            # amplitude with the engine's lineshape (no readout error in the model).
            from hybrid_engines import lorentzian_p
            if hn[0] == "lorentz":
                base = lambda D: lorentzian_p(D)
            else:
                base = lambda D: qp.rabi_p(D, omega=ap["omega"])
            if probe == "rabi":
                n = int(shots[0]) if hasattr(shots, "__len__") else int(shots)
                ll = self._loglik(Dg, counts, n, base(Dg))
            elif probe == "dither":
                n = (np.asarray(shots) // 2).astype(int)
                dd = ap["dither_delta"]
                ll = (self._loglik(Dg, counts[:, 0], n[0], base(Dg - dd)) +
                      self._loglik(Dg, counts[:, 1], n[0], base(Dg + dd)))
            else:
                raise ValueError("hybrid gauss likelihood supports rabi/dither only, got %r" % probe)
            self.post = self.gb.update(self.post, ll)
            mu, sd = self.gb.mean_std(self.post, window=self.window)
            self.last_mu, self.last_std = mu, sd
            return mu, sd
        if probe == "rabi" or probe == "rabi_legacy":
            n = int(shots[0]) if hasattr(shots, "__len__") else int(shots)
            p = qp.readout(qp.rabi_p(Dg, omega=ap["omega"]), ap["e01"], ap["e10"])
            ll = self._loglik(Dg, counts, n, p)
        elif probe == "dither":
            n = (np.asarray(shots) // 2).astype(int)
            dd = ap["dither_delta"]
            pp = qp.readout(qp.rabi_p(Dg - dd, omega=ap["omega"]), ap["e01"], ap["e10"])
            pm = qp.readout(qp.rabi_p(Dg + dd, omega=ap["omega"]), ap["e01"], ap["e10"])
            ll = (self._loglik(Dg, counts[:, 0], n[0], pp) +
                  self._loglik(Dg, counts[:, 1], n[0], pm))
        elif probe == "ramsey_iq":
            n = (np.asarray(shots) // 2).astype(int)
            if ap.get("contrast_power", 1.0) != 1.0 or ap.get("contrast_override") is not None:
                # M2 F1: oracle uses the TRUE contrast shape/override (S16 platform cells)
                from qphysics_ext import ramsey_p_ext
                pw = float(ap.get("contrast_power", 1.0))
                ov = ap.get("contrast_override")
                pI = qp.readout(ramsey_p_ext(Dg, ap["tau"], ap["t2"], "I", pw, ov), ap["e01"], ap["e10"])
                pQ = qp.readout(ramsey_p_ext(Dg, ap["tau"], ap["t2"], "Q", pw, ov), ap["e01"], ap["e10"])
            else:
                pI = qp.readout(qp.ramsey_p(Dg, ap["tau"], ap["t2"], "I"), ap["e01"], ap["e10"])
                pQ = qp.readout(qp.ramsey_p(Dg, ap["tau"], ap["t2"], "Q"), ap["e01"], ap["e10"])
            ll = (self._loglik(Dg, counts[:, 0], n[0], pI) +
                  self._loglik(Dg, counts[:, 1], n[0], pQ))
        else:
            raise ValueError(probe)
        self.post = self.gb.update(self.post, ll)
        mu, sd = self.gb.mean_std(self.post, window=self.window)
        self.last_mu, self.last_std = mu, sd
        return mu, sd

    def act(self, obs_or_state):
        st = self.env.public_state()
        mu, _ = self.step_filter(st["counts"], st["shots"], st["fc"], st["probe"])
        a = (mu - st["fc"]) / self._nominal("action_scale", 3000.0)
        return np.clip(a, -1.0, 1.0)

    def step_env_filter(self):
        """Called by BatchEnv after each measurement when filter_features=True."""
        st = self.env.public_state()
        self.step_filter(st["counts"], st["shots"], st["fc"], st["probe"])
