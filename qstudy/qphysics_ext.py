"""Extensions for the platform study. Does NOT modify qphysics.py; defaults reproduce qphysics exactly."""
import numpy as np
from qphysics import F_NOM, CLIP, Drift, ramsey_p, readout

TAU_NOM = 50e-6   # simulation Ramsey time; all platform parameters are mapped to this by dimensionless products (delta*tau)

# ------------------------------------------------------------------ Ramsey contrast with shape and override
def contrast_ext(tau, t2, power=1.0, override=None):
    """C = exp(-(tau/T2)^power); power=1 exponential, power=2 Gaussian (quasi-static noise). override: fixed visibility (photonic)."""
    if override is not None:
        return np.asarray(override, dtype=float) * np.ones(np.broadcast(np.asarray(tau), np.asarray(t2 if t2 is not None else 1.0)).shape)
    t2a = np.inf if t2 is None else np.asarray(t2, dtype=float)
    return np.exp(-(tau / t2a) ** power)

def ramsey_p_ext(delta, tau=50e-6, t2=200e-6, quad="I", power=1.0, override=None):
    c = contrast_ext(tau, t2, power, override)
    phi = 2 * np.pi * np.asarray(delta, dtype=float) * tau
    return 0.5 * (1.0 + c * (np.cos(phi) if quad == "I" else np.sin(phi)))

def dephasing_flip_prob(c):
    """Aer: a Z-flip channel with probability p=(1-C)/2 on the wait gives Ramsey contrast C for ANY decay shape at fixed tau."""
    return (1.0 - np.asarray(c, dtype=float)) / 2.0

# ------------------------------------------------------------------ drift with explicit parameters
class DriftExt(Drift):
    """Like Drift but with: telegraph flip probability, sinusoid period range (steps), onef corner coefficients and total std,
    and the S17.2 'regime' kind (hidden slow/fast Markov regimes + Poisson jump events)."""
    def __init__(self, kind, rng, sigma=500.0, theta=0.05, amp=3000.0, flip=0.02, period_range=(60, 120), theta_list=None, std_total=None,
                 regime_slow=250.0, regime_fast=1000.0, regime_switch=0.02,
                 jump_prob=0.01, jump_lo=2000.0, jump_hi=8000.0):
        super().__init__(kind, rng, sigma, theta, amp)
        self.flip, self.period_range = float(flip), period_range
        self.theta_list = None if theta_list is None else np.asarray(theta_list, float)
        self.std_total = std_total
        # S17.2 regime kind (defaults = the pre-registered training family; the held-out
        # family is the same kind with jump_prob=0.02, jump 8000-12000 Hz via drift_params)
        self.regime_slow, self.regime_fast = float(regime_slow), float(regime_fast)
        self.regime_switch = float(regime_switch)
        self.jump_prob, self.jump_lo, self.jump_hi = float(jump_prob), float(jump_lo), float(jump_hi)
        self.jump_log = []

    def reset(self, init_offset):
        if self.kind == "regime":
            self.t, self.x = 0, float(init_offset)
            self.regime = 0            # start in the slow regime (documented)
            self.jump_log = []
            return self.x
        if self.kind == "onef" and self.theta_list is not None:
            r = self.rng
            self.t, self.x = 0, float(init_offset)
            K = len(self.theta_list)
            v = (self.std_total ** 2) / K
            self.THETA = self.theta_list
            self.s = np.sqrt(v * (2 * self.theta_list - self.theta_list ** 2))
            self.comp = r.normal(0.0, np.sqrt(v), K)
            self.comp += (self.x - self.comp.sum()) / K
            return self.x
        x = super().reset(init_offset)
        # only re-draw the sinusoid period when a NON-default period_range is requested;
        # at the default (60,120) super().reset has already drawn it, so the RNG stream
        # stays IDENTICAL to the base Drift at defaults (S16.2 requirement).
        if self.kind == "sinusoid" and tuple(self.period_range) != (60.0, 120.0):
            self.period = self.rng.uniform(*self.period_range)
        return x

    def step(self):
        r, k = self.rng, self.kind
        if k == "regime":
            # S17.2: hidden slow/fast Markov regimes + Poisson jump events.
            # RNG draw order (fixed): regime-switch Bernoulli, gaussian increment,
            # jump Bernoulli, jump size uniform, jump sign Bernoulli.
            if r.random() < self.regime_switch:
                self.regime = 1 - self.regime
            sig = self.regime_slow if self.regime == 0 else self.regime_fast
            self.x += sig * r.normal()
            self.t += 1
            if r.random() < self.jump_prob:
                size = r.uniform(self.jump_lo, self.jump_hi) * (1.0 if r.random() < 0.5 else -1.0)
                self.x = float(np.clip(self.x + size, -CLIP, CLIP))
                self.jump_log.append((self.t, float(size)))
            else:
                self.x = float(np.clip(self.x, -CLIP, CLIP))
            return self.x
        if k == "onef" and self.theta_list is not None:
            self.comp += -self.THETA * self.comp + self.s * r.normal(size=self.comp.shape)
            self.x = float(np.clip(self.comp.sum(), -CLIP, CLIP))
            return self.x
        if k == "telegraph":
            if r.random() < self.flip:
                self.state = -self.state
            self.y += -0.2 * self.y + 0.5 * self.sigma * r.normal()
            self.x = float(np.clip(self.base + self.amp * self.state * 0.5 + self.y, -CLIP, CLIP))
            return self.x
        return super().step()

# ------------------------------------------------------------------ platform profiles (physical -> simulation units)
# tags: G = grounded in published ranges (order of magnitude), A = assumed modelling choice, S = speculative (no demonstrated qubit)
PROFILES = {
 "sc_transmon":   dict(label="Superconducting transmon (fixed-frequency, TLS-dominated)", tags="G/A", tau=20e-6, t2=60e-6, power=1.0, override=None, e01=0.01, e10=0.04,
                        shot_time=300e-6, default_shots=1024, drift="telegraph", D=300.0, jump_hz=8000.0, jump_rate=1/600.0),
 "sc_flux":       dict(label="Superconducting tunable transmon (1/f flux noise)", tags="G/A", tau=10e-6, t2=20e-6, power=2.0, override=None, e01=0.01, e10=0.04,
                        shot_time=200e-6, default_shots=1024, drift="onef", std_total=11000.0, corner_s=(10.0, 100.0, 1000.0, 10000.0)),
 "si_spin":       dict(label="Electron spin qubit in silicon quantum dot (charge/nuclear noise)", tags="G/A", tau=3e-6, t2=20e-6, power=2.0, override=None, e01=0.03, e10=0.08,
                        shot_time=400e-6, default_shots=1024, drift="onef", std_total=11000.0, corner_s=(1.0, 10.0, 100.0, 1000.0, 10000.0)),
 "ion_hyperfine": dict(label="Trapped-ion hyperfine qubit (magnetic-field noise)", tags="G/A", tau=3e-3, t2=30e-3, power=2.0, override=None, e01=0.002, e10=0.002,
                        shot_time=5e-3, default_shots=256, drift="sinusoid", D=0.1, line_hz=3.0, line_period_s=600.0),
 "atom_tweezer":  dict(label="Neutral-atom tweezer qubit (slow light-shift/field drift)", tags="A", tau=500e-6, t2=5e-3, power=2.0, override=None, e01=0.005, e10=0.03,
                        shot_time=100e-3, default_shots=64, drift="gaussian", D=1.0),
 "photonic_mzi":  dict(label="Photonic interferometer phase lock (visibility-limited)", tags="G/A", tau=None, t2=None, power=1.0, override=0.97, e01=0.01, e10=0.01,
                        shot_time=10e-6, default_shots=1024, drift="gaussian", D_phase=0.01),
 "majorana_hyp":  dict(label="HYPOTHETICAL Majorana-class parity qubit (no demonstrated qubit; speculative)", tags="S", tau=1e-3, t2=1.0, power=1.0, override=None, e01=0.05, e10=0.05,
                        shot_time=10e-3, default_shots=256, drift="onef", std_total=50.0, corner_s=(10.0, 100.0, 1000.0, 10000.0)),
}

def profile_to_cfg(name, shots=None, drift_mult=1.0):
    """Physical profile -> BatchEnv cfg overrides in simulation units (tau fixed at TAU_NOM). Only products delta*tau and tau/T2 matter.
    Returns (cfg_overrides, info) ; info holds physical-time bookkeeping and normalised numbers."""
    p = PROFILES[name]
    N = int(shots or p["default_shots"])
    T_step = N * p["shot_time"]                                     # physical seconds per control step
    tau_p = p["tau"] if p["tau"] is not None else TAU_NOM
    k = tau_p / TAU_NOM                                             # sim_Hz = phys_Hz * k
    cfg = dict(shots=N, e01=p["e01"], e10=p["e10"], tau=TAU_NOM, contrast_power=p["power"], contrast_override=p["override"], profile=name)
    cfg["t2"] = None if p["t2"] is None else p["t2"] / k            # keeps tau/T2 identical
    kind = p["drift"]
    dp = {}
    if "D_phase" in p:                                              # photonic: rad^2/s -> sim Hz via phase = 2*pi*delta*TAU_NOM
        sigma_sim = np.sqrt(drift_mult * p["D_phase"] * T_step) / (2 * np.pi * TAU_NOM)
    elif "D" in p:
        sigma_sim = np.sqrt(drift_mult * p["D"] * T_step) * k
    else:
        sigma_sim = 0.0
    if kind == "telegraph":
        dp = dict(flip=1 - np.exp(-drift_mult * p["jump_rate"] * T_step), amp=p["jump_hz"] * k)
    elif kind == "sinusoid":
        period_steps = max(4.0, p["line_period_s"] / T_step)
        dp = dict(amp=p["line_hz"] * k * np.sqrt(drift_mult), period_range=(0.8 * period_steps, 1.2 * period_steps))
    elif kind == "onef":
        th = [1 - np.exp(-T_step / c) for c in p["corner_s"]]
        dp = dict(theta_list=th, std_total=p["std_total"] * k * np.sqrt(drift_mult))
        sigma_sim = max(sigma_sim, 1e-6)
    cfg.update(drift=kind, drift_sigma=float(sigma_sim), drift_params=dp)
    info = dict(T_step_s=T_step, k=k, sigma_n=float(sigma_sim * TAU_NOM), c=float(contrast_ext(TAU_NOM, cfg["t2"], p["power"], p["override"])),
                tau_over_t2=(None if p["t2"] is None else tau_p / p["t2"]))
    return cfg, info
