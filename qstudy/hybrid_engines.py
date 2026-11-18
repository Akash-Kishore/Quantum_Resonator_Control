"""qstudy hybrid_engines — Part-2 S15.1 shared extensions.
HybridEngine(cfg): cfg lineshape in {"lorentz","rabi"} and noise in {"gauss","binom"}.
Measurement value m = lineshape(delta) + noise (S15.1a):
  - lorentz + gauss  = the CLASSICAL engine: replicates the amplitude model and noise of the
    classical GAUSSIAN resonator module (simulation\\resonator_model_gaussian.py:10-15:
    amplitude Lorentzian A = 1/(1+Q^2*((f-f0)/f0)^2) with Q=50, f0=F_NOM, i.e. standard-form
    half-width F_NOM/Q = 10000 Hz; additive Gaussian noise with the legacy noise floor
    noise_floor=0.02, resonator_model_gaussian.py:5; the legacy clip is [0,1.2]
    (resonator_model_gaussian.py:29) but Part-2 S15.1 specifies clip [0,1] — the Part-2
    clip governs here).
  - lorentz + binom: p = Lorentzian lineshape, observation = binomial(shots,p)/shots with
    readout error (e01/e10 as passed by the env).
  - rabi + gauss: p = exact Rabi lineshape (qphysics.rabi_p), observation = p + the legacy
    Gaussian noise (sigma = noise_floor).
  - rabi + binom  = the quantum surrogate (identical to SurrogateEngine for kinds rabi/
    ramsey_I/ramsey_Q with readout error applied last).
Supported probes: rabi and dither (two measurements at +-dither_delta; the env splits shots
in halves exactly as for the quantum probes). Observation layout identical to the quantum
rabi/dither layouts (so one policy runs in all four environments).
The counts() interface matches SurrogateEngine; for gauss noise the returned "counts" are
float m*shots so that counts/shots reproduces the continuous amplitude m exactly."""
import numpy as np

import qphysics as qp

F_NOM = qp.F_NOM
LEGACY_Q = 50.0            # simulation\resonator_model_gaussian.py:5 (q_factor=50)
LEGACY_NOISE_FLOOR = 0.02  # simulation\resonator_model_gaussian.py:5 (noise_floor=0.02)
LINESHAPES = ("lorentz", "rabi")
NOISES = ("gauss", "binom")


def lorentzian_p(delta, q=LEGACY_Q, f0=F_NOM):
    """Legacy classical amplitude: 1/(1 + q^2*((f-f0)/f0)^2) (resonator_model_gaussian.py:14-15).
    Standard Lorentzian form 1/(1+(delta/H)^2) with H = f0/q = 10000 Hz."""
    d = np.asarray(delta, dtype=float)
    return 1.0 / (1.0 + q ** 2 * (d / f0) ** 2)


class HybridEngine:
    """Part-2 S15.1a hybrid engine. cfg keys: lineshape, noise (both required),
    noise_floor (default legacy 0.02, used by gauss noise)."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.lineshape = str(cfg.get("lineshape", "rabi"))
        self.noise = str(cfg.get("noise", "binom"))
        self.noise_floor = float(cfg.get("noise_floor", LEGACY_NOISE_FLOOR))
        if self.lineshape not in LINESHAPES:
            raise ValueError("lineshape must be in %r, got %r" % (LINESHAPES, self.lineshape))
        if self.noise not in NOISES:
            raise ValueError("noise must be in %r, got %r" % (NOISES, self.noise))
        if self.noise == "binom" and self.lineshape == "rabi":
            self._surrogate = SurrogateEngine()

    def _lineshape_p(self, kind, delta, omega):
        if kind == "rabi_legacy":
            raise ValueError("rabi_legacy is not a hybrid-engine probe kind")
        if kind == "rabi":
            if self.lineshape == "lorentz":
                return lorentzian_p(delta)
            return qp.rabi_p(delta, omega=omega)
        raise ValueError("hybrid engine supports the rabi probe only (dither is two rabi calls)")

    def _one(self, kind, delta, shots, e01, e10, rngs, omega):
        """One measurement at each delta; returns counts (float for gauss, int for binom)."""
        delta = np.asarray(delta, dtype=float)
        shots = np.asarray(shots, dtype=np.int64)
        B = delta.shape[0]
        p_true = self._lineshape_p(kind, delta, omega)
        if self.noise == "binom":
            if self.lineshape == "rabi":
                return self._surrogate.counts(kind, delta, shots, np.full(B, np.inf),
                                              np.asarray(e01), np.asarray(e10), rngs, omega=omega)
            p = qp.readout(p_true, e01, e10)  # readout error applied last
            out = np.empty(B, dtype=np.int64)
            for b in range(B):
                rng = rngs[b] if not isinstance(rngs, np.random.Generator) else rngs
                out[b] = rng.binomial(int(shots[b]), float(np.clip(p[b], 0.0, 1.0)))
            return out
        # gauss: m = lineshape(delta) + N(0, noise_floor), clip [0,1]
        out = np.empty(B, dtype=float)
        for b in range(B):
            rng = rngs[b] if not isinstance(rngs, np.random.Generator) else rngs
            m = p_true[b] + rng.normal(0.0, self.noise_floor)
            out[b] = np.clip(m, 0.0, 1.0) * shots[b]
        return out

    def counts(self, kind, delta, shots, t2, e01, e10, rngs, omega=5000.0, tau=50e-6, depol_p=0.0):
        """Same interface as SurrogateEngine. Dither probes arrive as two calls (delta -+ dither)
        from BatchEnv, exactly as for the quantum dither probe; kind is 'rabi' in both calls.
        t2 is ignored for lorentz lineshapes (classical resonators have no Ramsey contrast)."""
        if kind in ("ramsey_I", "ramsey_Q"):
            raise ValueError("HybridEngine supports probe rabi/dither only (S15.1)")
        return self._one(kind, delta, shots, e01, e10, rngs, omega)


# imported at the bottom to avoid a cycle at module import time
from engines import SurrogateEngine  # noqa: E402
