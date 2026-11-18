"""qstudy fallback — Part-2 S17.5: fallback controller wrapper with FIXED thresholds
(exploratory; thresholds fixed here, never tuned):

  enter/hold active:  posterior sigma_f > 700 Hz OR the CUSUM alarm is active;
  while active:       action scale x0.5; (adaptive env only) force mode ramsey_iq
                      with 1024 shots;
  exit:               sigma_f < 350 Hz for 3 consecutive steps.

Compare on and off in the regime env and the nominal env; report failure severity
(CVaR95, p95 of per-episode rms) versus average performance (lock_w1000 mean).
"""
import numpy as np

FALLBACK_SIGMA_HI = 700.0    # Hz, enter threshold on the filter's posterior sigma_f
FALLBACK_SIGMA_LO = 350.0    # Hz, exit threshold
FALLBACK_EXIT_STEPS = 3      # consecutive steps below the exit threshold
FALLBACK_SCALE = 0.5         # action scale while active
ADAPTIVE_FORCE_MODE = "ramsey_iq"
ADAPTIVE_FORCE_SHOTS = 1024


class FallbackController:
    """Wraps a batched controller (bayes) or an action-producing callable (RL policy);
    sigma_provider() -> (B,) posterior sigma_f of the latest filter update;
    alarm_provider() -> (B,) bool CUSUM alarm flags (None -> no alarms)."""

    def __init__(self, inner, sigma_provider, alarm_provider=None, B=1):
        self.inner = inner
        self.sigma_provider = sigma_provider
        self.alarm_provider = alarm_provider
        self.B = int(B)
        self.active = np.zeros(self.B, bool)
        self.below_run = np.zeros(self.B, int)

    def _update_state(self):
        sigma = np.asarray(self.sigma_provider(), dtype=float)
        alarm = (np.zeros(self.B, bool) if self.alarm_provider is None
                 else np.asarray(self.alarm_provider(), dtype=bool))
        below = sigma < FALLBACK_SIGMA_LO
        self.below_run = np.where(below, self.below_run + 1, 0)
        trigger = (sigma > FALLBACK_SIGMA_HI) | alarm
        active = self.active | trigger
        self.active = np.where(self.below_run >= FALLBACK_EXIT_STEPS, False, active)
        return self.active.copy()

    def act(self, obs_or_state=None):
        a = np.asarray(self.inner.act(obs_or_state), dtype=float).reshape(-1)
        active = self._update_state()
        return np.where(active, a * FALLBACK_SCALE, a)


class AdaptiveFallback:
    """Fallback for the S17.1 AdaptiveEnv action layout (B, 3): a0 control (scaled x0.5
    while active), a1 mode -> ramsey_iq bin, a2 shots -> 1024 bin while active.
    inner must expose act(obs) -> (B,3) or (3,) actions (an RL policy wrapper)."""

    def __init__(self, inner, sigma_provider, alarm_provider=None, B=1):
        self.inner = inner
        self.sigma_provider = sigma_provider
        self.alarm_provider = alarm_provider
        self.B = int(B)
        self.active = np.zeros(self.B, bool)
        self.below_run = np.zeros(self.B, int)

    def _update_state(self):
        sigma = np.asarray(self.sigma_provider(), dtype=float)
        alarm = (np.zeros(self.B, bool) if self.alarm_provider is None
                 else np.asarray(self.alarm_provider(), dtype=bool))
        below = sigma < FALLBACK_SIGMA_LO
        self.below_run = np.where(below, self.below_run + 1, 0)
        trigger = (sigma > FALLBACK_SIGMA_HI) | alarm
        active = self.active | trigger
        self.active = np.where(self.below_run >= FALLBACK_EXIT_STEPS, False, active)
        return self.active.copy()

    def act(self, obs=None):
        a = np.asarray(self.inner.act(obs), dtype=float).reshape(-1)
        if a.size == 3:
            a = np.tile(a, self.B)
        assert a.size == 3 * self.B, "AdaptiveFallback expects (B,3) or (3,) actions"
        active = self._update_state()
        a0 = np.where(active, a[0::3] * FALLBACK_SCALE, a[0::3])
        # ramsey_iq bin: a1 >= 1/3 ; 1024-shots bin: a2 >= 1/3
        a1 = np.where(active, 1.0, a[1::3])
        a2 = np.where(active, 1.0, a[2::3])
        out = np.empty(3 * self.B)
        out[0::3], out[1::3], out[2::3] = a0, a1, a2
        return out
