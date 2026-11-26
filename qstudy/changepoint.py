"""qstudy changepoint — Part-2 S17.2: regime switching + CUSUM change-point detection.

Pre-registered, FIXED thresholds (S17.2; PREREG2: "thresholds in this file are never
changed after freezing"):
  - CUSUM on the standardised innovation z_t = (observed - predicted)/sqrt(predicted
    variance) from the filter's posterior predictive (total count over the probe's
    circuits; predictive variance = binomial + parameter/mixture variance over the grid);
  - S_t = max(0, S_{t-1} + |z_t| - 0.5); alarm when S_t > 5;
  - on alarm: inflate the filter's drift sigma by 10 for 5 steps and reset S;
  - detection window 10 steps; recovery = 3 consecutive |e| <= 1000 Hz (primary width).

The regime drift KIND lives in qphysics_ext.DriftExt ("regime": slow sigma 250 /
fast sigma 1000 hidden Markov regimes, switch probability 0.02 per step, plus jump
events: probability 0.01 per step, size uniform 2000-8000 Hz, random sign, applied
instantly; held-out family = same kind with jump_prob 0.02 and size 8000-12000 Hz
via drift_params).

Everything here is exploratory except the CP-1 contrast; the false-alarm rate is
always reported next to the detection rate (shot noise is reported honestly).
"""
import numpy as np

import qphysics as qp
from controllers import BayesTracker

CUSUM_K = 0.5             # drift allowance per step
CUSUM_THRESH = 5.0        # alarm threshold on S_t
INFLATE_FACTOR = 10.0     # filter drift sigma inflation on alarm
INFLATE_STEPS = 5         # ... for 5 steps
DETECT_WINDOW = 10        # detection within 10 steps of the jump
RECOVERY_CONSEC = 3       # recovery = 3 consecutive |e| <= RECOVERY_W
RECOVERY_W = 1000.0       # primary lock width (Hz)

REGIME_DEFAULTS = dict(regime_slow=250.0, regime_fast=1000.0, regime_switch=0.02,
                       jump_prob=0.01, jump_lo=2000.0, jump_hi=8000.0)
REGIME_HELDOUT = dict(REGIME_DEFAULTS, jump_prob=0.02, jump_lo=8000.0, jump_hi=12000.0)


class CusumDetector:
    """Batch CUSUM state (S_t, alarm flag, inflation countdown), one slot per episode."""

    def __init__(self, B):
        self.B = int(B)
        self.S = np.zeros(self.B)
        self.alarm_now = np.zeros(self.B, bool)
        self.inflate_left = np.zeros(self.B, int)

    def step(self, z):
        """One CUSUM step. z: (B,) standardised innovations; returns the (B,) alarm flags."""
        z = np.asarray(z, dtype=float)
        self.S = np.maximum(0.0, self.S + np.abs(z) - CUSUM_K)
        self.alarm_now = self.S > CUSUM_THRESH
        self.S = np.where(self.alarm_now, 0.0, self.S)
        self.inflate_left = np.where(self.alarm_now, INFLATE_STEPS,
                                     np.maximum(self.inflate_left - 1, 0))
        return self.alarm_now.copy()


def predictive_z(post, g, fc, counts, shots, probe, ap):
    """Standardised posterior-predictive innovation z_t, given the PREDICTED posterior
    (after the predict step, BEFORE the measurement update).

    Per probe circuit i with per-circuit shots n_i and model probability p_i(x) on the
    grid: predicted count = n_i * E[p_i]; predictive variance =
    n_i*(E[p_i] - E[p_i^2]) + n_i^2*(E[p_i^2] - E[p_i]^2)  (binomial + mixture spread),
    E[.] over the predicted posterior. The per-circuit innovations
    z_i = (obs_i - pred_i)/sqrt(var_i) are combined by their EUCLIDEAN NORM
    (components independent given the true offset; a plain sum would be blind to
    phase-like jumps that conserve the total count). ap = assumed (nominal) parameters."""
    post = np.asarray(post, dtype=float)
    g = np.asarray(g, dtype=float)
    B = post.shape[0]
    D = g[None, :] - np.asarray(fc, dtype=float)[:, None]
    sh = np.broadcast_to(np.asarray(shots, dtype=float), (B,))
    cnt = np.asarray(counts, dtype=float)
    if cnt.ndim == 1:
        cnt = cnt[:, None]

    # accumulate per-circuit innovations separately, then take the norm
    zis = []
    if probe in ("rabi", "rabi_legacy"):
        p = qp.readout(qp.rabi_p(D, omega=ap["omega"]), ap["e01"], ap["e10"])
        Ep = (post * p).sum(axis=1)
        Epp = (post * p * p).sum(axis=1)
        pred = sh * Ep
        var = sh * (Ep - Epp) + sh * sh * (Epp - Ep * Ep)
        zis.append((cnt[:, 0] - pred) / np.sqrt(np.clip(var, 1e-9, None)))
    elif probe == "dither":
        m = sh / 2.0
        dd = ap["dither_delta"]
        for col, Dd in ((0, D - dd), (1, D + dd)):
            p = qp.readout(qp.rabi_p(Dd, omega=ap["omega"]), ap["e01"], ap["e10"])
            Ep = (post * p).sum(axis=1)
            Epp = (post * p * p).sum(axis=1)
            pred = m * Ep
            var = m * (Ep - Epp) + m * m * (Epp - Ep * Ep)
            zis.append((cnt[:, col] - pred) / np.sqrt(np.clip(var, 1e-9, None)))
    elif probe == "ramsey_iq":
        m = sh / 2.0
        for col, quad in ((0, "I"), (1, "Q")):
            p = qp.readout(qp.ramsey_p(D, ap["tau"], ap["t2"], quad), ap["e01"], ap["e10"])
            Ep = (post * p).sum(axis=1)
            Epp = (post * p * p).sum(axis=1)
            pred = m * Ep
            var = m * (Ep - Epp) + m * m * (Epp - Ep * Ep)
            zis.append((cnt[:, col] - pred) / np.sqrt(np.clip(var, 1e-9, None)))
    else:
        raise ValueError("predictive_z: unsupported probe %r" % probe)
    return np.sqrt(np.sum(np.stack(zis, axis=0) ** 2, axis=0))


class BayesCusum(BayesTracker):
    """BayesTracker + CUSUM detector (S17.2). The likelihood/update code is identical
    to BayesTracker.step_filter; the predict step additionally uses an inflated-sigma
    transition matrix for episodes inside the 5-step post-alarm window, and the CUSUM
    consumes the posterior-predictive innovation of each new measurement.

    Continuous-amplitude (gauss-noise hybrid) engines have no count posterior predictive:
    the detector is idle there (z = 0, documented). last_alarm/last_z hold the most
    recent per-episode values for drivers and the S17.5 fallback wrapper."""

    def _init_filter(self):
        super()._init_filter()
        self.gb_inf = self.qp.GridBayes(half_range=30000.0, n=601,
                                        sigma=self.f_sigma * INFLATE_FACTOR, theta=self.f_theta)
        self.det = CusumDetector(self.B)
        self.last_alarm = np.zeros(self.B, bool)
        self.last_z = np.zeros(self.B)

    def step_filter(self, counts, shots, fc, probe):
        qp = self.qp
        ap = self._assumed_probe_params()
        # (1) predict, with sigma inflation where a CUSUM alarm is active
        if np.any(self.det.inflate_left > 0):
            pn = self.gb.predict(self.post)
            pi = self.gb_inf.predict(self.post)
            self.post = np.where((self.det.inflate_left > 0)[:, None], pi, pn)
        else:
            self.post = self.gb.predict(self.post)
        # (2) posterior-predictive innovation BEFORE the update
        if np.issubdtype(np.asarray(counts).dtype, np.integer):
            z = predictive_z(self.post, self.gb.g, fc, counts, shots, probe, ap)
        else:
            z = np.zeros(self.B)  # continuous-amplitude hybrid engine: CUSUM idle (documented)
        # (3) likelihood + update (identical to BayesTracker.step_filter)
        Dg = self.gb.g[None, :] - fc[:, None]
        hn = self._hybrid_noise()
        if hn is not None:
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
            self.last_alarm = self.det.step(z)
            self.last_z = z
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
            pI = qp.readout(qp.ramsey_p(Dg, ap["tau"], ap["t2"], "I"), ap["e01"], ap["e10"])
            pQ = qp.readout(qp.ramsey_p(Dg, ap["tau"], ap["t2"], "Q"), ap["e01"], ap["e10"])
            ll = (self._loglik(Dg, counts[:, 0], n[0], pI) +
                  self._loglik(Dg, counts[:, 1], n[0], pQ))
        else:
            raise ValueError(probe)
        self.post = self.gb.update(self.post, ll)
        mu, sd = self.gb.mean_std(self.post, window=self.window)
        self.last_mu, self.last_std = mu, sd
        # (4) CUSUM update
        self.last_alarm = self.det.step(z)
        self.last_z = z
        return mu, sd


def jump_metrics(e_tr, alarm_tr, shots_tr, jump_events, burn_in=20,
                detect_window=DETECT_WINDOW, W=RECOVERY_W, consec=RECOVERY_CONSEC):
    """Metrics per jump + false-alarm rate (S17.2).

    e_tr: (T, B) errors at probe time; alarm_tr: (T, B) bool; shots_tr: (T, B);
    jump_events: per-episode list of (t_jump, size) with t_jump the 0-based step index
    of the jump (DriftExt "regime" logs this). Burn-in steps are excluded from the
    false-alarm count (the filter warms up; documented).
    A false alarm = alarm at step t with no jump in the previous `detect_window` steps.
    Recovery time = steps from the jump until 3 consecutive |e| <= W complete.
    max|e| and measurement cost are taken over (t_jump, recovery], capped at the next
    jump. Returns a dict of aggregates plus the per-jump rows."""
    e_tr = np.asarray(e_tr, dtype=float)
    alarm_tr = np.asarray(alarm_tr, dtype=bool)
    shots_tr = np.asarray(shots_tr, dtype=float)
    T, B = e_tr.shape
    rows = []
    n_false = 0
    n_steps = 0
    for b in range(B):
        jumps = sorted(jump_events[b])
        for t in range(burn_in, T):
            if alarm_tr[t, b] and not any(0 <= t - tj <= detect_window for tj, _ in jumps):
                n_false += 1
            n_steps += 1
        for j, (tj, size) in enumerate(jumps):
            nxt = jumps[j + 1][0] if j + 1 < len(jumps) else T
            delay = None
            for t in range(tj, min(tj + detect_window + 1, T)):
                if alarm_tr[t, b]:
                    delay = t - tj   # 0 = alarm on the first post-jump measurement
                    break
            rec = None
            run = 0
            trec = None
            for t in range(tj + 1, T):
                if abs(e_tr[t, b]) <= W:
                    run += 1
                    if run >= consec:
                        trec = t
                        break
                else:
                    run = 0
            if trec is not None:
                rec = trec - tj
            end = min(trec if trec is not None else T - 1, nxt - 1)
            mx = float(np.max(np.abs(e_tr[tj + 1:end + 1, b]))) if end > tj else 0.0
            cost = float(np.sum(shots_tr[tj + 1:end + 1, b])) if end > tj else 0.0
            rows.append(dict(episode=b, t_jump=tj, size=size,
                             detected=delay is not None, delay=delay,
                             recovery=rec, max_abs_after=mx, shots_during=cost))
    detected = [r for r in rows if r["detected"]]
    recs = [r["recovery"] for r in rows if r["recovery"] is not None]
    out = dict(n_jumps=len(rows),
               detection_rate=float(len(detected) / len(rows)) if rows else None,
               mean_delay=float(np.mean([r["delay"] for r in detected])) if detected else None,
               false_alarms_per100=float(100.0 * n_false / n_steps) if n_steps else None,
               mean_recovery=float(np.mean(recs)) if recs else None,
               recovery_rate=float(len(recs) / len(rows)) if rows else None,
               mean_max_abs_after=float(np.mean([r["max_abs_after"] for r in rows])) if rows else None,
               mean_shots_during=float(np.mean([r["shots_during"] for r in rows])) if rows else None,
               per_jump=rows)
    return out
