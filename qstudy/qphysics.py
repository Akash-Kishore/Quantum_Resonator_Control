"""Reference physics for the quantum resonator-tracking study (verified against Qiskit Aer)."""
import numpy as np

F_NOM = 500000.0          # Hz, nominal qubit/resonator frequency
CLIP = 25000.0            # Hz, drift is clipped to F_NOM +/- CLIP

# ---------------------------------------------------------------- probes (ideal expected P(|1>))
def rabi_p(delta, omega=5000.0):
    """Detuned Rabi pi-pulse of duration t=1/(2*omega). delta, omega in Hz. Peak (P=1) at delta=0."""
    d = np.asarray(delta, dtype=float)
    w = np.sqrt(omega ** 2 + d ** 2)
    return (omega ** 2 / w ** 2) * np.sin(np.pi * w / (2.0 * omega)) ** 2

def rabi_unitary(delta, omega=5000.0):
    """2x2 unitary of the same pulse (for Aer). H=(wr/2)sx+(dl/2)sz, wr=2*pi*omega, dl=2*pi*delta, t=1/(2*omega)."""
    wr, dl = 2 * np.pi * omega, 2 * np.pi * float(delta)
    w = np.hypot(wr, dl)
    t = 1.0 / (2.0 * omega)
    c, s = np.cos(w * t / 2), np.sin(w * t / 2)
    sx = np.array([[0, 1], [1, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    return c * np.eye(2) - 1j * s * (wr * sx + dl * sz) / w

def ramsey_contrast(tau, t2):
    """exp(-tau/T2); t2 may be None/inf (no dephasing) or an array (per-episode)."""
    t2 = np.inf if t2 is None else np.asarray(t2, dtype=float)
    return np.exp(-tau / t2)

def ramsey_p(delta, tau=50e-6, t2=200e-6, quad="I"):
    """Ramsey with Rx(pi/2)-wait(tau)-[Rx|Ry](pi/2). I: (1+C cos phi)/2 ; Q: (1+C sin phi)/2 ; phi=2*pi*delta*tau."""
    c = ramsey_contrast(tau, t2)
    phi = 2 * np.pi * np.asarray(delta, dtype=float) * tau
    return 0.5 * (1.0 + c * (np.cos(phi) if quad == "I" else np.sin(phi)))

def readout(p1, e01=0.0, e10=0.0):
    """e01 = P(read 1 | state 0), e10 = P(read 0 | state 1)."""
    return e01 + (1.0 - e01 - e10) * np.asarray(p1, dtype=float)

def sample_counts(rng, p, shots):
    return rng.binomial(np.asarray(shots, dtype=np.int64), np.clip(p, 0.0, 1.0))

def ramsey_phase(pI_hat, pQ_hat):
    """Signed phase estimate in (-pi, pi]; detuning estimate = phase/(2*pi*tau), unambiguous for |delta|<1/(2*tau)."""
    return np.arctan2(2 * np.asarray(pQ_hat) - 1, 2 * np.asarray(pI_hat) - 1)

# ---------------------------------------------------------------- Fisher information / CRB (static estimate floor)
def fisher_ramsey_iq(delta, shots_each, tau=50e-6, t2=200e-6, e01=0.0, e10=0.0):
    c = ramsey_contrast(tau, t2)
    phi = 2 * np.pi * np.asarray(delta, float) * tau
    k = (1 - e01 - e10) * 0.5 * c * 2 * np.pi * tau
    pI = readout(ramsey_p(delta, tau, t2, "I"), e01, e10)
    pQ = readout(ramsey_p(delta, tau, t2, "Q"), e01, e10)
    dI, dQ = -k * np.sin(phi), k * np.cos(phi)
    eps = 1e-9
    return shots_each * (dI ** 2 / np.clip(pI * (1 - pI), eps, None) + dQ ** 2 / np.clip(pQ * (1 - pQ), eps, None))

def fisher_rabi(delta, shots, omega=5000.0, e01=0.0, e10=0.0, h=1.0):
    p = lambda d: readout(rabi_p(d, omega), e01, e10)
    dp = (p(np.asarray(delta, float) + h) - p(np.asarray(delta, float) - h)) / (2 * h)
    pp = p(delta)
    return shots * dp ** 2 / np.clip(pp * (1 - pp), 1e-9, None)

# ---------------------------------------------------------------- drift generators (per control step)
class Drift:
    """Offset (Hz) of the true frequency from F_NOM. kinds: gaussian, ou, onef, telegraph, sinusoid."""
    THETA_ONEF = np.array([0.5, 0.2, 0.08, 0.03, 0.012, 0.005])

    def __init__(self, kind, rng, sigma=500.0, theta=0.05, amp=3000.0):
        self.kind, self.rng, self.sigma, self.theta, self.amp = kind, rng, float(sigma), float(theta), float(amp)

    def reset(self, init_offset):
        r = self.rng
        self.t = 0
        self.x = float(init_offset)
        if self.kind == "onef":
            var_ou = self.sigma ** 2 / (2 * self.theta - self.theta ** 2)
            v = var_ou / len(self.THETA_ONEF)
            self.s = np.sqrt(v * (2 * self.THETA_ONEF - self.THETA_ONEF ** 2))
            self.comp = r.normal(0.0, np.sqrt(v), len(self.THETA_ONEF))
            self.comp += (self.x - self.comp.sum()) / len(self.THETA_ONEF)
        elif self.kind == "telegraph":
            self.state, self.y = 1.0 if r.random() < 0.5 else -1.0, 0.0
            self.x = self.amp * self.state * 0.0 + float(init_offset)
            self.base = float(init_offset)
        elif self.kind == "sinusoid":
            self.period, self.ph = r.uniform(60, 120), r.uniform(0, 2 * np.pi)
            self.y, self.base = 0.0, float(init_offset)
        return self.x

    def step(self):
        r, k = self.rng, self.kind
        if k == "gaussian":
            self.x += self.sigma * r.normal()
        elif k == "ou":
            self.x += -self.theta * self.x + self.sigma * r.normal()
        elif k == "onef":
            self.comp += -self.THETA_ONEF * self.comp + self.s * r.normal(size=self.comp.shape)
            self.x = float(self.comp.sum())
        elif k == "telegraph":
            if r.random() < 0.02:
                self.state = -self.state
            self.y += -0.2 * self.y + 0.5 * self.sigma * r.normal()
            self.x = self.base + self.amp * self.state * 0.5 + self.y
        elif k == "sinusoid":
            self.t += 1
            self.y += -0.2 * self.y + 0.5 * self.sigma * r.normal()
            self.x = self.base + self.amp * np.sin(2 * np.pi * self.t / self.period + self.ph) + self.y
        else:
            raise ValueError(k)
        self.x = float(np.clip(self.x, -CLIP, CLIP))
        return self.x

def quantize(x, bits, half_range=CLIP):
    if bits is None:
        return x
    lsb = 2.0 * half_range / (2 ** int(bits))
    return np.round(np.asarray(x, float) / lsb) * lsb

# ---------------------------------------------------------------- grid Bayes filter (vectorised over a batch)
class GridBayes:
    """Posterior over true-frequency offset on a grid. drift model: x' = x - theta*x + sigma*N (theta=0 -> random walk)."""
    def __init__(self, half_range=30000.0, n=601, sigma=500.0, theta=0.0):
        self.g = np.linspace(-half_range, half_range, n)
        d = self.g[:, None] - ((1 - theta) * self.g)[None, :]          # row i: new x_i ; col j: old x_j
        T = np.exp(-0.5 * (d / sigma) ** 2)
        self.T = T / T.sum(axis=0, keepdims=True)                        # columns sum to 1
        self.n = n

    def init(self, batch, prior_std=8000.0):
        p = np.exp(-0.5 * (self.g / prior_std) ** 2)
        return np.tile(p / p.sum(), (batch, 1))

    def predict(self, post):
        out = post @ self.T.T
        return out / out.sum(axis=1, keepdims=True)

    def update(self, post, loglik):
        l = loglik - loglik.max(axis=1, keepdims=True)
        out = post * np.exp(l)
        s = out.sum(axis=1, keepdims=True)
        return out / np.where(s > 0, s, 1.0)

    def mean_std(self, post, window=5000.0):
        """Mean/std of the dominant posterior mode: restrict to |g - g_MAP| <= window (Hz) and renormalise.
        Needed because Ramsey/Rabi likelihoods are periodic/symmetric and the plain posterior mean averages alias peaks."""
        if window is None:
            m = post @ self.g
            v = post @ (self.g ** 2) - m ** 2
            return m, np.sqrt(np.clip(v, 0, None))
        gm = self.g[np.argmax(post, axis=1)]
        mask = (np.abs(self.g[None, :] - gm[:, None]) <= window)
        w = post * mask
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-300, None)
        m = w @ self.g
        v = w @ (self.g ** 2) - m ** 2
        return m, np.sqrt(np.clip(v, 0, None))

def binom_loglik(k, n, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return k * np.log(p) + (n - k) * np.log(1 - p)
