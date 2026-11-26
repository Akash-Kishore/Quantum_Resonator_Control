"""qstudy calibrate_filter — Part-2 S17.3: uncertainty calibration of the Bayes filter.

For the 50% and 90% credible intervals of the WINDOWED posterior (same window as the
tracker's mean_std): empirical coverage, mean width, log score and NLL over episodes,
in the nominal condition and 10 fixed stress cells (pre-committed list below), per
probe. Also states the Fisher-information floor (CRB) versus the achieved estimation
rms for each probe and shot budget (64/256/1024/4096 shots at nominal drift).

The filter runs in its operating condition: bayes:nominal controls the env while the
calibration records the windowed posterior and the true x each step. Log score =
mean of log(windowed posterior mass of the grid cell containing the true x); NLL =
-mean log score. Cells not applicable to a probe (stress.applies_to) are skipped and
listed. Exploratory (PREREG2); this script never tunes anything.
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qphysics as qp
import stress
from envs import BatchEnv, DEFAULT_CFG
from controllers import BayesTracker
from evaluate import git_head, sha256_file, versions
from eval_changepoint import sealed_gate

# 10 fixed stress cells (pre-committed; nominal added separately)
CELLS10 = ["ofat_drift_sigma_1000", "ofat_drift_sigma_2000", "ofat_shots_64",
           "ofat_shots_128", "ofat_readout_0.10", "ofat_t2_50",
           "ofat_action_delay_2", "ofat_dac_bits_8",
           "ofat_drift_kind_onef", "ofat_drift_kind_telegraph"]
CRB_SHOT_BUDGETS = [64, 256, 1024, 4096]


def windowed(post, g, window):
    """Restrict each posterior row to |g - g_MAP| <= window; return renormalised weights."""
    gm = g[np.argmax(post, axis=1)]
    w = post * (np.abs(g[None, :] - gm[:, None]) <= window)
    return w / np.clip(w.sum(axis=1, keepdims=True), 1e-300, None)


def interval(w, g, lo_q, hi_q):
    """Central credible interval [q_lo, q_hi] of the windowed posterior (grid quantiles)."""
    cw = np.cumsum(w, axis=1)
    cw = cw / np.clip(cw[:, -1:], 1e-300, None)
    idx_lo = np.array([np.searchsorted(cw[b], lo_q) for b in range(w.shape[0])])
    idx_hi = np.array([np.searchsorted(cw[b], hi_q) for b in range(w.shape[0])])
    return g[idx_lo], g[idx_hi]


def cell_mass(w, g, x_true):
    """Windowed posterior mass of the grid cell containing the true x."""
    dg = g[1] - g[0]
    idx = np.clip(np.round((x_true - g[0]) / dg).astype(int), 0, len(g) - 1)
    return w[np.arange(w.shape[0]), idx]


def run_episodes(cfg, B, seed_base, burn_in):
    """Run bayes:nominal for one episode batch; return per-step (mu, windowed posterior,
    true x, control offset fc) after burn-in."""
    env = BatchEnv(cfg, B, seed_base)
    env.reset()
    flt = BayesTracker(cfg, assumed="nominal", env_view=env)
    g = flt.gb.g
    window = flt.window
    mus, ws, xs, fcs = [], [], [], []
    T = env.T
    for t in range(T):
        a = flt.act(None)
        obs, rew, done, info = env.step(a.reshape(-1, 1))
        if t + 1 >= burn_in:
            mus.append(flt.last_mu.copy())
            ws.append(windowed(flt.post, g, window))
            xs.append(info["x"].copy())
            fcs.append(info["fc"].copy())
    return np.asarray(mus), np.asarray(ws), np.asarray(xs), np.asarray(fcs), env


def calibrate(cfg, B, seed_base, burn_in):
    mu_tr, w_tr, x_tr, fc_tr, env = run_episodes(cfg, B, seed_base, burn_in)
    Tn, Bn = x_tr.shape
    g = np.linspace(-30000.0, 30000.0, 601)
    cov = {"50": [], "90": []}
    wid = {"50": [], "90": []}
    logs = []
    for t in range(Tn):
        w = w_tr[t]
        x = x_tr[t]
        for lev, lo, hi in (("50", 0.25, 0.75), ("90", 0.05, 0.95)):
            qlo, qhi = interval(w, g, lo, hi)
            cov[lev].append(float(((x >= qlo) & (x <= qhi)).mean()))
            wid[lev].append(float((qhi - qlo).mean()))
        p = np.clip(cell_mass(w, g, x), 1e-300, None)
        logs.append(np.log(p))
    return dict(coverage50=float(np.mean(cov["50"])), coverage90=float(np.mean(cov["90"])),
                width50=float(np.mean(wid["50"])), width90=float(np.mean(wid["90"])),
                logscore=float(np.mean(logs)), nll=float(-np.mean(logs)),
                steps_recorded=int(Tn * Bn))


def fisher_dither(delta, n, omega=5000.0, dd=2500.0, e01=0.02, e10=0.02, h=1.0):
    delta = np.asarray(delta, dtype=float)
    out = np.zeros_like(delta)
    for sgn in (-1.0, 1.0):
        d = delta + sgn * dd
        p = lambda x: qp.readout(qp.rabi_p(x, omega=omega), e01, e10)
        dp = (p(d + h) - p(d - h)) / (2 * h)
        pp = np.clip(p(d), 1e-9, None)
        out += n * dp ** 2 / np.clip(pp * (1 - pp), 1e-9, None)
    return out


def crb_row(cfg, B, seed_base, burn_in):
    """CRB floor vs achieved rms at the cell's shot budget (nominal drift). The Fisher
    information is evaluated at the mean OPERATING detuning |x - fc| observed in the run
    (the filter tracks x, so the relevant estimation error is x - fc)."""
    mu_tr, w_tr, x_tr, fc_tr, env = run_episodes(cfg, B, seed_base, burn_in)
    op_det = float(np.mean(np.abs(x_tr - fc_tr)))
    shots = float(cfg.get("shots", 1024))
    probe = cfg.get("probe", "rabi")
    if probe == "ramsey_iq":
        fi = qp.fisher_ramsey_iq(op_det, shots / 2.0,
                                 tau=cfg.get("tau", 50e-6), t2=cfg.get("t2", 200e-6),
                                 e01=cfg.get("e01", 0.02), e10=cfg.get("e10", 0.02))
    elif probe == "dither":
        fi = fisher_dither(op_det, shots / 2.0,
                           omega=cfg.get("omega", 5000.0), dd=cfg.get("dither_delta", 2500.0),
                           e01=cfg.get("e01", 0.02), e10=cfg.get("e10", 0.02))
    else:
        fi = qp.fisher_rabi(op_det, shots, omega=cfg.get("omega", 5000.0),
                            e01=cfg.get("e01", 0.02), e10=cfg.get("e10", 0.02))
    achieved = float(np.sqrt(np.mean((mu_tr - x_tr) ** 2)))
    return dict(crb_floor=float(1.0 / np.sqrt(max(float(fi), 1e-12))), achieved_rms=achieved,
                mean_operating_detuning=op_det)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["calib", "crb"])
    ap.add_argument("--probe", required=True, choices=["rabi", "dither", "ramsey_iq"])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--burn_in", type=int, default=20)
    ap.add_argument("--seed_base", type=int, required=True)
    ap.add_argument("--split", required=True, choices=["tune", "val", "sealed"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("SKIP (exists): %s" % a.out)
        return 0
    if not sealed_gate(a.split, a.seed_base):
        print("SEALED REFUSED: gate failed", flush=True)
        return 3
    rows = []
    t0 = time.time()
    if a.mode == "calib":
        for cond in ["nom"] + CELLS10:
            if not stress.applies_to(cond, a.probe):
                continue
            cfg = dict(DEFAULT_CFG)
            cfg.update(probe=a.probe)
            cfg.update(stress.cond_cfg(cond))
            r = dict(probe=a.probe, cond=cond, episodes=a.episodes, seed_base=a.seed_base)
            r.update(calibrate(cfg, a.episodes, a.seed_base, a.burn_in))
            rows.append(r)
            print("PROGRESS %s %s done (%.0fs)" % (a.probe, cond, time.time() - t0), flush=True)
    else:
        for shots in CRB_SHOT_BUDGETS:
            cfg = dict(DEFAULT_CFG)
            cfg.update(probe=a.probe, shots=shots)
            r = dict(probe=a.probe, cond="crb_shots%d" % shots, episodes=a.episodes,
                     seed_base=a.seed_base, shots=shots)
            r.update(crb_row(cfg, a.episodes, a.seed_base, a.burn_in))
            rows.append(r)
            print("PROGRESS %s crb shots=%d done (%.0fs)" % (a.probe, shots, time.time() - t0), flush=True)
    out = dict(status="ok", mode=a.mode, probe=a.probe, split=a.split,
               cells=CELLS10, git_head=git_head(), versions=versions(),
               cmd=" ".join(sys.argv), rows=rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("DONE calibrate_filter %s %s rows=%d -> %s" % (a.mode, a.probe, len(rows), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
