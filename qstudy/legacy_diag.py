"""qstudy legacy_diag — Stage 5: diagnose the legacy Qiskit result without touching the legacy tree.
5.1: run the fine-tuned v4_qft_v5 policy in the legacy QiskitResonatorEnvV3 for 200 deterministic
     episodes (seed base 777000 + episode index), recording amplitude, true error, control freq.
5.2: lineshape replica test — rabi_legacy(delta) vs the legacy env's own Aer measurement at 20
     detunings x 100000 shots; plus the Fisher-information floor comparison (legacy single probe vs
     exact-Rabi single vs dither vs Ramsey I/Q) at shots 1024, e=0.02.
Outputs -> STUDY\\results\\legacy\\legacy_diag.npz + legacy_diag.json + lineshape_replica.json +
fisher_floors.json. Writes NOTHING into pre-existing legacy folders."""
import datetime, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engines import rabi_legacy_p

PROJECT_ROOT = os.path.dirname(HERE)
LEGACY_CKPT = os.path.join(PROJECT_ROOT, "data_logs", "qiskit_aer_simulation_2", "v4_qft_v5")
LEGACY_MODEL = os.path.join(LEGACY_CKPT, "best_model")
LEGACY_VN = os.path.join(LEGACY_CKPT, "vec_normalize_qft_v5.pkl")
OUT = os.path.join(HERE, "results", "legacy")
os.makedirs(OUT, exist_ok=True)
N_EPISODES = 200
SEED_BASE = 777000


def log(m):
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), m), flush=True)


def run_legacy_episodes():
    # loads the same checkpoint+VecNormalize stats as phase3_evaluate_v5.py; per-episode seeding is
    # done via the legacy env's constructor (it has no .seed() method; X3: never edit legacy code).
    sys.path.insert(0, PROJECT_ROOT)
    from qiskit_integration.qiskit_resonator_env_v3 import QiskitResonatorEnvV3
    from stable_baselines3 import PPO
    import pickle

    model = PPO.load(LEGACY_MODEL, device="cpu")
    with open(LEGACY_VN, "rb") as f:
        vn = pickle.load(f)
    obs_rms = vn.obs_rms
    clip = getattr(vn, "clip_obs", 10.0)
    eps_vn = getattr(vn, "epsilon", 1e-8)

    def norm(o):
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + eps_vn), -clip, clip).astype(np.float32)

    T = 200
    maes, nears, lock_lost, mae_gt5000 = [], [], [], 0
    trace_err = np.zeros((N_EPISODES, T))
    trace_fc = np.zeros((N_EPISODES, T))
    trace_x = np.zeros((N_EPISODES, T))
    trace_amp = np.zeros((N_EPISODES, T))
    t0 = time.time()
    for ep in range(N_EPISODES):
        env = QiskitResonatorEnvV3(shots=1024, seed=SEED_BASE + ep)
        obs, _ = env.reset()
        done = False
        errs, fcs, xs, amps = [], [], [], []
        run_bad = 0
        worst_run = 0
        step = 0
        while not done:
            a, _ = model.predict(norm(obs), deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            # attributes per Q23: info['amplitude'], info['freq_error_hz'], info['f_probe'], info['f_qubit']
            err = float(info["freq_error_hz"])
            errs.append(err)
            fcs.append(float(info["f_probe"]))
            xs.append(float(info["f_qubit"]))
            amps.append(float(info["amplitude"]))
            run_bad = run_bad + 1 if err > 10000 else 0
            worst_run = max(worst_run, run_bad)
            step += 1
        errs = np.array(errs[:T])
        trace_err[ep, :len(errs)] = errs
        trace_fc[ep, :len(fcs)] = fcs
        trace_x[ep, :len(xs)] = xs
        trace_amp[ep, :len(amps)] = amps
        maes.append(float(errs.mean()))
        nears.append(float((np.array(amps) > 0.90).mean()))
        lost = 1.0 if worst_run >= 20 else 0.0
        lock_lost.append(lost)
        if errs.mean() > 5000:
            mae_gt5000 += 1
        if (ep + 1) % 10 == 0:
            log("episode %d/%d mae_so_far=%.0f elapsed=%.0fs" % (ep + 1, N_EPISODES, np.mean(maes), time.time() - t0))
    np.savez_compressed(os.path.join(OUT, "legacy_diag.npz"),
                        mae=np.array(maes), near=np.array(nears), lock_lost=np.array(lock_lost),
                        trace_err=trace_err, trace_fc=trace_fc, trace_x=trace_x, trace_amp=trace_amp)
    summary = dict(status="ok", episodes=N_EPISODES, seed_base=SEED_BASE,
                   mae_mean=float(np.mean(maes)), mae_std=float(np.std(maes)),
                   near_mean=float(np.mean(nears)), near_std=float(np.std(nears)),
                   lock_lost_rate=float(np.mean(lock_lost)),
                   frac_mae_gt_5000=mae_gt5000 / N_EPISODES,
                   finished=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(OUT, "legacy_diag.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("5.1 done: %s" % json.dumps(summary))
    return summary


def lineshape_replica():
    """5.2: rabi_legacy(delta) = sin^2(pi*omega/(2*sqrt(omega^2+delta^2))) (legacy env v3, omega_rabi=2500;
    rx(theta) from |0>, qiskit_resonator_env_v3.py:54-61). Compare against the legacy env's own Aer
    measurement at 20 detunings x 100000 shots."""
    sys.path.insert(0, PROJECT_ROOT)
    from qiskit_integration.qiskit_resonator_env_v3 import QiskitResonatorEnvV3
    from engines import rabi_legacy_p
    env = QiskitResonatorEnvV3(shots=100000)
    env.f0_current = 500000.0
    deltas = np.linspace(-25000, 25000, 20)
    rows = []
    maxdiff = 0.0
    for d in deltas:
        f = 500000.0 + d
        theta = float(np.pi * env.omega_rabi / np.sqrt(env.omega_rabi ** 2 + d ** 2))
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(1, 1)
        qc.rx(theta, 0)
        qc.measure(0, 0)
        r = env.simulator.run(qc, shots=100000).result().get_counts()
        p_env = r.get("1", 0) / 100000
        p_rep = float(rabi_legacy_p(d, omega=2500.0))
        diff = abs(p_env - p_rep)
        maxdiff = max(maxdiff, diff)
        rows.append(dict(delta=float(d), p_env=p_env, p_replica=p_rep, absdiff=diff))
        log("5.2 d=%+7.0f env=%.4f replica=%.4f diff=%.4f" % (d, p_env, p_rep, diff))
    ok = maxdiff <= 0.01
    out = dict(status="ok" if ok else "BLOCKED_mismatch", max_abs_difference=maxdiff,
               threshold=0.01, rows=rows)
    with open(os.path.join(OUT, "lineshape_replica.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("5.2 %s maxdiff=%.4f" % (out["status"], maxdiff))
    return out, ok


def fisher_floors():
    """5.2b: static Fisher floors vs detuning: legacy single probe, exact-Rabi single, two-point dither,
    Ramsey I/Q — shots 1024 total, e=0.02 (dither/ramsey split shots in halves)."""
    import qphysics as qp
    deltas = np.linspace(-12000, 12000, 241)
    e = 0.02
    shots = 1024
    om_leg = 2500.0
    # legacy single probe: numeric derivative of readout(rabi_legacy_p)
    h = 25.0
    p = qp.readout(rabi_legacy_p(deltas, om_leg), e, e)
    dp = (qp.readout(rabi_legacy_p(deltas + h, om_leg), e, e) - qp.readout(rabi_legacy_p(deltas - h, om_leg), e, e)) / (2 * h)
    fish_legacy = shots * dp ** 2 / np.clip(p * (1 - p), 1e-9, None)
    fish_rabi = qp.fisher_rabi(deltas, shots, omega=5000.0, e01=e, e10=e)
    # dither: two rabi probes at +-2500, 512 shots each
    dd = 2500.0
    n = shots / 2
    pp = qp.readout(qp.rabi_p(deltas - dd, 5000.0), e, e)
    pm = qp.readout(qp.rabi_p(deltas + dd, 5000.0), e, e)
    dpp = (pp - qp.readout(qp.rabi_p(deltas - dd - h, 5000.0), e, e)) / h
    dpm = (pm - qp.readout(qp.rabi_p(deltas + dd - h, 5000.0), e, e)) / h
    fish_dither = n * (dpp ** 2 / np.clip(pp * (1 - pp), 1e-9, None) + dpm ** 2 / np.clip(pm * (1 - pm), 1e-9, None))
    fish_ramsey = qp.fisher_ramsey_iq(deltas, shots / 2, tau=50e-6, t2=200e-6, e01=e, e10=e)
    np.savez_compressed(os.path.join(OUT, "fisher_floors.npz"),
                        deltas=deltas, legacy=fish_legacy, rabi=fish_rabi,
                        dither=fish_dither, ramsey=fish_ramsey)
    out = dict(status="ok",
               min_std_floor_hz=dict(
                   legacy=float(1.0 / np.sqrt(fish_legacy.max())),
                   rabi=float(1.0 / np.sqrt(fish_rabi.max())),
                   dither=float(1.0 / np.sqrt(fish_dither.max())),
                   ramsey=float(1.0 / np.sqrt(fish_ramsey.max()))))
    with open(os.path.join(OUT, "fisher_floors.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("5.2b floors: %s" % json.dumps(out["min_std_floor_hz"]))
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if which in ("all", "diag"):
        try:
            run_legacy_episodes()
        except Exception as ex:
            import traceback
            log("5.1 BLOCKED: %r" % ex)
            with open(os.path.join(OUT, "legacy_diag.json"), "w") as f:
                json.dump(dict(status="BLOCKED", reason=repr(ex),
                               traceback=traceback.format_exc()[-2000:]), f, indent=1)
            ok = False
    if which in ("all", "replica"):
        _, ok_r = lineshape_replica()
        ok = ok and ok_r
    if which in ("all", "fisher"):
        fisher_floors()
    log("LEGACY DIAG DONE ok=%s" % ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
