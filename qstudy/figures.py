"""qstudy figures — F01-F28 generator (Stage 13.3). Early figures (F01-F07) are generated from
physics/validation/legacy data as soon as those exist; RL-dependent figures are added in Stage 13.
Each figure: PNG (300 dpi) + PDF; consistent per-method-family colours; unit-bearing axis labels;
figures\\INDEX.md with caption stubs (n, seeds, conditions)."""
import glob, json, os, sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
RES = os.path.join(HERE, "results")

COLORS = {"noctrl": "#888888", "oracle": "#222222", "hill": "#e67e22", "pid": "#3498db",
          "bayes": "#9b59b6", "rl": "#27ae60", "legacy": "#c0392b", "rabi": "#e74c3c",
          "dither": "#f39c12", "ramsey": "#2980b9"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("figure", name, "written")


def f01_circuits():
    import qiskit
    from qiskit import QuantumCircuit
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    names = ["Rabi pi-pulse", "Ramsey I", "Ramsey Q"]
    try:
        for ax, nm in zip(axs, names):
            qc = QuantumCircuit(1)
            if nm.startswith("Rabi"):
                qc.x(0)
            else:
                qc.h(0)
                qc.id(0)
                qc.rz(0.7, 0)
                qc.h(0)
            try:
                qc.draw("mpl", ax=ax, scale=0.8)
                ax.set_title(nm)
            except Exception:
                ax.text(0.5, 0.5, qc.draw("text").single_string(), fontsize=5, ha="center", va="center")
                ax.set_title(nm)
                ax.axis("off")
    except Exception as e:
        for ax, nm in zip(axs, names):
            ax.text(0.5, 0.5, "circuit drawing unavailable: %r" % e, ha="center", va="center")
            ax.set_title(nm)
            ax.axis("off")
    save(fig, "fig01_circuits")


def f02_lineshapes():
    import qphysics as qp
    from engines import rabi_legacy_p
    d = np.linspace(-25000, 25000, 1000)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d, qp.rabi_p(d, 5000), label="exact Rabi pi-pulse (omega=5 kHz)", color=COLORS["rabi"])
    ax.plot(d, rabi_legacy_p(d, 2500), label="legacy rx-angle (omega=2.5 kHz)", color=COLORS["legacy"], ls="--")
    ax.plot(d, qp.ramsey_p(d, quad="I"), label="Ramsey I (tau=50 us, T2=200 us)", color=COLORS["ramsey"])
    ax.plot(d, qp.ramsey_p(d, quad="Q"), label="Ramsey Q", color=COLORS["ramsey"], ls=":")
    for s in (-2500, 2500):
        ax.axvline(s, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("detuning delta (Hz)")
    ax.set_ylabel("P(|1>)")
    ax.legend(fontsize=8)
    ax.set_title("Probe line shapes; dotted verticals mark the +-dither points")
    save(fig, "fig02_lines_shapes")


def f03_fisher():
    z = np.load(os.path.join(RES, "legacy", "fisher_floors.npz"))
    d = z["deltas"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, c in (("legacy", COLORS["legacy"]), ("rabi", COLORS["rabi"]), ("dither", COLORS["dither"]), ("ramsey", COLORS["ramsey"])):
        floor = 1.0 / np.sqrt(np.clip(z[k], 1e-12, None))
        floor = np.where(np.isfinite(floor), floor, np.nan)
        ax.plot(d, floor, label=k, color=c)
    ax.set_xlabel("detuning (Hz)")
    ax.set_ylabel("static CRB std floor (Hz)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Fisher/CRB floor vs detuning per probe (shots 1024, e=0.02)")
    save(fig, "fig03_fisher_floor")


def f04_bloch():
    fig = plt.figure(figsize=(9, 3))
    try:
        from qiskit.visualization import plot_bloch_vector
        ax = fig.add_subplot(131, projection="3d")
        plot_bloch_vector([0, 0, 1], ax=ax)
        ax.set_title("delta=0")
    except Exception:
        ax = fig.add_subplot(131, projection="3d")
        ax.set_title("delta=0 (bloch unavailable)")
    # manual sphere schematic
    for i, (d, lab) in enumerate(((1000, "delta=+1000 Hz"), (4000, "delta=+4000 Hz"))):
        ax = fig.add_subplot(1, 3, i + 2, projection="3d")
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), alpha=0.1, color="gray")
        th = np.pi / 2 + 2 * np.pi * d * 50e-6
        ax.quiver(0, 0, 0, np.cos(th), np.sin(th), 0, color=COLORS["ramsey"])
        ax.set_title(lab, fontsize=9)
        ax.set_axis_off()
    save(fig, "fig04_bloch_paths")


def f05_iq_clouds():
    import qphysics as qp
    rng = np.random.default_rng(5)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, d in zip(axs, (0.0, 1000.0, 4000.0)):
        for shots, alpha in ((64, 0.4), (1024, 0.9)):
            pI = qp.readout(qp.ramsey_p(d, quad="I"), 0.02, 0.02)
            pQ = qp.readout(qp.ramsey_p(d, quad="Q"), 0.02, 0.02)
            x = rng.binomial(shots, pI, 400) / shots
            y = rng.binomial(shots, pQ, 400) / shots
            ax.scatter(x, y, s=6, alpha=alpha, label="%d shots" % shots)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("P_I")
        ax.set_ylabel("P_Q")
        ax.set_title("delta=%.0f Hz" % d, fontsize=9)
        ax.legend(fontsize=7)
    save(fig, "fig05_iq_phasor_clouds")


def f06_drift_catalog():
    import qphysics as qp
    fig, axs = plt.subplots(2, 5, figsize=(16, 5.5))
    for j, kind in enumerate(("gaussian", "ou", "onef", "telegraph", "sinusoid")):
        d = qp.Drift(kind, np.random.default_rng(11), sigma=500.0)
        d.reset(500.0)
        xs = [d.step() for _ in range(500)]
        axs[0, j].plot(xs, lw=0.7)
        axs[0, j].set_title(kind)
        axs[0, j].set_ylabel("offset (Hz)" if j == 0 else "")
        f, p = _psd(xs)
        axs[1, j].semilogy(f[1:], p[1:], lw=0.7)
        axs[1, j].set_xlabel("freq (1/step)")
        axs[1, j].set_ylabel("PSD" if j == 0 else "")
    save(fig, "fig06_drift_catalog")


def _psd(x):
    from scipy.signal import welch
    return welch(np.asarray(x, float), fs=1.0, nperseg=min(256, len(x)))


def f07_validation():
    v1 = json.load(open(os.path.join(RES, "validation", "v1_expected_probability.json")))
    rows = v1["rows"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    pa = np.array([r["p_aer"] for r in rows])
    ps = np.array([r["p_sur"] for r in rows])
    axs[0].scatter(ps, pa, s=8)
    axs[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axs[0].set_xlabel("surrogate P")
    axs[0].set_ylabel("Aer P (200000 shots)")
    axs[0].set_title("V1 parity (117 cells)")
    res = pa - ps
    axs[1].hist(res, bins=30)
    axs[1].set_xlabel("p_aer - p_sur")
    axs[1].set_title("V1 residual histogram")
    j3 = json.load(open(os.path.join(RES, "validation", "v3_closed_loop.json")))
    j3b_path = os.path.join(RES, "validation", "v3b_bayes_closed_loop.json")
    labels = ["HillClimb\nsurrogate", "HillClimb\nAer"]
    vals = [j3["lock_w1000"]["surrogate"], j3["lock_w1000"]["aer"]]
    if os.path.exists(j3b_path):
        j3b = json.load(open(j3b_path))
        labels += ["Bayes\nsurrogate", "Bayes\nAer"]
        vals += [j3b["lock_w1000"]["surrogate"], j3b["lock_w1000"]["aer"]]
    axs[2].bar(labels, vals, color=["#e67e22", "#f39c12", "#9b59b6", "#b58ad0"][:len(labels)])
    axs[2].set_ylabel("lock_w1000 (100 episodes)")
    axs[2].set_title("V3/V3b closed-loop agreement")
    save(fig, "fig07_surrogate_vs_aer")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "early"
    if which in ("early", "all"):
        f01_circuits()
        f02_lineshapes()
        f03_fisher()
        f04_bloch()
        f05_iq_clouds()
        f06_drift_catalog()
        f07_validation()
    print("FIGURES stage: early set done")


if __name__ == "__main__":
    main()
