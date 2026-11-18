"""qstudy engines — SurrogateEngine (analytic) + AerEngine (Qiskit Aer, Stage 4)."""
import numpy as np
import qphysics as qp


def rabi_legacy_p(delta, omega=2500.0):
    """Legacy Qiskit env lineshape (qiskit_resonator_env_v3.py:54-61): rx(theta) pulse with
    theta = pi*omega/sqrt(omega^2+delta^2) measured from |0>: P(|1>) = sin^2(theta/2).
    NOTE: this is NOT the exact detuned-Rabi-pulse curve rabi_p (different lineshape)."""
    d = np.asarray(delta, dtype=float)
    w = np.sqrt(omega ** 2 + d ** 2)
    return np.sin(np.pi * omega / (2.0 * w)) ** 2


class SurrogateEngine:
    """Analytic engine. counts(kind, delta[B], shots[B], t2[B], e01[B], e10[B], rngs, omega, tau)
    -> per-episode binomial draws from the exact probe probability with readout error applied LAST.
    Part-2 S16.2: optional cfg with contrast_power (default 1.0) and contrast_override
    (default None) routes every Ramsey probability through qphysics_ext.ramsey_p_ext
    (identical to qp.ramsey_p at the defaults)."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.contrast_power = float(cfg.get("contrast_power", 1.0))
        self.contrast_override = cfg.get("contrast_override", None)

    def _ramsey_p(self, delta, tau, t2, quad):
        if self.contrast_power == 1.0 and self.contrast_override is None:
            return qp.ramsey_p(delta, tau=tau, t2=t2, quad=quad)
        from qphysics_ext import ramsey_p_ext
        return ramsey_p_ext(delta, tau=tau, t2=t2, quad=quad,
                            power=self.contrast_power, override=self.contrast_override)

    def counts(self, kind, delta, shots, t2, e01, e10, rngs, omega=5000.0, tau=50e-6, depol_p=0.0):
        delta = np.asarray(delta, dtype=float)
        shots = np.asarray(shots, dtype=np.int64)
        t2 = np.asarray(t2, dtype=float)
        e01 = np.asarray(e01, dtype=float)
        e10 = np.asarray(e10, dtype=float)
        B = delta.shape[0]
        if kind == "rabi":
            p = qp.rabi_p(delta, omega=omega)
        elif kind == "rabi_legacy":
            p = rabi_legacy_p(delta, omega=omega)  # legacy rx-angle lineshape (Stage 5)
        elif kind == "ramsey_I":
            p = self._ramsey_p(delta, tau, t2, "I")
        elif kind == "ramsey_Q":
            p = self._ramsey_p(delta, tau, t2, "Q")
        else:
            raise ValueError(kind)
        if kind in ("ramsey_I", "ramsey_Q") and depol_p:
            # depolarizing gate-error factor (1-4p/3)^(#gates) applied to the Ramsey contrast (Stage 8)
            p = 0.5 + (p - 0.5) * (1.0 - 4.0 * depol_p / 3.0) ** 3
        p = qp.readout(p, e01, e10)  # readout error applied last
        out = np.empty(B, dtype=np.int64)
        for b in range(B):
            rng = rngs[b] if not isinstance(rngs, np.random.Generator) else rngs
            out[b] = rng.binomial(int(shots[b]), float(np.clip(p[b], 0.0, 1.0)))
        return out


class AerEngine:
    """Qiskit Aer engine with the SAME counts() interface as SurrogateEngine.

    Circuits exactly as validated in tests/test_aer_physics.py:
      Rabi:     UnitaryGate(rabi_unitary(delta))
      Ramsey I: rx(pi/2), id, rz(phi), rx(pi/2)
      Ramsey Q: rx(pi/2), id, rz(phi), ry(pi/2)
      phi = 2*pi*delta*tau
    Noise model (built once per cfg): thermal_relaxation_error(T1=100*T2, T2, tau) on 'id' of qubit 0;
    ReadoutError([[1-e01, e01], [e10, 1-e10]]). Optional depolarizing gate error on 'id'.
    Batch execution: all B circuits of a step run in ONE backend.run(...), shots uniform across the
    batch, noise parameters uniform across the batch (domain randomization is surrogate-only).
    seed_simulator drawn from the per-episode measurement RNG.
    Aer rabi pulses are treated noise-free apart from readout (documented limitation)."""

    def __init__(self, cfg=None):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, thermal_relaxation_error, depolarizing_error, pauli_error
        self._QC = QuantumCircuit
        self._UG = UnitaryGate
        cfg = cfg or {}
        self.omega = float(cfg.get("omega", 5000.0))
        self.tau = float(cfg.get("tau", 50e-6))
        self.t2 = float(cfg.get("t2", 200e-6))
        self.e01 = float(cfg.get("e01", 0.02))
        self.e10 = float(cfg.get("e10", 0.02))
        self.depol_p = float(cfg.get("depol_p", 0.0))
        # Part-2 S16.2: contrast_power/contrast_override select the Z-flip channel with
        # p=(1-C)/2 on the id gate INSTEAD of thermal relaxation whenever set (validated
        # equivalence of Appendix F); keep thermal relaxation otherwise.
        self.contrast_power = float(cfg.get("contrast_power", 1.0))
        self.contrast_override = cfg.get("contrast_override", None)
        self.noise_on = not (self.e01 == 0 and self.e10 == 0 and np.isinf(self.t2))
        nm = NoiseModel()
        use_zflip = (self.contrast_power != 1.0) or (self.contrast_override is not None)
        if use_zflip:
            from qphysics_ext import contrast_ext, dephasing_flip_prob
            c = float(contrast_ext(self.tau, self.t2, self.contrast_power, self.contrast_override))
            pz = float(dephasing_flip_prob(c))
            nm.add_quantum_error(pauli_error([("Z", pz), ("I", 1 - pz)]), ["id"], [0])
        else:
            if not np.isinf(self.t2):
                nm.add_quantum_error(thermal_relaxation_error(100 * self.t2, self.t2, self.tau), ["id"], [0])
        if self.depol_p > 0:
            nm.add_quantum_error(depolarizing_error(self.depol_p, 1), ["id"], [0])
        if self.e01 > 0 or self.e10 > 0:
            nm.add_readout_error(ReadoutError([[1 - self.e01, self.e01], [self.e10, 1 - self.e10]]), [0])
        self.noise_model = nm if not nm.is_ideal() else None
        import os
        cores = os.cpu_count() or 4
        self.sim = AerSimulator(noise_model=self.noise_model,
                                max_parallel_threads=max(1, cores // 4))

    def _rabi_circuit(self, delta):
        qc = self._QC(1)
        qc.append(self._UG(qp.rabi_unitary(float(delta), omega=self.omega)), [0])
        qc.measure_all()
        return qc

    def _ramsey_circuit(self, delta, quad):
        qc = self._QC(1)
        phi = 2 * np.pi * float(delta) * self.tau
        qc.rx(np.pi / 2, 0)
        qc.id(0)
        qc.rz(phi, 0)
        qc.rx(np.pi / 2, 0) if quad == "I" else qc.ry(np.pi / 2, 0)
        qc.measure_all()
        return qc

    def counts(self, kind, delta, shots, t2, e01, e10, rngs, omega=5000.0, tau=50e-6, depol_p=0.0):
        delta = np.asarray(delta, dtype=float)
        shots = np.asarray(shots, dtype=np.int64)
        B = delta.shape[0]
        n = int(shots[0])
        assert np.all(shots == n), "AerEngine requires uniform shots across the batch"
        assert kind != "rabi_legacy", "rabi_legacy is a surrogate-only probe kind"
        if kind == "rabi":
            circuits = [self._rabi_circuit(d) for d in delta]
        elif kind in ("ramsey_I", "ramsey_Q"):
            circuits = [self._ramsey_circuit(d, kind[-1]) for d in delta]
        else:
            raise ValueError(kind)
        seeds = []
        for b in range(B):
            rng = rngs[b] if not isinstance(rngs, np.random.Generator) else rngs
            seeds.append(int(rng.integers(0, 2 ** 31 - 1)))
        # one seed for the single batched run, drawn from the episode-0 RNG is not per-episode;
        # Aer supports one seed per run, so we batch per unique seed by running in groups of one
        # seed: to keep the interface simple we use a single run per step with seed from a
        # dedicated draw, which preserves identical stochasticity across engines only in
        # distribution (documented in docs/approach_5_engines.md).
        run_seed = int(rngs[0].integers(0, 2 ** 31 - 1)) if not isinstance(rngs, np.random.Generator) else int(rngs.integers(0, 2 ** 31 - 1))
        result = self.sim.run(circuits, shots=n, seed_simulator=run_seed).result()
        out = np.empty(B, dtype=np.int64)
        for b in range(B):
            counts = result.get_counts(b)
            out[b] = counts.get("1", 0)
        return out
