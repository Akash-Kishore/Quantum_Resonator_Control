"""qstudy stress — pre-registered condition catalogue (Appendix D)."""
import numpy as np

OFAT = {
    "drift_kind": ["gaussian", "ou", "onef", "telegraph", "sinusoid"],
    "drift_sigma": [250, 500, 1000, 2000, 4000],
    "shots": [64, 128, 256, 512, 1024, 4096, 16384],
    "readout": [0.0, 0.02, 0.05, 0.10, 0.20],
    "t2": [25e-6, 50e-6, 100e-6, 200e-6, 800e-6, np.inf],
    "action_delay": [0, 1, 2, 3],
    "dac_bits": [6, 8, 10, 12, 16],
    "omega": [2500, 5000, 10000],
    "tau": [25e-6, 50e-6, 100e-6],
}

VAL_SET = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]


def _fmt(v, factor=None):
    if isinstance(v, float) and np.isinf(v):
        return "inf"
    if factor in ("t2", "tau") and v is not None:
        return "%g" % (v * 1e6)  # condition ids express t2/tau in microseconds (Appendix D)
    if isinstance(v, float):
        return ("%g" % v)
    return str(v)


def _ofat_id(factor, value):
    return "ofat_%s_%s" % (factor, _fmt(value, factor) if factor != "drift_kind" else value)


def cond_cfg(cond_id):
    """cfg overrides for a condition id (relative to the nominal condition)."""
    if cond_id == "nom":
        return {}
    if not cond_id.startswith("ofat_"):
        if cond_id.startswith("g2_shotsT2_"):
            i, j = [int(x) for x in cond_id[len("g2_shotsT2_"):].split("_")]
            return {"shots": OFAT["shots"][i], "t2": OFAT["t2"][j]}
        if cond_id.startswith("g2_sigmaRO_"):
            i, j = [int(x) for x in cond_id[len("g2_sigmaRO_"):].split("_")]
            return {"drift_sigma": OFAT["drift_sigma"][i], "e01": OFAT["readout"][j], "e10": OFAT["readout"][j]}
        raise KeyError(cond_id)
    rest = cond_id[len("ofat_"):]
    for factor in sorted(OFAT, key=len, reverse=True):
        if rest.startswith(factor + "_"):
            raw = rest[len(factor) + 1:]
            if factor == "drift_kind":
                assert raw in OFAT[factor], raw
                return {"drift": raw}
            vals = OFAT[factor]
            for v in vals:
                if _fmt(v, factor) == raw:
                    if factor == "readout":
                        return {"e01": v, "e10": v}
                    return {factor: v}
            raise KeyError("no OFAT value matches %s" % cond_id)
    raise KeyError(cond_id)


def applies_to(cond_id, probe):
    """Factors that do not apply to a probe are skipped for that probe (documented in PREREG)."""
    if cond_id.startswith("ofat_omega_") and probe in ("ramsey_iq",):
        return False
    if cond_id.startswith("ofat_tau_") and probe in ("rabi", "dither", "rabi_legacy"):
        return False
    if cond_id.startswith("ofat_t2_") and probe in ("rabi", "dither", "rabi_legacy"):
        return False  # T2 acts only through the Ramsey contrast
    return True


# ---------------------------------------------------------------- Part-2 O3: DR-support split
def dr_support(cond_id):
    """IN/OUT: is this stress cell inside the DR sampling ranges of Stage 11.1?
    A cell is IN if ALL its overridden values lie inside the DR ranges
    (drift kind in {gaussian, ou, onef, telegraph, sinusoid}; drift sigma 250-1500;
    shots 64-4096; e01<=0.05 and e10<=0.08; t2 40-400 us or nominal; action_delay<=2;
    dac_bits>=8; omega/tau variations count as IN), else OUT.
    The NOMINAL condition is IN (nominal values are inside every range)."""
    if cond_id == "nom":
        return "IN"
    overrides = cond_cfg(cond_id)
    for k, v in overrides.items():
        if k == "drift":
            if v not in ("gaussian", "ou", "onef", "telegraph", "sinusoid"):
                return "OUT"
        elif k == "drift_sigma":
            if not (250 <= float(v) <= 1500):
                return "OUT"
        elif k == "shots":
            if not (64 <= int(v) <= 4096):
                return "OUT"
        elif k in ("e01", "e10"):
            lim = 0.05 if k == "e01" else 0.08
            if float(v) > lim + 1e-12:
                return "OUT"
        elif k == "t2":
            if not (np.isinf(v) or 40e-6 <= float(v) <= 400e-6):
                return "OUT"
        elif k == "action_delay":
            if int(v) > 2:
                return "OUT"
        elif k == "dac_bits":
            if int(v) < 8:
                return "OUT"
        elif k in ("omega", "tau"):
            pass  # omega/tau variations count as IN
    return "IN"


def list_conds(set_name):
    if set_name == "val":
        return list(VAL_SET)
    if set_name == "ofat":
        out = []
        for factor, vals in OFAT.items():
            for v in vals:
                out.append(_ofat_id(factor, v))
        return out
    if set_name == "grids":
        out = []
        for i in range(len(OFAT["shots"])):
            for j in range(len(OFAT["t2"])):
                out.append("g2_shotsT2_%d_%d" % (i, j))
        for i in range(len(OFAT["drift_sigma"])):
            for j in range(len(OFAT["readout"])):
                out.append("g2_sigmaRO_%d_%d" % (i, j))
        return out
    raise KeyError(set_name)


if __name__ == "__main__":
    import sys
    print("val:", list_conds("val"))
    print("ofat:", len(list_conds("ofat")), "grids:", len(list_conds("grids")))
    for cid in list_conds("val") + list_conds("ofat")[:5]:
        print(cid, "->", cond_cfg(cid))
