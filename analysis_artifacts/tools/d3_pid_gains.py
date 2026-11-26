"""D3 (P2): which gains does ctrl:pid actually use per OFAT drift_kind cell?

Replicates the exact lookup evaluate.py performs (merged cfg has "drift": <kind>
from stress.cond_cfg; PIDController(cfg, env) is evaluate.py make_controller).
"""
import sys

sys.path.insert(0, r"C:\Projects\MiniProject_Sem4\qstudy")
import controllers as C


class StubEnv:
    def __init__(self, probe):
        self.probe, self.B = probe, 1


KINDS = ["gaussian", "ou", "onef", "telegraph", "sinusoid"]
for probe in ("dither", "ramsey_iq"):
    env = StubEnv(probe)
    for k in KINDS:
        c = C.PIDController({"probe": probe, "drift": k}, env)
        ap = c.assumed_params()
        print("ofat_drift_kind_%-10s %-10s kp=%.3g ki=%.3g kd=%.3g | %s"
              % (k, probe, c.kp, c.ki, c.kd, ap["pid_gains_source"]))
    c = C.PIDController({"probe": probe}, env)
    print("nom(no drift key)      %-10s kp=%.3g ki=%.3g kd=%.3g | %s"
          % (probe, c.kp, c.ki, c.kd, c.assumed_params()["pid_gains_source"]))
