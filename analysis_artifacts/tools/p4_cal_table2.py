import json, os, datetime
Q = r"C:\Projects\MiniProject_SemUDY_WORK"  # placeholder, replaced below
Q = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\queue"
SM = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\smoke"
def wall(jid):
    s = json.load(open(os.path.join(Q, jid + ".status.json"), encoding="utf-8"))
    a = datetime.datetime.fromisoformat(s["started"]); b = datetime.datetime.fromisoformat(s["finished"])
    return (b - a).total_seconds()
NC = {"rl": 34, "pid": 34, "bayes": 67, "aer": 6}
LO_HI = {"rl": ("p4_cal_rl_5", 5, "p4_cal_rl_300", 300), "pid": ("p4_cal_pid_5", 5, "p4_cal_pid_300", 300),
         "bayes": ("p4_cal_bayes_5", 5, "p4_cal_bayes_300", 300), "aer": ("p4_cal_aer_2", 2, "p4_cal_aer_50", 50)}
cls = {}
for c, (lo, elo, hi, ehi) in LO_HI.items():
    tlo, thi = wall(lo), wall(hi)
    slope = max((thi - tlo) / (ehi - elo), 0.0)
    fixed = tlo - elo * slope
    pcf, pce = fixed / NC[c], slope / NC[c]
    cls[c] = (slope, fixed, pcf, pce)
    print("%-6s slope_job=%8.4f fixed_job=%7.2f per_cell_fixed=%7.4f per_cell_episode=%8.6f" % (c, slope, fixed, pcf, pce))
RLC, RLE = 9300, 1647000; NBC, NBE = 936, 247500; BAC, BAE = 624, 165000; AEC, AEE = 160, 16000
rl_s = RLC*cls["rl"][2] + RLE*cls["rl"][3]
nb_s = NBC*cls["pid"][2] + NBE*cls["pid"][3]
ba_s = BAC*cls["bayes"][2] + BAE*cls["bayes"][3]
ae_s = AEC*cls["aer"][2] + AEE*cls["aer"][3]
print("seconds: rl=%.0f non-bayes=%.0f bayes=%.0f aer=%.0f" % (rl_s, nb_s, ba_s, ae_s))
print("cpu_wall_hours=%.2f  aer_hours=%.2f  larger=%.2f" % ((rl_s+nb_s+ba_s)/14400.0, ae_s/3600.0, max((rl_s+nb_s+ba_s)/14400.0, ae_s/3600.0)))
json.dump({"per_class": {k: {"slope_job": v[0], "fixed_job": v[1], "per_cell_fixed": v[2], "per_cell_episode": v[3]} for k, v in cls.items()},
           "seconds": {"rl": rl_s, "nonbayes": nb_s, "bayes": ba_s, "aer": ae_s},
           "cpu_wall_hours": (rl_s+nb_s+ba_s)/14400.0, "aer_hours": ae_s/3600.0},
          open(os.path.join(SM, "p4_cal_table_percell.json"), "w"), indent=1)
