import json, os, datetime
Q = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\queue"
def wall(jid):
    s = json.load(open(os.path.join(Q, jid + ".status.json"), encoding="utf-8"))
    f = datetime.datetime.fromisoformat(s["finished"]); t = datetime.datetime.fromisoformat(s["started"])
    return (f - t).total_seconds(), s["state"]
rows = {}
for cls, lo, hi, nlo, nhi in [("rl","p4_cal_rl_5","p4_cal_rl_300",5,300),
                             ("pid","p4_cal_pid_5","p4_cal_pid_300",5,300),
                             ("bayes","p4_cal_bayes_5","p4_cal_bayes_300",5,300),
                             ("aer","p4_cal_aer_2","p4_cal_aer_50",2,50)]:
    tlo, slo = wall(lo); thi, shi = wall(hi)
    assert slo == "DONE" and shi == "DONE", (lo, slo, hi, shi)
    pe = (thi - tlo) / (nhi - nlo)
    fx = tlo - nlo * pe
    rows[cls] = (tlo, thi, pe, fx)
    print("%-6s t%d=%7.1fs t%d=%7.1fs per_episode_s=%9.4f fixed_s=%8.2f" % (cls, nlo, tlo, nhi, thi, pe, fx))
RL = (9300, 1647000); NB = (936, 247500); BA = (624, 165000); AE = (160, 16000)
cpu_s = RL[0]*rows["rl"][3] + RL[1]*rows["rl"][2] + NB[0]*rows["pid"][3] + NB[1]*rows["pid"][2] + BA[0]*rows["bayes"][3] + BA[1]*rows["bayes"][2]
aer_s = AE[0]*rows["aer"][3] + AE[1]*rows["aer"][2]
print("cpu_hours=(%.0f)/4/3600=%.2f  aer_hours=(%.0f)/3600=%.2f  larger=%.2f" % (cpu_s, cpu_s/14400.0, aer_s, aer_s/3600.0, max(cpu_s/14400.0, aer_s/3600.0)))
json.dump({k: {"t_lo": v[0], "t_hi": v[1], "per_episode_s": v[2], "fixed_s": v[3]} for k, v in rows.items()},
          open(r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\smoke\p4_cal_table.json", "w"), indent=1)
