import json, csv, os
S = r"C:\Projects\MiniProject_Sem4\qstudy"
T = os.path.join(S, "results", "tables")
os.makedirs(T, exist_ok=True)
V = os.path.join(S, "results", "validation")

j1 = json.load(open(V + "\\v1_expected_probability.json"))
rows = sorted(j1["rows"], key=lambda r: -abs(r["p_aer"] - r["p_sur"]))
with open(T + "\\v1_expected_probability_worst.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "noise", "delta", "p_aer", "p_sur", "tol", "absdiff", "pass"])
    for r in rows[:8]:
        w.writerow([r["kind"], r["noise"], r["delta"], "%.5f" % r["p_aer"], "%.5f" % r["p_sur"],
                    "%.5f" % r["tol"], "%.5f" % abs(r["p_aer"] - r["p_sur"]), int(r["pass_"])])
    w.writerow(["TOTAL_CELLS", j1["n_cells"], "", "N_PASS", j1["n_pass"], "", "", int(j1["pass"])])

j2 = json.load(open(V + "\\v2_ks_distribution.json"))
with open(T + "\\v2_ks_pvalues.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "delta", "shots", "ks_stat", "p_value", "pass"])
    for r in j2["rows"]:
        w.writerow([r["kind"], r["delta"], r["shots"], "%.4f" % r["ks_stat"], "%.4f" % r["p_value"], int(r["pass_"])])
    w.writerow(["TOTAL_CELLS", j2["n_cells"], "N_PASS", j2["n_pass"], "", ""])

j3 = json.load(open(V + "\\v3_closed_loop.json"))
j4 = json.load(open(V + "\\v4_speed.json"))
with open(T + "\\v3_v4_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test", "engine", "lock_w1000", "abs_difference", "pass"])
    w.writerow(["V3_hillclimb", "surrogate", "%.4f" % j3["lock_w1000"]["surrogate"], "", ""])
    w.writerow(["V3_hillclimb", "aer", "%.4f" % j3["lock_w1000"]["aer"], "%.4f" % j3["abs_difference"], int(j3["pass"])])
    w.writerow(["V4_speed", "surrogate", "%.0f" % j4["surrogate_steps_per_s"], "steps_per_s", ""])
    w.writerow(["V4_speed", "aer", "%.0f" % j4["aer_steps_per_s"], "steps_per_s", ""])

# legacy diag tables (Stage 5)
L = os.path.join(S, "results", "legacy")
jl = json.load(open(L + "\\legacy_diag.json"))
jr = json.load(open(L + "\\lineshape_replica.json"))
jf = json.load(open(L + "\\fisher_floors.json"))
with open(T + "\\legacy_diag_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    for k, v in jl.items():
        if isinstance(v, (int, float)):
            w.writerow([k, "%.4f" % v if isinstance(v, float) else v])
with open(T + "\\lineshape_replica.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["delta", "p_env", "p_replica", "absdiff"])
    for r in jr["rows"]:
        w.writerow([r["delta"], "%.5f" % r["p_env"], "%.5f" % r["p_replica"], "%.5f" % r["absdiff"]])
    w.writerow(["max_abs_difference", "", "", "%.5f" % jr["max_abs_difference"]])
with open(T + "\\fisher_floors.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["probe", "min_std_floor_hz"])
    for k, v in jf["min_std_floor_hz"].items():
        w.writerow([k, "%.2f" % v])
print("tables written to", T)
