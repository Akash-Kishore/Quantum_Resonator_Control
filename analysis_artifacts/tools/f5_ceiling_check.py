"""M2 F5: oracle ceiling check on the 4 val cells x 3 probes.
- nominal ctrl:oracle lock_w1000 must be 0.954 +- 0.02 (Part 1 I5);
- no method may exceed the oracle of the same cell by more than 0.02 (else ANOMALY)."""
import json, os, re, sqlite3, statistics

S = r"C:\Projects\MiniProject_Sem4\qstudy"
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
VAL_CONDS = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]

oracle = {}
for probe in ("rabi", "dither", "ramsey_iq"):
    for cond in VAL_CONDS:
        p = os.path.join(S, "results", "val", "ctrl_oracle__%s__%s.json" % (probe, cond))
        j = json.load(open(p, encoding="utf-8"))
        oracle[(probe, cond)] = j["lock_w1000_mean"]

print("ctrl:oracle measured rows (lock_w1000):")
for probe in ("rabi", "dither", "ramsey_iq"):
    print("  %-10s %s" % (probe, " ".join("%.4f" % oracle[(probe, c)] for c in VAL_CONDS)))

bad = [(p, c, v) for (p, c), v in oracle.items() if c == "nom" and abs(v - 0.954) > 0.02]
print("I5 nominal check (0.954 +- 0.02):", "PASS" if not bad else "FAIL %s" % bad)

# per-(method-family, probe, cell) mean over seeds from result files
con = sqlite3.connect(os.path.join(S, "db", "qstudy.sqlite"))
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute("SELECT * FROM evals WHERE split='val'")]
con.close()

viol, table = [], {}
for r in rows:
    cond = r["cond_id"]
    if cond not in VAL_CONDS or r["lock_w1000_mean"] is None:
        continue
    npz = r["npz_path"] or ""
    p = npz if os.path.isabs(npz) else os.path.join(S, npz)
    if "_invalid" in p.lower() or not os.path.exists(p[:-4] + ".json" if p.endswith(".npz") else p):
        continue
    m = r["method"]
    if m == "ctrl:oracle":
        continue
    md = r["model_dir"] or ""
    cfgm = re.search(r"runs[/\\](.+?)[/\\]seed_\d+$", md)
    fam = cfgm.group(1) if cfgm else (m if m.startswith("ctrl:") else m)
    probe = (r["method"] and None)
    j = json.load(open(p[:-4] + ".json" if p.endswith(".npz") else p, encoding="utf-8"))
    probe = (j.get("cfg") or {}).get("probe")
    key = (fam, probe, cond)
    table.setdefault(key, []).append(r["lock_w1000_mean"])

for (fam, probe, cond), vals in sorted(table.items()):
    mean = statistics.mean(vals)
    o = oracle.get((probe, cond))
    if o is not None and mean > o + 0.02:
        viol.append((fam, probe, cond, round(mean, 4), round(o, 4)))
print("ceiling rule (no method > oracle + 0.02 in the same cell):",
      "PASS" if not viol else "ANOMALY %s" % viol)
json.dump(dict(oracle={"%s|%s" % k: v for k, v in oracle.items()},
               i5_pass=not bad, ceiling_pass=not viol, violations=viol),
          open(os.path.join(W, "results", "f5_oracle_check.json"), "w"), indent=1)
