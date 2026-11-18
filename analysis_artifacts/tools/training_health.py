"""PHASE D D1 - tools\training_health.py (WP2 tooling, CPU only, report only).
Per config over finished seeds: final validation score, best validation score,
steps to reach 90 percent of the best, and COLLAPSE_CANDIDATE when final
validation score < 0.6 x the median of its config. Reads qstudy\runs\<cfg>\seed_<n>\train_summary.json.
Run for Z1 (9 configs x 10 seeds, seeds 101-110) and the Z2A arms. Never retrains or drops.
"""
import json, os, statistics, sys

STUDY = r"C:\Projects\MiniProject_Sem4\qstudy"
RUNS = os.path.join(STUDY, "runs")
OUT = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\results\training_health_report.json"

Z1 = [("rl__rabi__plain",), ("rl__rabi__gru8",), ("rl__dither__plain",), ("rl__dither__filter",),
      ("rl__dither__DR",), ("rl__dither__DRcurr",), ("rl__dither__full",),
      ("rl__ramsey_iq__plain",), ("rl__ramsey_iq__gru8",)]
Z1_SEEDS = list(range(101, 111))
Z2A = ["s15__E1", "s15__E4", "s15__FTB_50000", "s15__FTB_200000", "s15__FTB_500000",
       "s15__FTB_1000000", "s15__FTA_200000"]
Z2A_SEEDS = list(range(200, 205))


def read(cfg, seed):
    p = os.path.join(RUNS, cfg, "seed_%d" % seed, "train_summary.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def steps_to_90(vh, best):
    if best <= 0:
        return None
    thr = 0.9 * best
    for e in vh:
        if e.get("score", 0) >= thr:
            return e.get("step")
    return None


def main():
    groups = [("Z1", Z1, Z1_SEEDS)] + [("Z2A", [(c,) for c in Z2A], Z2A_SEEDS)]
    report = {"groups": [], "collapse_candidates": []}
    lines = []
    for gname, cfgs, seeds in groups:
        for (cfg,) in cfgs:
            rows = []
            for s in seeds:
                t = read(cfg, s)
                if t is None:
                    continue
                vh = t.get("val_history", [])
                best = t.get("best_val")
                final = vh[-1]["score"] if vh else None
                rows.append({"seed": s, "best_val": best, "final_val": final,
                             "steps_to_90pct_best": steps_to_90(vh, best),
                             "n_val": len(vh)})
            if not rows:
                continue
            finals = [r["final_val"] for r in rows if r["final_val"] is not None]
            med = statistics.median(finals) if finals else None
            for r in rows:
                flag = None
                if med is not None and r["final_val"] is not None and r["final_val"] < 0.6 * med:
                    flag = "COLLAPSE_CANDIDATE"
                    report["collapse_candidates"].append({"group": gname, "cfg": cfg, "seed": r["seed"],
                                                          "final_val": r["final_val"], "config_median": med})
                r["collapse_candidate"] = flag
                r["config_median_final"] = med
            report["groups"].append({"group": gname, "cfg": cfg, "seeds": len(rows), "rows": rows})
            lines.append("%s %-28s seeds=%d median_final=%s collapses=%d" % (
                gname, cfg, len(rows),
                ("%.4f" % med) if med is not None else "NA",
                sum(1 for r in rows if r["collapse_candidate"])))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\n".join(lines))
    print("report -> %s (collapse_candidates=%d)" % (OUT, len(report["collapse_candidates"])))


if __name__ == "__main__":
    main()
