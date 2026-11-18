#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs7_summaries.py (v7) - READ-ONLY. Tabulates every runs\\<config>\\seed_<n>\\train_summary.json without assuming key names.
Writes a CSV (all numeric scalar keys, flattened) and prints a compact per-group table. No interpretation.
Usage: python qs7_summaries.py --study STUDY --out train_summaries.csv"""
import os, re, sys, csv, json, argparse

SEED = re.compile(r"^seed_(\d+)$")

def flat(o, pre="", depth=0, out=None):
    out = {} if out is None else out
    if isinstance(o, dict) and depth < 3:
        for k, v in o.items():
            flat(v, (pre + "." if pre else "") + str(k), depth + 1, out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out[pre] = float(o)
    return out

def group(seed):
    if 0 <= seed <= 4:
        return "exploratory_0-4"
    if seed == 100:
        return "tuning_100"
    if 101 <= seed <= 110:
        return "confirmatory_101-110"
    if 200 <= seed <= 204:
        return "transfer_200-204"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = os.path.join(a.study, "runs")
    recs, bad = [], []
    for dp, dn, fn in os.walk(runs):
        m = SEED.match(os.path.basename(dp))
        if m and "train_summary.json" in fn:
            p = os.path.join(dp, "train_summary.json")
            try:
                with open(p, "rb") as f:
                    raw = f.read()
                txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
                o = json.loads(txt)
            except Exception:
                bad.append(p)
                continue
            seed = int(m.group(1))
            rec = dict(config=os.path.basename(os.path.dirname(dp)), seed=seed, group=group(seed))
            for k, v in flat(o).items():
                rec["summary_" + k if k in rec else k] = v   # a key named seed/config/group inside the file must not overwrite ours
            recs.append(rec)
    keycount = {}
    for r in recs:
        for k in r:
            if k not in ("config", "seed", "group") and not k.startswith("summary_"):
                keycount[k] = keycount.get(k, 0) + 1
    keys = sorted(keycount, key=lambda k: (-keycount[k], k))[:60]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "seed", "group"] + keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(recs, key=lambda r: (r["group"], r["config"], r["seed"])):
            w.writerow(r)
    print("SUMMARIES runs_with_summary=%d unreadable=%d" % (len(recs), len(bad)))
    for p in bad[:5]:
        print("  unreadable: " + p[-100:])
    top = keys[:4]
    print("KEYS_MOST_COMMON: " + ", ".join("%s(%d)" % (k, keycount[k]) for k in keys[:8]))
    agg = {}
    for r in recs:
        agg.setdefault((r["group"], r["config"]), []).append(r)
    for (g, c) in sorted(agg):
        rs = agg[(g, c)]
        line = "  %-22s %-34s n=%-2d seeds=%s" % (g, c[:34], len(rs), ",".join(str(x["seed"]) for x in sorted(rs, key=lambda x: x["seed"]))[:30])
        for k in top:
            vals = [x[k] for x in rs if k in x]
            if vals:
                line += " | %s mean=%.4g min=%.4g max=%.4g" % (k.split(".")[-1][:14], sum(vals) / len(vals), min(vals), max(vals))
        print(line[:230])
    print("SUMMARIES-DONE")

if __name__ == "__main__":
    main()
