#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs8_compare.py (v8) - READ-ONLY. Compares two folders of JSON result files number by number.
Used to prove that cleaned code reproduces the as-run results. A leaf passes if abs diff <= --abs OR relative diff <= --rel;
it fails if it exceeds BOTH hard limits (--hard-abs and --hard-rel); in between it is a WARN.
Volatile keys (times, dates, hosts, versions, paths, ids) are ignored.
Usage: python qs8_compare.py DIR_A DIR_B [--match-after __] [--abs 0.005 --rel 0.005 --hard-abs 0.01 --hard-rel 0.01] [--csv out.csv]
Exit code 0 = no FAIL and no MISSING, 1 otherwise."""
import os, re, sys, csv, json, argparse

IGN = re.compile(r"^(wall_s|wall_time|wall_seconds|elapsed|elapsed_s|timestamp|created_at|date|host|hostname|started|finished|version|python_version|"
                 r"torch_version|numpy_version|pid|cmd|log|path|dir|file|sha256|hash|tag)$|(_path|_dir|_file|_sha256|_hash|_version|_timestamp|_host)$", re.I)
IGNORED = set()
EXTRA = None

def load(p):
    raw = open(p, "rb").read()
    t = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    return json.loads(t)

def flat(o, pre="", out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            if IGN.search(str(k)) or (EXTRA is not None and EXTRA.search(str(k))):
                IGNORED.add(str(k))
                continue
            flat(v, pre + "/" + str(k), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            flat(v, pre + "[%d]" % i, out)
    elif isinstance(o, bool) or o is None or isinstance(o, str):
        out[pre] = o
    elif isinstance(o, (int, float)):
        out[pre] = float(o)
    return out

MATCH = ""

def files(d):
    """relative name -> path. With --match-after TOKEN the part of the file name BEFORE the token is dropped
    (p8rl1__nom.json and p8rl2__nom.json both become nom.json), for runs whose names carry a run prefix."""
    res = {}
    for dp, dn, fn in os.walk(d):
        for n in fn:
            if n.lower().endswith(".json") and not n.endswith(".DONE.json"):
                key = n.split(MATCH, 1)[1] if MATCH and MATCH in n else n
                res[os.path.relpath(os.path.join(dp, key), d)] = os.path.join(dp, n)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--abs", type=float, default=0.005, dest="tabs")
    ap.add_argument("--rel", type=float, default=0.005, dest="trel")
    ap.add_argument("--hard-abs", type=float, default=0.01, dest="habs")
    ap.add_argument("--hard-rel", type=float, default=0.01, dest="hrel")
    ap.add_argument("--csv", default="")
    ap.add_argument("--ignore-extra", default="", help="extra regex of key names to ignore")
    ap.add_argument("--match-after", default="", help="match files by the part of the name after this token, e.g. __")
    g = ap.parse_args()
    global EXTRA, MATCH
    EXTRA = re.compile(g.ignore_extra) if g.ignore_extra else None
    MATCH = g.match_after
    fa, fb = files(g.a), files(g.b)
    rows, missing, worst = [], [], []
    npass = nwarn = nfail = leaves = 0
    gmax = 0.0
    for rel in sorted(fa):
        if rel not in fb:
            missing.append(rel)
            continue
        try:
            xa, xb = flat(load(fa[rel])), flat(load(fb[rel]))
        except Exception as e:
            rows.append((rel, "", "UNREADABLE", type(e).__name__, "", ""))
            nfail += 1
            continue
        fmax, fkey = 0.0, ""
        for k in sorted(set(xa) | set(xb)):
            leaves += 1
            va, vb = xa.get(k, "<absent>"), xb.get(k, "<absent>")
            if isinstance(va, float) and isinstance(vb, float):
                d = abs(va - vb)
                r = d / max(abs(va), abs(vb), 1e-300)
                if d > fmax:
                    fmax, fkey = d, k
                if d <= g.tabs or r <= g.trel:
                    npass += 1
                elif d <= g.habs or r <= g.hrel:
                    nwarn += 1
                    rows.append((rel, k, "WARN", va, vb, d))
                else:
                    nfail += 1
                    rows.append((rel, k, "FAIL", va, vb, d))
            elif va != vb:
                nfail += 1
                rows.append((rel, k, "FAIL_TEXT", va, vb, ""))
            else:
                npass += 1
        gmax = max(gmax, fmax)
        worst.append((fmax, rel, fkey))
    if g.csv:
        with open(g.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "key", "verdict", "a", "b", "abs_diff"])
            w.writerows(rows)
            for m in missing:
                w.writerow([m, "", "MISSING_IN_B", "", "", ""])
    extra = sorted(set(fb) - set(fa))
    print("COMPARE files_a=%d files_b=%d matched=%d missing_in_b=%d extra_in_b=%d" % (len(fa), len(fb), len(fa) - len(missing), len(missing), len(extra)))
    print("LEAVES compared=%d pass=%d warn=%d fail=%d   max_abs_diff=%.3g" % (leaves, npass, nwarn, nfail, gmax))
    print("EXACT MATCH" if gmax == 0 and not missing and nfail == 0 else "NOT EXACT")
    for fm, rel, k in sorted(worst, reverse=True)[:8]:
        if fm > 0:
            print("  worst %.3g  %s %s" % (fm, rel[-70:], k[-40:]))
    for m in missing[:5]:
        print("  MISSING_IN_B " + m[-90:])
    for r in [x for x in rows if x[2].startswith("FAIL")][:8]:
        print("  FAIL %s %s a=%s b=%s" % (r[0][-50:], r[1][-30:], r[3], r[4]))
    print("IGNORED_KEYS: " + (", ".join(sorted(IGNORED)[:15]) or "none"))
    print("QS8-COMPARE-DONE")
    sys.exit(0 if (nfail == 0 and not missing) else 1)

if __name__ == "__main__":
    main()
