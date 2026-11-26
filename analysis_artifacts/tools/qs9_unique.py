#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs9_unique.py (v9) - READ-ONLY. Is a folder redundant? Lists every file of --candidate whose CONTENT (size + sha256) exists nowhere in --against.
UNIQUE files are the only ones a deletion could lose. Deletes/moves nothing.
Usage: python qs9_unique.py --candidate C:\\Projects\\OLD --against C:\\Projects\\LIVE C:\\Projects\\OTHER --skip .git .opencode --csv unique.csv"""
import os, sys, csv, hashlib, argparse, time

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()

def walk(root, skip):
    for dp, dn, fn in os.walk(root, onerror=lambda e: None):
        dn[:] = [d for d in dn if d not in skip]
        for n in fn:
            p = os.path.join(dp, n)
            try:
                yield p, os.path.getsize(p)
            except OSError:
                pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--against", nargs="+", required=True)
    ap.add_argument("--skip", nargs="*", default=[".git", ".opencode", "__pycache__"])
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    t0 = time.time()
    cand = [(p, s) for p, s in walk(a.candidate, set(a.skip)) if s > 0]
    sizes = {s for _, s in cand}
    print("CANDIDATE files=%d  bytes=%.1f MB  (skipped folders: %s)" % (len(cand), sum(s for _, s in cand) / 1048576, ",".join(a.skip)))
    known = set()
    nagainst = 0
    for root in a.against:
        for p, s in walk(root, set(a.skip)):
            if s in sizes and not os.path.normcase(p).startswith(os.path.normcase(a.candidate) + os.sep):
                nagainst += 1
                try:
                    known.add((s, sha(p)))
                except OSError:
                    pass
    print("COMPARED against %d files with a matching size" % nagainst)
    uniq = []
    for p, s in cand:
        try:
            if (s, sha(p)) not in known:
                uniq.append((p, s))
        except OSError:
            uniq.append((p, s))
    ub = sum(s for _, s in uniq)
    print("UNIQUE files=%d  bytes=%.1f MB   redundant files=%d" % (len(uniq), ub / 1048576, len(cand) - len(uniq)))
    top = {}
    for p, s in uniq:
        k = os.path.relpath(p, a.candidate).split(os.sep)[0]
        top[k] = top.get(k, [0, 0])
        top[k][0] += 1
        top[k][1] += s
    for k, v in sorted(top.items(), key=lambda kv: -kv[1][1])[:12]:
        print("  unique in %-34s %5d files %9.2f MB" % (k[:34], v[0], v[1] / 1048576))
    for p, s in sorted(uniq, key=lambda x: -x[1])[:15]:
        print("  - %s (%.2f MB)" % (p[-100:], s / 1048576))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "bytes"])
            w.writerows(uniq)
    print("VERDICT: " + ("REDUNDANT (every file exists elsewhere)" if not uniq else "NOT REDUNDANT (%d unique files)" % len(uniq)))
    print("QS9-UNIQUE-DONE (%.0f s)" % (time.time() - t0))

if __name__ == "__main__":
    main()
