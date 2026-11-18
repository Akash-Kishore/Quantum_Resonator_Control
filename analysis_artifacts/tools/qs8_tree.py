#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs8_tree.py (v8) - READ-ONLY. Size, file count and newest file date of every top-level folder of the given roots.
Usage: python qs8_tree.py --roots R1 R2 [--md out.md]"""
import os, sys, time, argparse

def stat(p):
    tot = n = 0
    newest = 0.0
    for r, d, f in os.walk(p, onerror=lambda e: None):
        for x in f:
            try:
                s = os.stat(os.path.join(r, x))
            except OSError:
                continue
            tot += s.st_size
            n += 1
            newest = max(newest, s.st_mtime)
    return tot, n, newest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    L = ["| root | folder | MB | files | newest file |", "|---|---|---|---|---|"]
    for root in a.roots:
        loose = [x for x in os.listdir(root) if os.path.isfile(os.path.join(root, x))]
        lt = sum(os.path.getsize(os.path.join(root, x)) for x in loose)
        L.append("| %s | (loose files: %d) | %.1f | %d | |" % (root, len(loose), lt / 1048576, len(loose)))
        for d in sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))):
            t, n, m = stat(os.path.join(root, d))
            L.append("| %s | %s | %.1f | %d | %s |" % (root, d, t / 1048576, n, time.strftime("%Y-%m-%d %H:%M", time.localtime(m)) if m else "-"))
    print("\n".join(L[:60]))
    if a.md:
        open(a.md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("QS8-TREE-DONE")

if __name__ == "__main__":
    main()
