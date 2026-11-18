#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs9_legacy_refs.py (v9) - READ-ONLY. For old folders that live code or result files still mention, shows WHAT is mentioned:
which files mention them, which exact paths, whether those paths exist, and how big they are. Deletes/moves nothing.
Usage: python qs9_legacy_refs.py --root C:\\Projects --scan-roots R1 R2 --names NAME1 NAME2 ... --csv out.csv"""
import os, re, sys, csv, argparse

SKIP = {"__pycache__", ".git", "graveyard", "quarantine", "backup", "site-packages", "logs"}
EXT = {".py", ".ps1", ".bat", ".cmd", ".json", ".cfg", ".toml", ".yaml", ".yml", ".ini", ".md", ".txt", ".csv"}
SEP = r"(?:\\\\|\\|/)+"

def rd(p):
    try:
        if os.path.getsize(p) > 3 * 1024 * 1024:
            return ""
        raw = open(p, "rb").read()
    except OSError:
        return ""
    return raw.decode("utf-16", "ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "ignore")

def norm(p):
    return re.sub(r"(\\\\|\\|/)+", lambda m: os.sep, p)

def size_of(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    t = 0
    for r, d, f in os.walk(p):
        for x in f:
            try:
                t += os.path.getsize(os.path.join(r, x))
            except OSError:
                pass
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scan-roots", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    pats = {n: re.compile(r"(?:[A-Za-z]:" + SEP + r")?(?:[^\"'\s<>|,{}\[\]]*?" + SEP + r")?" + re.escape(n) + r"(?![A-Za-z0-9_])(?:" + SEP + r"[^\"'\s<>|,{}\[\]]*)?", re.I) for n in a.names}
    refs = {n: {} for n in a.names}      # name -> path -> set(files)
    files = {n: {} for n in a.names}     # name -> top folder -> count
    for sr in a.scan_roots:
        for dp, dn, fn in os.walk(sr):
            dn[:] = [d for d in dn if d not in SKIP]
            for x in fn:
                if os.path.splitext(x)[1].lower() not in EXT:
                    continue
                p = os.path.join(dp, x)
                txt = rd(p)
                if not txt:
                    continue
                top = os.path.relpath(p, sr).split(os.sep)[0]
                for n, rx in pats.items():
                    hits = rx.findall(txt) if n.lower() in txt.lower() else []
                    for h in hits:
                        refs[n].setdefault(norm(h), set()).add(p)
                    if hits:
                        files[n][top] = files[n].get(top, 0) + 1
    rows = []
    for n in a.names:
        base = os.path.join(a.root, n)
        print("FOLDER %s  exists=%s  size=%.1f MB" % (n, os.path.isdir(base), size_of(base) / 1048576 if os.path.isdir(base) else 0))
        print("  mentioned in files under: " + (", ".join("%s(%d)" % kv for kv in sorted(files[n].items(), key=lambda kv: -kv[1])[:6]) or "nowhere"))
        ex = dang = 0
        exts = {}
        tot = 0
        for path in sorted(refs[n]):
            # map the mentioned path onto the real folder under root, whatever drive/prefix was written
            i = path.lower().find(n.lower())
            real = os.path.join(a.root, path[i:]) if i >= 0 else path
            e = os.path.exists(real)
            sz = size_of(real) if e else 0
            if e:
                ex += 1
                tot += sz
                exts[os.path.splitext(real)[1].lower() or "<dir>"] = exts.get(os.path.splitext(real)[1].lower() or "<dir>", 0) + 1
            else:
                dang += 1
            rows.append([n, path, int(e), sz, len(refs[n][path]), sorted(refs[n][path])[0]])
        print("  distinct paths mentioned=%d  exist=%d  missing=%d  existing bytes=%.1f MB  types=%s" % (len(refs[n]), ex, dang, tot / 1048576, dict(sorted(exts.items(), key=lambda kv: -kv[1])[:5])))
        for path in sorted(refs[n])[:4]:
            print("    e.g. " + path[-110:])
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder", "mentioned_path", "exists", "bytes", "files_mentioning", "first_file"])
        w.writerows(rows)
    print("QS9-LEGACY-REFS-DONE")

if __name__ == "__main__":
    main()
