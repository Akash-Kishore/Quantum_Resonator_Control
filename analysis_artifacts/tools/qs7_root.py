#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs7_root.py (v7) - tidy the top-level projects folder (e.g. C:\\Projects).
  report : read-only table of every top-level item and a proposal.
  apply  : moves (never deletes) the approved legacy items into <root>\\_LEGACY_MiniProject_Sem4 (needs --confirm GO-ROOT).
  purge  : PERMANENTLY deletes only test-sandbox style folders (name contains sandbox, audit or fixrun) that are already inside
           <root>\\_LEGACY_MiniProject_Sem4 and that no script/config/pending job refers to. Writes a manifest first. Needs --confirm GO-PURGE.
           ARCHIVE, BACKUP and zip items are never purged by this command.
Rules: only items whose name starts with 'MiniProject_Sem4' can ever be proposed. The live folders, every QSTUDY_* file and
every other project (Birthday, Q-Quest, ...) are never touched. An item that a script/config/pending job refers to is KEPT.
Usage: python qs7_root.py report --root C:\\Projects --scan-roots STUDYROOT WORK --out root_report.csv
       python qs7_root.py apply  --root C:\\Projects --plan root_report.csv --confirm GO-ROOT"""
import os, sys, re, csv, json, time, shutil, argparse

LIVE_DEFAULT = ["MiniProject_Sem4", "MiniProject_Sem4_QSTUDY_WORK"]
CODE_EXT = {".py", ".ps1", ".bat", ".cmd", ".json", ".cfg", ".toml", ".yaml", ".yml", ".ini"}
DOC_EXT = {".md", ".txt"}
SKIP_DIRS = {"results", "runs", "runs_smoke", "results_smoke", "quarantine", "graveyard", "backup", "__pycache__", "logs", "smoke",
             ".git", "site-packages"}
LEGACY = "_LEGACY_MiniProject_Sem4"

def tree_size(p):
    if os.path.isfile(p):
        return os.path.getsize(p), 1
    tot, n = 0, 0
    for r, d, f in os.walk(p, onerror=lambda e: None):
        for x in f:
            try:
                tot += os.path.getsize(os.path.join(r, x))
                n += 1
            except OSError:
                pass
    return tot, n

def read_text(p):
    try:
        if os.path.getsize(p) > 2 * 1024 * 1024:
            return ""
        data = open(p, "rb").read()
    except OSError:
        return ""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", "ignore")
    return data.decode("utf-8", "ignore")

def job_is_done(path):
    if not path.endswith(".job.json"):
        return False
    sp = path[:-len(".job.json")] + ".status.json"
    try:
        with open(sp, encoding="utf-8-sig") as f:
            return json.load(f).get("state") == "DONE"
    except Exception:
        return False

def find_refs(cands, scan_roots, root):
    pats = {c: re.compile(re.escape(c) + r"(?![A-Za-z0-9_])", re.I) for c in cands}
    code = {c: [] for c in cands}
    doc = {c: 0 for c in cands}
    files = []
    for sr in scan_roots:
        for dp, dn, fn in os.walk(sr):
            dn[:] = [d for d in dn if d.lower() not in SKIP_DIRS]
            for n in fn:
                files.append(os.path.join(dp, n))
    for n in os.listdir(root):
        p = os.path.join(root, n)
        if os.path.isfile(p) and os.path.splitext(n)[1].lower() in DOC_EXT:
            files.append(p)
    for p in files:
        ext = os.path.splitext(p)[1].lower()
        if ext not in CODE_EXT and ext not in DOC_EXT:
            continue
        if ext in CODE_EXT and job_is_done(p):
            continue
        txt = read_text(p)
        if not txt:
            continue
        for c, rx in pats.items():
            if rx.search(txt):
                if ext in CODE_EXT:
                    code[c].append(p)
                else:
                    doc[c] += 1
    return code, doc

def cmd_report(a):
    root = a.root
    live = a.live or LIVE_DEFAULT
    items = sorted(os.listdir(root))
    cands = [n for n in items if n.lower().startswith("miniproject_sem4") and not (n in live and os.path.isdir(os.path.join(root, n)))]
    code, doc = find_refs(cands, a.scan_roots, root)
    rows = []
    for n in items:
        p = os.path.join(root, n)
        size, nf = tree_size(p)
        mt = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(p)))
        if n == LEGACY:
            cls, prop = "LEGACY_FOLDER", "KEEP"
        elif n in live and os.path.isdir(p):
            cls, prop = "LIVE", "KEEP"
        elif n.upper().startswith("QSTUDY_"):
            cls, prop = "PROMPT_FILE", "KEEP"
        elif n not in cands:
            cls, prop = "OTHER_PROJECT", "KEEP (not part of MiniProject_Sem4; never touched)"
        elif code[n]:
            cls, prop = "LEGACY_CANDIDATE", "KEEP_REFERENCED"
        else:
            cls, prop = "LEGACY_CANDIDATE", "MOVE_TO_LEGACY"
        rows.append(dict(name=n, kind="dir" if os.path.isdir(p) else "file", size_mb=round(size / 1048576, 1), files=nf,
                         modified=mt, cls=cls, code_refs=len(code.get(n, [])), doc_mentions=doc.get(n, 0), proposal=prop,
                         first_ref=(code[n][0] if code.get(n) else "")))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    mv = [r for r in rows if r["proposal"] == "MOVE_TO_LEGACY"]
    print("ROOT %s items=%d" % (root, len(rows)))
    for r in rows:
        print("  %-48s %-4s %9.1f MB %6d files %s  %-16s refs=%d/%d  -> %s" % (r["name"][:48], r["kind"], r["size_mb"], r["files"], r["modified"],
              r["cls"], r["code_refs"], r["doc_mentions"], r["proposal"][:28]))
    print("PROPOSED_MOVES %d  (%.1f MB)" % (len(mv), sum(r["size_mb"] for r in mv)))
    for r in rows:
        if r["proposal"] == "KEEP_REFERENCED":
            print("  KEPT because referenced: %s  by  %s" % (r["name"], r["first_ref"][-90:]))
    try:
        import shutil as sh
        du = sh.disk_usage(root)
        print("DISK free %.1f GB of %.1f GB" % (du.free / 1e9, du.total / 1e9))
    except Exception:
        pass
    print("ROOT-REPORT-DONE")

def cmd_apply(a):
    rows = list(csv.DictReader(open(a.plan, encoding="utf-8", newline="")))
    live = a.live or LIVE_DEFAULT
    mv = [r for r in rows if r["proposal"] == "MOVE_TO_LEGACY"]
    for r in mv:
        n = r["name"]
        if n in live or n.upper().startswith("QSTUDY_") or not n.lower().startswith("miniproject_sem4"):
            print("REFUSED: %s is not movable" % n)
            sys.exit(2)
        if not os.path.exists(os.path.join(a.root, n)):
            print("REFUSED: %s no longer exists" % n)
            sys.exit(2)
    print("PRECHECK moves=%d" % len(mv))
    if a.confirm != "GO-ROOT":
        print("DRY RUN ONLY (add --confirm GO-ROOT). Nothing changed.")
        for r in mv:
            print("  would move " + r["name"])
        return
    dst = os.path.join(a.root, LEGACY)
    os.makedirs(dst, exist_ok=True)
    lp = os.path.join(dst, "MOVE_LOG.csv")
    new = not os.path.exists(lp)
    moved, fails = 0, 0
    with open(lp, "a", newline="", encoding="utf-8") as lf:
        w = csv.writer(lf)
        if new:
            w.writerow(["when", "name", "from", "to", "size_mb"])
        for r in mv:
            s, d = os.path.join(a.root, r["name"]), os.path.join(dst, r["name"])
            if os.path.exists(d):
                print("  skipped (target exists): " + r["name"])
                continue
            try:
                shutil.move(s, d)
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), r["name"], s, d, r["size_mb"]])
                moved += 1
            except Exception as e:
                fails += 1
                print("  !! FAILED %s %s" % (r["name"], type(e).__name__))
    open(os.path.join(dst, "README_LEGACY.txt"), "w", encoding="utf-8").write(
        "Old MiniProject_Sem4 folders moved here by qs7_root.py. Nothing was deleted. See MOVE_LOG.csv. "
        "Delete this folder manually only after the paper is submitted.\n")
    print("MOVED %d FAILED %d into %s" % (moved, fails, dst))
    print("ROOT-APPLY-DONE")

def cmd_purge(a):
    import stat
    leg = os.path.join(a.root, LEGACY)
    if not os.path.isdir(leg):
        print("NO LEGACY FOLDER: " + leg)
        sys.exit(2)
    items = [n for n in sorted(os.listdir(leg)) if os.path.isdir(os.path.join(leg, n))
             and any(k in n.lower() for k in ("sandbox", "audit", "fixrun"))]
    code, doc = find_refs(items, a.scan_roots, a.root)
    ok = [n for n in items if not code[n]]
    for n in items:
        sz, nf = tree_size(os.path.join(leg, n))
        print("  %-48s %9.1f MB %7d files  %s" % (n[:48], sz / 1048576, nf, "PURGE" if n in ok else "KEPT (referenced by " + code[n][0][-60:] + ")"))
    print("PURGE_CANDIDATES %d" % len(ok))
    if a.confirm != "GO-PURGE":
        print("DRY RUN ONLY (add --confirm GO-PURGE). Nothing deleted.")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mp = os.path.join(leg, "PURGE_MANIFEST_%s.csv" % stamp)
    def unlock(func, path, exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    with open(mp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder", "relative_path", "bytes", "modified"])
        for n in ok:
            base = os.path.join(leg, n)
            for r, d, fl in os.walk(base):
                for x in fl:
                    p = os.path.join(r, x)
                    try:
                        w.writerow([n, os.path.relpath(p, base), os.path.getsize(p), time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))])
                    except OSError:
                        pass
    gone, fail = 0, 0
    for n in ok:
        shutil.rmtree(os.path.join(leg, n), onerror=unlock)
        if os.path.exists(os.path.join(leg, n)):
            fail += 1
            print("  !! could not fully delete " + n)
        else:
            gone += 1
    print("PURGED %d FAILED %d  manifest=%s" % (gone, fail, mp))
    print("ROOT-PURGE-DONE")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--root", required=True)
    r.add_argument("--live", nargs="*")
    r.add_argument("--scan-roots", nargs="+", required=True)
    r.add_argument("--out", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--root", required=True)
    p.add_argument("--live", nargs="*")
    p.add_argument("--plan", required=True)
    p.add_argument("--confirm", default="")
    q = sub.add_parser("purge")
    q.add_argument("--root", required=True)
    q.add_argument("--scan-roots", nargs="+", required=True)
    q.add_argument("--confirm", default="")
    a = ap.parse_args()
    {"report": cmd_report, "apply": cmd_apply, "purge": cmd_purge}[a.cmd](a)

if __name__ == "__main__":
    main()
