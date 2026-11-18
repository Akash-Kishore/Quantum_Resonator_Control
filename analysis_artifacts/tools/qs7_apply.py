#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs7_apply.py (v7) - executes a plan made by qs7_plan.py. DRY RUN unless --confirm GO-DELETE is given.
GRAVEYARD rows are MOVED (never erased) to WORK\\graveyard\\<stamp>\\<original path>; the user purges that folder later.
RESET_JOB rows re-queue the same job id (removes <id>.status.json and <id>.ok), exactly as earlier sessions did.
Nothing is changed unless ALL pre-checks pass.
Usage: python qs7_apply.py plan.csv --work WORK --study STUDY [--confirm GO-DELETE]"""
import os, sys, csv, time, shutil, hashlib, argparse

ALLOW = ["runs", "results", "runs_smoke", "results_smoke", "smoke", "smoke_cfgs"]
FORBID = ["prereg", "/configs/", "/tools/", "/queue/", "/graveyard/", "/quarantine/", "/backup/", "/logs/"]

def N(p):
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))

def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def newest_mtime(p):
    """newest modification time of a file, or of the files inside a folder (the folder's own time is ignored
    unless it holds no files)."""
    if os.path.isfile(p):
        return os.path.getmtime(p)
    m = None
    for r, d, f in os.walk(p):
        for n in f:
            try:
                t = os.path.getmtime(os.path.join(r, n))
            except OSError:
                continue
            m = t if m is None else max(m, t)
    return m if m is not None else os.path.getmtime(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--work", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--confirm", default="")
    ap.add_argument("--recent-min", type=float, default=15)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.plan, encoding="utf-8", newline="")))
    if any(r["action"] in ("BLOCK", "NEEDS_CONTROLLER") for r in rows):
        print("REFUSED: plan contains BLOCK or NEEDS_CONTROLLER rows. Controller decision required.")
        sys.exit(2)
    allow = []
    for x in ALLOW:
        allow += [N(os.path.join(a.study, x)) + os.sep, N(os.path.join(a.work, x)) + os.sep]
    gy = [r for r in rows if r["action"] in ("GRAVEYARD", "GRAVEYARD_DIR")]
    rs = [r for r in rows if r["action"] == "RESET_JOB"]
    bad, gone = [], 0
    now = time.time()
    for r in gy:
        t = r["target"]
        nt = N(t)
        ntl = nt.replace("\\", "/").lower() + "/"
        if not os.path.exists(t):
            gone += 1
            continue
        if not any(nt.startswith(x) for x in allow):
            bad.append("NOT_ALLOWED_PATH " + t)
        if any(f in ntl for f in FORBID) or nt.lower().endswith((".db", ".sqlite", ".sha256")):
            bad.append("FORBIDDEN_PATH " + t)
        if now - newest_mtime(t) < a.recent_min * 60:
            bad.append("RECENTLY_MODIFIED " + t)
        if r.get("sha256") and os.path.isfile(t) and sha256_of(t) != r["sha256"]:
            bad.append("CHANGED_SINCE_SCAN " + t)
    for r in rs:
        jp = os.path.join(a.work, "queue", r["target"] + ".job.json")
        if not os.path.exists(jp):
            bad.append("JOB_FILE_MISSING " + r["target"])
    print("PRECHECK graveyard_rows=%d (already_gone=%d) reset_rows=%d problems=%d" % (len(gy), gone, len(rs), len(bad)))
    for b in bad[:30]:
        print("  !! " + b[-130:])
    if bad:
        print("REFUSED: fix the problems above; nothing was changed.")
        sys.exit(2)
    if a.confirm != "GO-DELETE":
        print("DRY RUN ONLY (add --confirm GO-DELETE to execute). Nothing changed.")
        for r in gy[:10]:
            print("  would move  " + r["target"][-110:])
        for r in rs[:10]:
            print("  would reset " + r["target"])
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    gdir = os.path.join(a.work, "graveyard", stamp)
    os.makedirs(gdir, exist_ok=True)
    logp = os.path.join(gdir, "GRAVEYARD_LOG.csv")
    moved, fails = 0, 0
    with open(logp, "w", newline="", encoding="utf-8") as lf:
        w = csv.writer(lf)
        w.writerow(["src", "dst", "kind", "bytes_or_files", "sha256", "reason"])
        for r in gy:
            t = r["target"]
            if not os.path.exists(t):
                continue
            rel = os.path.splitdrive(os.path.abspath(t))[1].lstrip("\\/")
            dst = os.path.join(gdir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isdir(t):
                nfiles = sum(len(f) for _, _, f in os.walk(t))
                info, sh = "files=%d" % nfiles, ""
            else:
                info = str(os.path.getsize(t))
                sh = sha256_of(t) if os.path.getsize(t) < 200 * 1024 * 1024 else ""
            try:
                shutil.move(t, dst)
                ok = (not os.path.exists(t)) and os.path.exists(dst)
            except Exception as e:
                ok = False
                print("  !! MOVE FAILED %s %s" % (t[-100:], type(e).__name__))
            if ok:
                moved += 1
                w.writerow([t, dst, r["action"], info, sh, r["reason"]])
            else:
                fails += 1
    print("MOVED %d  FAILED %d  graveyard=%s" % (moved, fails, gdir))
    if fails:
        print("STOP: a move failed; job resets were NOT done. Report to controller.")
        sys.exit(2)
    done = []
    for r in rs:
        base = os.path.join(a.work, "queue", r["target"])
        sp = base + ".status.json"
        cur = None
        if os.path.exists(sp):
            try:
                import json
                with open(sp, encoding="utf-8-sig") as f:
                    cur = json.load(f).get("state")
            except Exception:
                cur = "UNREADABLE"
        if cur == "RUNNING":
            print("  skipped (RUNNING now): " + r["target"])
            continue
        for ext in (".status.json", ".ok"):
            try:
                os.remove(base + ext)
            except FileNotFoundError:
                pass
        done.append(r["target"])
    ev = os.path.join(a.work, "logs", "events.md")
    try:
        with open(ev, "a", encoding="utf-8") as f:
            f.write("%s INFRASTRUCTURE REGENERATION (restart damage, user-approved GO-DELETE): moved %d items to graveyard %s; re-queued %d jobs (same ids): %s\n"
                    % (time.strftime("%Y-%m-%d"), moved, stamp, len(done), ",".join(done[:25]) + ("..." if len(done) > 25 else "")))
    except OSError:
        pass
    print("RESET_JOBS %d" % len(done))
    print("APPLY-DONE")

if __name__ == "__main__":
    main()
