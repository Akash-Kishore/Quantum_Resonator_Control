#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs7_plan.py (v7) - READ-ONLY. Turns a scan CSV into a repair plan. Moves/deletes nothing.
Plan actions: GRAVEYARD (file), GRAVEYARD_DIR (run folder), RESET_JOB (job to re-run, same id),
BLOCK (needs controller), NEEDS_CONTROLLER (no job can regenerate it), REPORT_ONLY (not auto-repairable).
Usage: python qs7_plan.py --scan scan.csv --work WORK --study STUDY --out plan.csv"""
import os, sys, re, csv, json, glob, argparse

SEED_RE = re.compile(r"^seed_\d+$")

def N(p):
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))

def load_jobs(work):
    jobs = {}
    for p in sorted(glob.glob(os.path.join(work, "queue", "*.job.json"))):
        try:
            with open(p, encoding="utf-8-sig") as f:
                j = json.load(f)
        except Exception:
            continue
        base = p[:-len(".job.json")]
        jid = j.get("id") or os.path.basename(base)
        state, attempts = "PENDING", 0
        sp = base + ".status.json"
        if os.path.exists(sp):
            try:
                with open(sp, encoding="utf-8-sig") as f:
                    st = json.load(f)
                state, attempts = st.get("state", "PENDING"), st.get("attempts", 0)
            except Exception:
                state = "STATUS_UNREADABLE"
        jobs[jid] = dict(job=j, state=state, attempts=attempts, spath=sp, okpath=base + ".ok")
    return jobs

def job_outputs(info, study):
    cwd = info["job"].get("cwd") or study
    res = []
    for p in info["job"].get("expected_outputs") or []:
        res.append(N(p if os.path.isabs(p) else os.path.join(cwd, p)))
    return res

def resolve_listed(p, marker_path, study):
    base = os.path.basename(p.replace("\\", "/"))
    cands = []
    if os.path.isabs(p):
        cands.append(p)
    else:
        cands.append(os.path.join(os.path.dirname(marker_path), p))
        if study:
            cands.append(os.path.join(study, p))
    cands.append(os.path.join(os.path.dirname(marker_path), base))
    for c in cands:
        if os.path.exists(os.path.normpath(c)):
            return N(c)
    return None

def marker_index(roots, study):
    idx = {}
    for root in roots:
        for dp, dn, fn in os.walk(root):
            for n in fn:
                if n.endswith(".DONE.json"):
                    mp = os.path.join(dp, n)
                    try:
                        with open(mp, encoding="utf-8-sig") as f:
                            o = json.load(f)
                    except Exception:
                        idx[N(mp)] = set()
                        continue
                    s = set()
                    for e in (o.get("files") or []) if isinstance(o, dict) else []:
                        if isinstance(e, dict) and e.get("path"):
                            r = resolve_listed(e["path"], mp, study)
                            if r:
                                s.add(r)
                    idx[N(mp)] = s
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = load_jobs(a.work)
    print("JOBS_LOADED %d" % len(jobs))
    if not jobs:
        print("NO JOBS FOUND - STOP")
        sys.exit(3)
    out_map, rev = {}, {}
    for jid, info in jobs.items():
        for o in job_outputs(info, a.study):
            out_map.setdefault(o, set()).add(jid)
        for d in info["job"].get("depends_on") or []:
            rev.setdefault(d, set()).add(jid)
    mk = marker_index([os.path.join(a.study, "results"), os.path.join(a.work, "results")], a.study)
    rows = list(csv.DictReader(open(a.scan, encoding="utf-8", newline="")))
    cand = [r for r in rows if r["verdict"] == "CORRUPT" and r["in_window"] == "1" and r["deletable"] == "1"]
    other = [r for r in rows if r["verdict"] == "CORRUPT" and not (r["in_window"] == "1" and r["deletable"] == "1")]
    gy, producers, replaced, needs = {}, set(), set(), {}
    for r in cand:
        path, reason = r["path"], r["reason"]
        rel = [x for x in r.get("related", "").split(";") if x]
        seed_dir, cur = None, path
        while True:
            if SEED_RE.match(os.path.basename(cur)):
                seed_dir = cur
                break
            par = os.path.dirname(cur)
            if par == cur:
                break
            cur = par
        found = set()
        if seed_dir:
            gy[seed_dir] = ("GRAVEYARD_DIR", "RUN_DIR:" + reason, "")
            nd = N(seed_dir) + os.sep
            for jid, info in jobs.items():
                if any(o.startswith(nd) for o in job_outputs(info, a.study)):
                    found.add(jid)
                    replaced.add(jid)
        else:
            gy[path] = ("GRAVEYARD", reason, r.get("sha256", ""))
            targets = {N(path)} | {N(x) for x in rel}
            for x in rel:
                gy[x] = ("GRAVEYARD", "RELATED_TO:" + os.path.basename(path), "")
            for m, listed in mk.items():
                if targets & listed or m in targets:
                    if m != N(path):
                        gy[m] = ("GRAVEYARD", "MARKER_OF_CORRUPT", "")
                    targets.add(m)
            for t in targets:
                found |= out_map.get(t, set())
        if not found:
            needs[path] = "NO_PRODUCER_JOB"
        producers |= found
    resets, blocks = {}, {}
    for jid in sorted(producers):
        st = jobs[jid]["state"]
        if st == "RUNNING":
            blocks[jid] = "PRODUCER_RUNNING"
        else:
            resets[jid] = "PRODUCER_OF_CORRUPT(state=%s)" % st
    # dependents of replaced training runs
    seen, stack = set(), list(replaced)
    while stack:
        cur = stack.pop()
        for d in rev.get(cur, ()):
            if d in seen:
                continue
            seen.add(d)
            stack.append(d)
    for d in sorted(seen):
        st = jobs[d]["state"]
        if st == "RUNNING":
            blocks[d] = "DEPENDENT_RUNNING_ON_REPLACED_MODEL"
            continue
        if st == "PENDING":
            continue
        for o in job_outputs(jobs[d], a.study):
            if os.path.exists(o):
                gy.setdefault(o, ("GRAVEYARD", "DEPENDENT_OF_REPLACED_MODEL", ""))
                for t in mk.get(o, ()):
                    gy.setdefault(t, ("GRAVEYARD", "DEPENDENT_OF_REPLACED_MODEL", ""))
        if st in ("DONE", "FAILED", "BLOCKED"):
            resets.setdefault(d, "DEPENDENT_OF_REPLACED_MODEL(state=%s)" % st)
    dirs = [N(k) + os.sep for k, v in gy.items() if v[0] == "GRAVEYARD_DIR"]
    out_rows = []
    for k in sorted(gy):
        if gy[k][0] == "GRAVEYARD" and any(N(k).startswith(d) for d in dirs):
            continue
        out_rows.append((gy[k][0], k, gy[k][1], gy[k][2], ""))
    for jid in sorted(resets):
        out_rows.append(("RESET_JOB", jid, resets[jid], "", jobs[jid]["state"]))
    for jid in sorted(blocks):
        out_rows.append(("BLOCK", jid, blocks[jid], "", jobs[jid]["state"]))
    for p in sorted(needs):
        out_rows.append(("NEEDS_CONTROLLER", p, needs[p], "", ""))
    for r in other:
        out_rows.append(("REPORT_ONLY", r["path"], "%s %s window=%s" % (r["category"], r["reason"], r["in_window"]), "", ""))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "target", "reason", "sha256", "job_state"])
        w.writerows(out_rows)
    cnt = {}
    for r in out_rows:
        cnt[r[0]] = cnt.get(r[0], 0) + 1
    print("PLAN " + " ".join("%s=%d" % (k, cnt[k]) for k in sorted(cnt)) if cnt else "PLAN empty (nothing to repair)")
    for r in out_rows:
        if r[0] in ("BLOCK", "NEEDS_CONTROLLER"):
            print("  !! %s %s %s" % (r[0], r[1][-100:], r[2]))
    for r in out_rows:
        if r[0] == "REPORT_ONLY" and (" QUEUE " in " " + r[2] or " DB " in " " + r[2]):
            print("  !! REPORT_ONLY critical: %s %s" % (r[1][-100:], r[2]))
    for r in out_rows[:40]:
        print("  %-16s %-9s %s | %s" % (r[0], r[4], r[1][-95:], r[2][:45]))
    print("PLAN-DONE")

if __name__ == "__main__":
    main()
