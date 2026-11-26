#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs10_triage.py (v10) - after a restart: list FAILED/BLOCKED jobs and incomplete training folders; with --apply re-queue them.
Rules enforced by the code (not by the operator):
  * only job ids starting with --prefix (default z1) are touched; a RUNNING job is never touched;
  * a FAILED job is re-queued at most --max-requeues times in total (ledger audit\\requeue_ledger.json); BLOCKED jobs are re-queued freely
    (they only wait for a dependency);
  * an incomplete training folder (no train_summary.json) of a FAILED/BLOCKED training job is RENAMED to <name>.partial_<stamp>, never deleted;
  * re-queue = remove <id>.status.json and <id>.ok, exactly as earlier sessions did; the ids are appended to audit\\reset_ids.txt.
Usage: python qs10_triage.py --work WORK [--prefix z1] [--max-requeues 2] [--apply]"""
import os, sys, re, json, glob, time, argparse

def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--prefix", default="z1")
    ap.add_argument("--max-requeues", type=int, default=2)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    q = os.path.join(a.work, "queue")
    audit = os.path.join(a.work, "audit")
    os.makedirs(audit, exist_ok=True)
    lp = os.path.join(audit, "requeue_ledger.json")
    ledger = load(lp) if os.path.exists(lp) else {}
    rows, partial = [], []
    for jp in sorted(glob.glob(os.path.join(q, a.prefix + "*.job.json"))):
        base = jp[:-len(".job.json")]
        jid = os.path.basename(base)
        sp = base + ".status.json"
        try:
            st = load(sp) if os.path.exists(sp) else {}
            job = load(jp)
        except Exception:
            rows.append((jid, "UNREADABLE", 0, "CONTROLLER"))
            continue
        state = st.get("state", "PENDING")
        if state not in ("FAILED", "BLOCKED"):
            continue
        n = ledger.get(jid, 0)
        act = "REQUEUE" if state == "BLOCKED" else ("REQUEUE" if n < a.max_requeues else "GIVE_UP")
        rows.append((jid, state, st.get("attempts", 0), act, n))
        cwd = job.get("cwd") or ""
        for o in job.get("expected_outputs") or []:
            if o.lower().endswith("train_summary.json"):
                o = o if os.path.isabs(o) else os.path.join(cwd, o)
                d = os.path.dirname(o)
                if os.path.isdir(d) and not os.path.exists(o):
                    partial.append(d)
    print("TRIAGE failed_or_blocked=%d  incomplete_training_folders=%d  (prefix %s)" % (len(rows), len(partial), a.prefix))
    for r in rows[:40]:
        print("  %-42s %-8s att=%s requeued_before=%s -> %s" % (r[0][:42], r[1], r[2], r[4] if len(r) > 4 else "-", r[3]))
    for d in partial[:15]:
        print("  PARTIAL " + d)
    give = [r for r in rows if r[3] in ("GIVE_UP", "CONTROLLER")]
    if give:
        print("  !! NEEDS CONTROLLER: " + ", ".join(r[0] for r in give[:10]))
    if not a.apply:
        print("DRY RUN ONLY (add --apply). Nothing changed.")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    renamed = 0
    for d in partial:
        os.rename(d, d + ".partial_" + stamp)
        renamed += 1
    done = []
    for r in rows:
        if r[3] != "REQUEUE":
            continue
        base = os.path.join(q, r[0])
        try:
            cur = load(base + ".status.json").get("state") if os.path.exists(base + ".status.json") else "PENDING"
        except Exception:
            cur = "UNREADABLE"
        if cur == "RUNNING":
            continue
        for ext in (".status.json", ".ok"):
            try:
                os.remove(base + ext)
            except FileNotFoundError:
                pass
        if r[1] == "FAILED":
            ledger[r[0]] = ledger.get(r[0], 0) + 1
        done.append(r[0])
    json.dump(ledger, open(lp, "w"), indent=1)
    rp = os.path.join(audit, "reset_ids.txt")
    old = [x.strip() for x in open(rp, encoding="utf-8-sig")] if os.path.exists(rp) else []
    open(rp, "w", encoding="utf-8").write("\n".join(sorted(set(x for x in old + done if x))) + "\n")
    try:
        with open(os.path.join(a.work, "logs", "events.md"), "a", encoding="utf-8") as f:
            f.write("%s restart triage (qs10_triage): renamed %d incomplete training folders, re-queued %d jobs\n" % (time.strftime("%Y-%m-%d"), renamed, len(done)))
    except OSError:
        pass
    print("APPLIED renamed=%d requeued=%d" % (renamed, len(done)))
    print("TRIAGE-DONE")

if __name__ == "__main__":
    main()
