#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs7_report.py (v7) - READ-ONLY. Builds the interim results report (markdown) from verified files only.
It never summarises sealed result values (only counts and integrity). Missing inputs are written as 'not available'.
Usage: python qs7_report.py --work WORK --study STUDY --out WORK\\audit\\RESULTS_REPORT_INTERIM.md
       [--scan scan.csv] [--plan plan.csv] [--summaries train_summaries.csv] [--root-report root_report.csv] [--tests unit_tests.txt] [--boots boots.txt]"""
import os, re, sys, csv, json, glob, time, argparse, platform

def read_csv(p):
    if not p or not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))

def tail_lines(p, n):
    if not p or not os.path.exists(p):
        return None
    raw = open(p, "rb").read()
    txt = raw.decode("utf-16", "ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "ignore")
    return txt.replace("\r", "").split("\n")[-n:]

def jobs_state(work):
    res = []
    for p in glob.glob(os.path.join(work, "queue", "*.job.json")):
        base = p[:-len(".job.json")]
        jid = os.path.basename(base)
        st, att, note = "PENDING", 0, ""
        sp = base + ".status.json"
        if os.path.exists(sp):
            try:
                o = json.load(open(sp, encoding="utf-8-sig"))
                st, att, note = o.get("state", "PENDING"), o.get("attempts", 0), (o.get("note") or o.get("reason") or "")
            except Exception:
                st = "UNREADABLE"
        res.append((jid, st, att, str(note)))
    return res

def parse_ts(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        for f in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return time.mktime(time.strptime(v[:19], f))
            except ValueError:
                pass
    return None

def read_text_any(p):
    raw = open(p, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", "ignore")
    return raw.decode("utf-8-sig", "ignore")

def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out

def version(pkg):
    try:
        from importlib.metadata import version as v
        return v(pkg)
    except Exception:
        return "not installed"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    for k in ("scan", "plan", "summaries", "root-report", "tests", "boots"):
        ap.add_argument("--" + k, default="")
    ap.add_argument("--appendix", nargs="*", default=[])
    a = ap.parse_args()
    L = []
    L += ["# Interim results report - QSTUDY / ARCS", "", "Generated: %s  |  Host: %s  |  Python %s" % (time.strftime("%Y-%m-%d %H:%M"), platform.node(), platform.python_version()), "",
          "> INTERIM. Built only from files checked in this session. Sealed (confirmatory) result values are NOT summarised here: they stay blinded until the controller releases GO-UNBLIND. Training-time validation scores are shown because they are not sealed data.", ""]
    L += ["Contents: 1 Environment | 2 Job queue and compute | 3 File integrity | 4 Training runs | 5 Sealed evaluation (blinded) | 6 Test suite | 7 Folder structure | 8 Anomalies and incidents | 9 What may not be concluded | 10 Paper checklist | Appendices", ""]
    # 1 environment
    L += ["## 1. Environment", ""]
    L += table(["package", "version"], [(p, version(p)) for p in ("numpy", "torch", "stable-baselines3", "sb3-contrib", "gymnasium", "qiskit", "qiskit-aer", "qutip", "scipy", "pandas")])
    bt = tail_lines(a.boots, 8)
    L += ["", "Restarts found (Windows event log): " + (" ; ".join(x for x in bt if x.strip()) if bt else "not available"), ""]
    # 2 queue
    jobs = jobs_state(a.work)
    L += ["## 2. Job queue", ""]
    cnt = {}
    for j in jobs:
        cnt[j[1]] = cnt.get(j[1], 0) + 1
    L += ["Total jobs: %d. " % len(jobs) + ", ".join("%s=%d" % (k, cnt[k]) for k in sorted(cnt)), ""]
    exp = [("z1a", "z1 training wave A (seeds 101-105)", 45), ("z1b", "z1 training wave B (seeds 106-110)", 45), ("z1e", "z1 RL evaluations", 270),
           ("z1v", "z1 baseline evaluations", 45), ("z1x", "z1 Aer evaluations", 26)]
    rows = []
    for pre, desc, n in exp:
        sel = [j for j in jobs if j[0].startswith(pre + "_")]
        d = sum(1 for j in sel if j[1] == "DONE")
        f = sum(1 for j in sel if j[1] in ("FAILED", "BLOCKED"))
        r = sum(1 for j in sel if j[1] == "RUNNING")
        rows.append((desc, "%d / %d" % (d, n), r, f, len(sel)))
    L += table(["group", "done / planned", "running", "failed+blocked", "jobs in queue"], rows)
    L += [""]
    # compute accounting from status files
    acc = {}
    for jid, st, att, note in jobs:
        if st != "DONE":
            continue
        sp = os.path.join(a.work, "queue", jid + ".status.json")
        jp = os.path.join(a.work, "queue", jid + ".job.json")
        try:
            so = json.load(open(sp, encoding="utf-8-sig"))
            lane = json.load(open(jp, encoding="utf-8-sig")).get("lane", "?")
        except Exception:
            continue
        t0 = parse_ts(so.get("started") or so.get("start"))
        t1 = parse_ts(so.get("finished") or so.get("end"))
        if t0 and t1 and t1 >= t0:
            g = jid.split("_")[0] if re.match(r"^z\d", jid) else "other"
            k = (g, lane)
            acc.setdefault(k, [0, 0.0])
            acc[k][0] += 1
            acc[k][1] += t1 - t0
    L += ["### Compute accounting (finished jobs, wall time summed over jobs)", ""]
    if acc:
        L += table(["group", "lane", "jobs", "wall hours"], [(k[0], k[1], v[0], "%.1f" % (v[1] / 3600)) for k, v in sorted(acc.items())])
    else:
        L += ["not available (start/finish times not readable in status files)"]
    L += [""]
    bad = [j for j in jobs if j[1] in ("FAILED", "BLOCKED", "UNREADABLE") and re.match(r"^(z|p)\d", j[0])]
    L += ["Failed/blocked current-study jobs: %d" % len(bad)]
    for j in bad[:20]:
        L.append("- %s %s attempts=%s %s" % (j[0], j[1], j[2], j[3][:80].replace("\n", " ")))
    L += [""]
    # 3 integrity
    sc = read_csv(a.scan)
    L += ["## 3. File integrity after the restart", ""]
    if sc is None:
        L += ["not available", ""]
    else:
        cor = [r for r in sc if r["verdict"] == "CORRUPT"]
        cw = [r for r in cor if r["in_window"] == "1"]
        L += ["Corrupt files found: %d (inside the restart window: %d, outside: %d)." % (len(cor), len(cw), len(cor) - len(cw)), ""]
        by = {}
        for r in cw:
            k = (r["category"], r["reason"].split(":")[0])
            by[k] = by.get(k, 0) + 1
        L += table(["category", "damage type", "files"], [(k[0], k[1], by[k]) for k in sorted(by)]) if by else ["No damaged files inside the restart window."]
        L += [""]
    pl = read_csv(a.plan)
    if pl is not None:
        c2 = {}
        for r in pl:
            c2[r["action"]] = c2.get(r["action"], 0) + 1
        L += ["Repair plan: " + (", ".join("%s=%d" % (k, c2[k]) for k in sorted(c2)) or "empty"), ""]
    gl = sorted(glob.glob(os.path.join(a.work, "graveyard", "*", "GRAVEYARD_LOG.csv")))
    if gl:
        n = sum(len(read_csv(p) or []) for p in gl)
        L += ["Items moved to graveyard so far: %d (folders: %d). Kept for evidence until the user purges them." % (n, len(gl)), ""]
    # 4 training
    sm = read_csv(a.summaries)
    L += ["## 4. Training runs (validation scores recorded at training time)", ""]
    if sm is None:
        L += ["not available", ""]
    else:
        keys = [k for k in (sm[0].keys() if sm else []) if k not in ("config", "seed", "group")][:4]
        groups = {}
        for r in sm:
            groups.setdefault((r["group"], r["config"]), []).append(r)
        rows = []
        for (g, c) in sorted(groups):
            rs = groups[(g, c)]
            cells = []
            for k in keys:
                vals = [float(x[k]) for x in rs if x.get(k) not in (None, "")]
                cells.append("%.4g / %.4g / %.4g" % (sum(vals) / len(vals), min(vals), max(vals)) if vals else "-")
            rows.append([g, c, len(rs)] + cells)
        L += ["Columns show mean / min / max over seeds for the most common numeric keys of train_summary.json.", ""]
        L += table(["group", "config", "runs"] + [k.split(".")[-1] for k in keys], rows)
        L += [""]
        conf = [r for r in sm if r["group"].startswith("confirmatory")]
        if conf:
            L += ["### Every confirmatory training run (collapsed seeds are data, never dropped)", ""]
            rr2 = []
            for r in sorted(conf, key=lambda r: (r["config"], int(float(r["seed"])))):
                rr2.append([r["config"], r["seed"]] + [("%.4g" % float(r[k])) if r.get(k) not in (None, "") else "-" for k in keys])
            L += table(["config", "seed"] + [k.split(".")[-1] for k in keys], rr2[:150])
            L += [""]
    # 5 sealed
    sealed = os.path.join(a.study, "results", "sealed")
    nfiles = nmark = 0
    for dp, dn, fn in os.walk(sealed):
        for n in fn:
            if n.endswith(".DONE.json"):
                nmark += 1
            elif n.endswith(".json"):
                nfiles += 1
    L += ["## 5. Sealed evaluation (blinded)", "", "Condition result files: %d. DONE markers: %d. Values are intentionally not summarised." % (nfiles, nmark), ""]
    # 6 tests
    L += ["## 6. Test suite", ""]
    tl = tail_lines(a.tests, 12)
    L += ["```", *(x for x in tl if x.strip()), "```"] if tl else ["not available"]
    L += [""]
    # 7 structure
    rr = read_csv(a.root_report)
    L += ["## 7. Project folder structure", ""]
    if rr is None:
        L += ["not available", ""]
    else:
        L += table(["item", "class", "MB", "proposal / state"], [(r["name"], r["cls"], r["size_mb"], r["proposal"][:40]) for r in rr])
        L += [""]
    # 8 anomalies
    ev = os.path.join(a.work, "logs", "events.md")
    L += ["## 8. Recorded anomalies (last 12 lines of events.md that mention ANOMALY)", ""]
    if os.path.exists(ev):
        lines = [x for x in open(ev, encoding="utf-8", errors="ignore").read().split("\n") if "ANOMALY" in x.upper()]
        L += ["- " + x[:200] for x in lines[-12:]] or ["none recorded"]
    else:
        L += ["not available"]
    L += ["", "## 8b. Infrastructure incidents (disclose in the methods section)", ""]
    if os.path.exists(ev):
        kw = ("INFRASTRUCTURE", "RESTART", "REBOOT", "ORPHAN", "REGENERATION", "STALE", "RE-QUEUED", "RENAMED")
        lines = [x for x in open(ev, encoding="utf-8", errors="ignore").read().split("\n") if any(k in x.upper() for k in kw)]
        L += ["- " + x[:220] for x in lines[-25:]] or ["none recorded"]
    else:
        L += ["not available"]
    L += ["", "## 9. What may NOT be concluded yet", "",
          "- No effect sizes, p-values or rankings of sealed results (blinded).",
          "- No claim that RL beats or loses to PID/Bayes on the confirmatory design until all 90 trainings and their evaluations are finished.",
          "- Re-executed jobs (restart damage) must be disclosed in the paper's methods as infrastructure re-runs.", "",
          "## 10. Paper readiness checklist", "",
          "- [ ] z1 confirmatory trainings 90/90 and evaluations complete (section 2)",
          "- [ ] integrity scan clean after the last restart (section 3)",
          "- [ ] interim figures from exploratory/validation data only",
          "- [ ] research roadmap items R1-R8 decided (see audit/Q1_PAPER_ROADMAP.md)",
          "- [ ] reproducibility package prepared (code, configs, seeds, environment lock, hashes)", ""]
    if a.appendix:
        L += ["", "---", "", "# Appendices: existing project documents, included unchanged (exploratory / validation data only)", ""]
        for i, p in enumerate(a.appendix):
            tag = "ABCDEFGH"[i % 8]
            if not os.path.exists(p):
                L += ["## Appendix %s: %s - file not found" % (tag, p), ""]
                continue
            txt = read_text_any(p).replace("\r", "").split("\n")
            cut = len(txt) > 400
            L += ["## Appendix %s: %s" % (tag, os.path.basename(p)),
                  "Source: %s | modified %s | %d lines%s" % (p, time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p))), len(txt), " | TRUNCATED to 400" if cut else ""),
                  "", "~~~text", *txt[:400], "~~~", ""]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("REPORT written: %s (%d lines)" % (a.out, len(L)))
    print("REPORT-DONE")

if __name__ == "__main__":
    main()
