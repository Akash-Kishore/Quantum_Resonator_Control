"""D5 (P2): pre-submit safety checks for z1 queue files.

Usage: python validate_jobs.py <json_file> [<json_file> ...]
Checks (a)-(j) over the union of the given files; prints one compact report
(max 15 lines); exit 0 iff all checks PASS.
"""
import json, os, shlex, sys

W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
LANES = {"gpu", "aer", "cpu1", "cpu2", "cpu3", "cpu4"}


def parse_cmd(cmd):
    head = cmd.split(" > ")[0] if " > " in cmd else cmd
    try:
        return shlex.split(head, posix=False)
    except ValueError:
        return head.split()


def argv_opts(tokens):
    """--key value map from a token list; tokens[0]=python, tokens[1]=script."""
    opts, i = {}, 2 if len(tokens) > 1 else 1
    while i < len(tokens):
        if tokens[i].startswith("--") and i + 1 < len(tokens):
            opts[tokens[i][2:]] = tokens[i + 1].strip('"')
            i += 2
        else:
            i += 1
    return opts


def rpath(cwd, p):
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(cwd, p))


def main():
    files = sys.argv[1:]
    jobs, per_file = [], {}
    for fp in files:
        js = json.load(open(fp, encoding="utf-8"))
        per_file[os.path.basename(fp)] = len(js)
        jobs.extend(js)
    queue = os.path.join(W, "queue")
    by_id = {j["id"]: j for j in jobs}

    def in_queue(i):
        return os.path.exists(os.path.join(queue, i + ".job.json"))

    def queued_job(i):
        with open(os.path.join(queue, i + ".job.json"), encoding="utf-8") as f:
            return json.load(f)

    fails = {k: [] for k in "abcdefghij"}
    if len(set(by_id)) != len(jobs):
        fails["a"].append("duplicate ids in files")
    coll = [i for i in by_id if in_queue(i)]
    if coll:
        fails["a"].append("already in queue: %s" % coll[:5])
    for j in jobs:
        i = j["id"]
        if j.get("lane") not in LANES:
            fails["b"].append(i)
        if j.get("lane") == "gpu" and j.get("depends_on"):
            fails["c"].append(i)
        for d in j.get("depends_on", []):
            if d not in by_id and not in_queue(d):
                fails["d"].append("%s->%s" % (i, d))
        tok = parse_cmd(j["cmd"])
        script = tok[1].strip('"') if len(tok) > 1 else ""
        if script == "-c":
            continue
        if script != "-m" and not os.path.exists(rpath(j["cwd"], script)):
            fails["e"].append(i)
        o = argv_opts(tok)
        cfgf = o.get("config") or o.get("cfg")
        if cfgf and not os.path.exists(rpath(j["cwd"], cfgf)):
            fails["g"].append("%s %s" % (i, cfgf))
        if "set" in o and "tag" in o and "outdir" in o:
            marker = rpath(j["cwd"], os.path.join(o["outdir"], "%s__%s.DONE.json" % (o["tag"], o["set"])))
            if [os.path.normpath(p) for p in j.get("expected_outputs", [])] != [marker]:
                fails["f"].append(i)
        elif "out" in o and "config" in o:
            outdir = os.path.normcase(rpath(j["cwd"], o["out"]))
            for p in j.get("expected_outputs", []):
                if os.path.normcase(os.path.dirname(os.path.normpath(p))) != outdir:
                    fails["f"].append(i)
        if o.get("method", "").startswith("rl:"):
            mdp = os.path.normcase(rpath(j["cwd"], o["method"][3:]))
            if not os.path.isdir(mdp):
                dep = (j.get("depends_on") or [None])[0]
                dj = by_id.get(dep) or (queued_job(dep) if dep and in_queue(dep) else None)
                okh = bool(dj and dj.get("expected_outputs")
                           and os.path.normcase(os.path.dirname(os.path.normpath(dj["expected_outputs"][0]))) == mdp)
                if not okh:
                    fails["h"].append("%s %s" % (i, o["method"]))
        if not os.path.isdir(j["cwd"]):
            fails["i"].append(i)
    man = os.path.join(W, "jobs_json", "z1_counts.json")
    if not os.path.exists(man):
        fails["j"].append("manifest z1_counts.json missing")
    else:
        m = json.load(open(man, encoding="utf-8"))
        rel = {fn: n for fn, n in per_file.items() if fn in m}
        for fn, n in rel.items():
            if m.get(fn) != n:
                fails["j"].append("%s %d!=%s" % (fn, n, m.get(fn)))
    labels = {"a": "unique ids, none in queue", "b": "lanes valid", "c": "gpu jobs no depends_on",
              "d": "depends_on resolvable", "e": "cmd script exists", "f": "expected_outputs producible",
              "g": "--cfg files exist", "h": "rl: model dir exists/produced", "i": "cwd exists",
              "j": "counts match generator manifest"}
    print("VALIDATE %s (%d jobs)" % (",".join(per_file), len(jobs)))
    man = os.path.join(W, "jobs_json", "z1_counts.json")
    note = ""
    if os.path.exists(man):
        mkeys = json.load(open(man, encoding="utf-8"))
        if not any(fn in mkeys for fn in per_file):
            note = " (n/a, no z1 files given)"
    else:
        note = " (manifest missing)"
    for k in "abcdefghij":
        print("(%s) %s: %s%s" % (k, labels[k],
                                 "PASS" if not fails[k] else "FAIL " + "; ".join(fails[k][:4]),
                                 note if k == "j" else ""))
    ok = not any(fails.values())
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
