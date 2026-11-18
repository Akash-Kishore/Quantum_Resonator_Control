import datetime, glob, json, os, sys
w = sys.argv[1]; q = os.path.join(w, "queue")
n = {}; run = []; failed = []
for p in glob.glob(os.path.join(q, "*.job.json")):
    sp = p.replace(".job.json", ".status.json")
    try:
        s = json.load(open(sp, encoding="utf-8"))
    except Exception:
        s = {"state": "PENDING"}
    st = s.get("state", "PENDING"); n[st] = n.get(st, 0) + 1
    jid = os.path.basename(p)[:-9]
    if st == "RUNNING": run.append(jid)
    if st == "FAILED": failed.append(jid)
mark = os.path.join(w, "brief_last_failed.json")
try:
    old = set(json.load(open(mark)))
except Exception:
    old = set()
new = sorted(set(failed) - old)
json.dump(sorted(failed), open(mark, "w"))
stale = []
for l in glob.glob(os.path.join(w, "heartbeat_*.txt")):
    age = datetime.datetime.now().timestamp() - os.path.getmtime(l)
    if age > 240: stale.append(os.path.basename(l)[10:-4] + ":%ds" % age)
print("JOBS", n, "| running:", (",".join(run))[:150], "| NEW_FAILED:", ((",".join(new))[:200] or "-"), "| STALE_LANES:", (",".join(stale) or "-"))
