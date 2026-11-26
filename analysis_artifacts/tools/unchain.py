import glob, json, os, sys, time

def rd(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def wr(p, d):
    for _ in range(8):
        try:
            with open(p + ".tmp", "w", encoding="utf-8") as f: json.dump(d, f, indent=1)
            os.replace(p + ".tmp", p); return True
        except PermissionError:
            time.sleep(1)
    return False

w = sys.argv[1]; q = os.path.join(w, "queue")
jobs = {}
for p in glob.glob(os.path.join(q, "*.job.json")):
    j = rd(p); jobs[j["id"]] = (p, j)
changed = unblocked = 0
for jid, (p, j) in jobs.items():
    sp = p.replace(".job.json", ".status.json")
    s = rd(sp) if os.path.exists(sp) else {"state": "PENDING"}
    if s["state"] not in ("PENDING", "BLOCKED") or j["lane"] != "gpu":
        continue
    old = j.get("depends_on", [])
    keep = [d for d in old if d in jobs and jobs[d][1]["lane"] != "gpu"]
    if keep != old:
        j["depends_on"] = keep
        if wr(p, j): changed += 1
    if s["state"] == "BLOCKED":
        s["state"] = "PENDING"; s["note"] = "unchained (was blocked by a failed predecessor)"
        if wr(sp, s): unblocked += 1
print("unchained jobs:", changed, " unblocked jobs:", unblocked)
