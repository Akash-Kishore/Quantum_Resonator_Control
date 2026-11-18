import glob, json, os, sys

def submit(work, jid, lane, cwd, cmd, depends_on=(), expected_outputs=(), env=None, max_retries=2, stall_minutes=60, max_wall_hours=24, watch=()):
    p = os.path.join(work, "queue", jid + ".job.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p): return "EXISTS"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dict(id=jid, lane=lane, cwd=cwd, cmd=cmd, depends_on=list(depends_on), expected_outputs=list(expected_outputs), env=env or {}, max_retries=max_retries, stall_minutes=stall_minutes, max_wall_hours=max_wall_hours, watch=list(watch)), f, indent=1)
    return "SUBMITTED"

if __name__ == "__main__":
    w = sys.argv[1]
    if len(sys.argv) > 3 and sys.argv[2] == "submit":
        for j in json.load(open(sys.argv[3], encoding="utf-8")):
            j = dict(j); jid = j.pop("id")
            print(jid, submit(w, jid, **j))
        sys.exit(0)
    n = {}
    for p in sorted(glob.glob(os.path.join(w, "queue", "*.job.json"))):
        with open(p, encoding="utf-8") as f: j = json.load(f)
        sp = p.replace(".job.json", ".status.json")
        s = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {"state": "PENDING"}
        n[s["state"]] = n.get(s["state"], 0) + 1
        if "-v" in sys.argv or s["state"] in ("RUNNING", "FAILED", "BLOCKED"):
            print(j["id"], j["lane"], s["state"], s.get("attempts", 0), (s.get("tail") or s.get("note") or "")[-80:].replace("\n", " "))
    print("COUNTS", n)
