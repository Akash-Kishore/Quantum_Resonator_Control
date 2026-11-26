import datetime, json, os, sys

def path(w): return os.path.join(w, "STATE.json")

def load(w):
    try:
        with open(path(w), encoding="utf-8") as f: return json.load(f)
    except Exception:
        return {"steps": {}}

def save(w, s):
    with open(path(w) + ".tmp", "w", encoding="utf-8") as f: json.dump(s, f, indent=1)
    os.replace(path(w) + ".tmp", path(w))

if __name__ == "__main__":
    w, cmd = sys.argv[1], sys.argv[2]
    s = load(w)
    if cmd == "set":
        s["steps"][sys.argv[3]] = {"state": sys.argv[4], "note": " ".join(sys.argv[5:]), "time": datetime.datetime.now().isoformat(timespec="seconds")}
        save(w, s)
    elif cmd == "show":
        for k, v in s["steps"].items(): print(k, v["state"], v["time"], v["note"][:100])
    elif cmd == "get":
        print(s["steps"].get(sys.argv[3], {}).get("state", "NONE"))
