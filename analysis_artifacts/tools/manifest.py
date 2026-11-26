import csv, hashlib, os, sys
SKIP = {".git", "__pycache__", "node_modules"}

def walk(root):
    for d, ds, fs in os.walk(root):
        ds[:] = [x for x in ds if x not in SKIP and not os.path.exists(os.path.join(d, x, "pyvenv.cfg")) and not os.path.exists(os.path.join(d, x, "conda-meta"))]
        for f in fs: yield os.path.join(d, f)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def build(root, out):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["rel", "size", "mtime", "sha256"])
        for p in walk(root):
            try:
                s = os.stat(p); w.writerow([os.path.relpath(p, root), s.st_size, int(s.st_mtime), sha(p)])
            except Exception as e:
                w.writerow([os.path.relpath(p, root), "", "", "ERR:%s" % type(e).__name__])

def load(p):
    with open(p, encoding="utf-8") as f: return {r["rel"]: r for r in csv.DictReader(f)}

if __name__ == "__main__":
    if sys.argv[1] == "build": build(sys.argv[2], sys.argv[3])
    else:
        a, b = load(sys.argv[2]), load(sys.argv[3])
        print("ADDED", sorted(set(b) - set(a))); print("REMOVED", sorted(set(a) - set(b)))
        print("CHANGED", sorted(k for k in a if k in b and a[k]["sha256"] != b[k]["sha256"]))
