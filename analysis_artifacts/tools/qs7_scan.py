#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs7_scan.py (v7) - READ-ONLY integrity scan. Never modifies, moves or deletes anything.
Finds files damaged by a restart: truncated or empty JSON/ZIP/pickle/SQLite/python files, NUL-filled tails,
DONE markers that disagree with their files, and training run folders that never finished.
Usage: python qs7_scan.py --roots R1 R2 --study STUDY --work WORK --boot "YYYY-MM-DD HH:MM" --out scan.csv
Only files whose modification time lies in [boot-before, boot+after] are marked in_window=1 (candidates for repair)."""
import os, sys, re, csv, json, time, zipfile, pickletools, sqlite3, ast, hashlib, argparse
from pathlib import Path

SKIP_NAMES = {"__pycache__", ".git", ".pytest_cache", ".ipynb_checkpoints", "node_modules", ".mypy_cache",
              "site-packages", ".venv", "venv"}
TEXTX = {".csv", ".log", ".txt", ".md", ".ps1", ".cfg", ".ini", ".yaml", ".yml", ".toml", ".tsv", ".bat", ".cmd"}
BIG = 300 * 1024 * 1024
SEED_DIR = re.compile(r"^seed_\d+$")
PARTIAL_DIR = re.compile(r"^seed_\d+\.partial_.*$")

def parse_boot(s):
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return time.mktime(time.strptime(s.strip(), f))
        except ValueError:
            pass
    raise SystemExit("bad --boot value (use 'YYYY-MM-DD HH:MM'): " + s)

def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def decode_text(data):
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")

def nul_damage(data):
    if len(data) >= 64 and data[-64:] == b"\x00" * 64:
        return True
    return (b"\x00" * 64) in data

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
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c
    return None

def check_json(path, size, study):
    if size == 0:
        return "CORRUPT", "ZERO_BYTES", []
    if size > BIG:
        return "INFO", "TOO_BIG_SKIPPED", []
    with open(path, "rb") as f:
        data = f.read()
    if nul_damage(data):
        return "CORRUPT", "NUL_BYTES", []
    try:
        obj = json.loads(decode_text(data))
    except Exception as e:
        return "CORRUPT", "JSON_PARSE:" + type(e).__name__, []
    if os.path.basename(path).endswith(".DONE.json") and isinstance(obj, dict):
        probs, rel = [], []
        for f in obj.get("files") or []:
            if not isinstance(f, dict) or not f.get("path"):
                continue
            r = resolve_listed(f["path"], path, study)
            bn = os.path.basename(f["path"].replace("\\", "/"))
            if r is None:
                probs.append("MISSING:" + bn)
                continue
            sha = f.get("sha256")
            if sha:
                try:
                    if sha256_of(r).lower() != str(sha).lower():
                        probs.append("SHA:" + bn)
                        rel.append(r)
                except OSError:
                    probs.append("UNREADABLE:" + bn)
        if probs:
            return "CORRUPT", "MARKER_" + ";".join(probs[:3]) + ("..." if len(probs) > 3 else ""), rel
    return "OK", "", []

def check_zip(path, size):
    if size == 0:
        return "CORRUPT", "ZERO_BYTES", []
    try:
        if not zipfile.is_zipfile(path):
            return "CORRUPT", "NOT_A_ZIP", []
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
        if bad:
            return "CORRUPT", "ZIP_BAD_MEMBER:" + str(bad), []
    except Exception as e:
        return "CORRUPT", "ZIP_ERROR:" + type(e).__name__, []
    return "OK", "", []

def check_pickle(path, size):
    if size == 0:
        return "CORRUPT", "ZERO_BYTES", []
    last = None
    try:
        with open(path, "rb") as f:
            for op, arg, pos in pickletools.genops(f):
                last = op.name
    except Exception as e:
        return "CORRUPT", "PICKLE_TRUNCATED:" + type(e).__name__, []
    if last != "STOP":
        return "CORRUPT", "PICKLE_NO_STOP", []
    return "OK", "", []

def check_npy(path, size):
    if size < 64:
        return "CORRUPT", "NPY_TOO_SMALL", []
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            return "CORRUPT", "NPY_BAD_MAGIC", []
    return "OK", "", []

def check_sqlite(path, size):
    if size == 0:
        return "CORRUPT", "ZERO_BYTES", []
    try:
        con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            r = con.execute("PRAGMA quick_check").fetchall()
        finally:
            con.close()
        if not r or r[0][0] != "ok":
            return "CORRUPT", "SQLITE_QUICK_CHECK", []
    except Exception as e:
        return "CORRUPT", "SQLITE_ERROR:" + type(e).__name__, []
    return "OK", "", []

def check_py(path, size):
    if size == 0:
        return ("OK", "", []) if os.path.basename(path) == "__init__.py" else ("CORRUPT", "ZERO_BYTES", [])
    with open(path, "rb") as f:
        data = f.read()
    if nul_damage(data):
        return "CORRUPT", "NUL_BYTES", []
    try:
        ast.parse(decode_text(data))
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return "CORRUPT", "PY_SYNTAX", []
    return "OK", "", []

def check_text(path, size):
    if size == 0:
        return "OK", "", []
    with open(path, "rb") as f:
        data = f.read(min(size, 50 * 1024 * 1024))
    if nul_damage(data):
        return "CORRUPT", "NUL_TAIL", []
    return "OK", "", []

def check_file(path, size, study):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return check_json(path, size, study)
    if ext in (".zip", ".npz"):
        return check_zip(path, size)
    if ext in (".pkl", ".pickle"):
        return check_pickle(path, size)
    if ext == ".npy":
        return check_npy(path, size)
    if ext in (".db", ".sqlite", ".sqlite3"):
        return check_sqlite(path, size)
    if ext == ".py":
        return check_py(path, size)
    if ext in TEXTX:
        return check_text(path, size)
    return "OK", "", []

def categorize(path):
    pl = path.replace("\\", "/").lower()
    ext = os.path.splitext(pl)[1]
    if "/queue/" in pl or "/jobs_json/" in pl:
        return "QUEUE"
    if ext in (".db", ".sqlite", ".sqlite3"):
        return "DB"
    if "/smoke" in pl or "/runs_smoke/" in pl or "/results_smoke/" in pl:
        return "SMOKE"
    if "/runs/" in pl or pl.endswith("/runs"):
        return "RUN"
    if "/results/" in pl or "/results_smoke/" in pl:
        return "RESULT"
    if "/configs/" in pl:
        return "CONFIG"
    if "/tools/" in pl:
        return "TOOLS"
    if ext == ".py":
        return "CODE"
    if "/logs/" in pl:
        return "LOG"
    if ext in (".md", ".txt"):
        return "DOC"
    return "OTHER"

DELETABLE = {"RUN", "RESULT", "SMOKE"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--study", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--boot", action="append", default=[])
    ap.add_argument("--before-min", type=float, default=180)
    ap.add_argument("--after-min", type=float, default=10)
    ap.add_argument("--recent-min", type=float, default=15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    boots = [parse_boot(b) for b in a.boot]
    now = time.time()
    skip_abs = set()
    if a.work:
        for d in ("graveyard", "quarantine", "backup"):
            skip_abs.add(os.path.normcase(os.path.normpath(os.path.join(a.work, d))))
    def in_win(m):
        return any(b - a.before_min * 60 <= m <= b + a.after_min * 60 for b in boots)
    rows, ok_counts, recent_n, scanned = [], {}, 0, 0
    run_dirs = []
    for root in a.roots:
        for dp, dn, fn in os.walk(root, followlinks=False):
            dn[:] = [d for d in dn if d not in SKIP_NAMES and
                     os.path.normcase(os.path.normpath(os.path.join(dp, d))) not in skip_abs]
            b = os.path.basename(dp)
            if SEED_DIR.match(b) or PARTIAL_DIR.match(b):
                run_dirs.append(dp)
            for name in fn:
                p = os.path.join(dp, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                scanned += 1
                cat = categorize(p)
                if now - st.st_mtime < a.recent_min * 60:
                    recent_n += 1
                    continue
                try:
                    verdict, reason, rel = check_file(p, st.st_size, a.study)
                except Exception as e:
                    verdict, reason, rel = "INFO", "CHECK_ERROR:" + type(e).__name__, []
                if verdict == "OK":
                    ok_counts[cat] = ok_counts.get(cat, 0) + 1
                    continue
                sha = ""
                if verdict == "CORRUPT" and st.st_size < 200 * 1024 * 1024:
                    try:
                        sha = sha256_of(p)
                    except OSError:
                        pass
                rows.append(dict(path=p, size=st.st_size, mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                                 category=cat, verdict=verdict, reason=reason, in_window=int(in_win(st.st_mtime)),
                                 deletable=int(cat in DELETABLE), sha256=sha, related=";".join(rel)))
    # training run folders
    for dp in run_dirs:
        if not re.search(r"[\\/]runs[\\/]", dp):      # the original study keeps models in rl_training\\trained_models (older layout): never judged
            continue
        names, newest, total = set(), 0.0, 0
        for r2, d2, f2 in os.walk(dp):
            for n in f2:
                try:
                    s2 = os.stat(os.path.join(r2, n))
                except OSError:
                    continue
                total += s2.st_size
                newest = max(newest, s2.st_mtime)
                if r2 == dp:
                    names.add(n)
        if newest and now - newest < max(a.recent_min * 4, 60) * 60:
            continue
        base = os.path.basename(dp)
        verdict = reason = None
        if PARTIAL_DIR.match(base):
            verdict, reason = "INFO", "PARTIAL_DIR"
        elif ("progress.log" in names or "best_model.zip" in names) and "train_summary.json" not in names:
            verdict, reason = "CORRUPT", "INCOMPLETE_RUN"
        elif "train_summary.json" in names:
            miss = [n for n in ("best_model.zip", "vec_normalize.pkl") if n not in names]
            if miss:
                verdict, reason = "CORRUPT", "RUN_MISSING_FILES:" + ",".join(miss)
        if verdict:
            rows.append(dict(path=dp, size=total, mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest or now)),
                             category="RUN", verdict=verdict, reason=reason, in_window=int(in_win(newest or 0)),
                             deletable=1, sha256="", related=""))
    cols = ["path", "size", "mtime", "category", "verdict", "reason", "in_window", "deletable", "sha256", "related"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    corrupt = [r for r in rows if r["verdict"] == "CORRUPT"]
    cw = [r for r in corrupt if r["in_window"] == 1]
    print("SCAN files_seen=%d recent_skipped=%d ok=%d non_ok_rows=%d" % (scanned, recent_n, sum(ok_counts.values()), len(rows)))
    print("BOOT_WINDOWS: " + ("; ".join(time.strftime("%Y-%m-%d %H:%M", time.localtime(b)) for b in boots) or "none"))
    print("CORRUPT total=%d  in_window=%d  out_of_window=%d" % (len(corrupt), len(cw), len(corrupt) - len(cw)))
    by = {}
    for r in cw:
        k = (r["category"], r["reason"].split(":")[0].split(";")[0])
        by[k] = by.get(k, 0) + 1
    for k in sorted(by):
        print("  IN_WINDOW %-8s %-28s %d" % (k[0], k[1], by[k]))
    for r in corrupt:
        if r["category"] in ("QUEUE", "DB") :
            print("  !! %s_CORRUPT %s (%s)" % (r["category"], r["path"], r["reason"]))
    print("  INFO_ROWS: " + str(sum(1 for r in rows if r["verdict"] == "INFO")))
    for r in cw[:15]:
        print("  - %s | %s | %s" % (r["path"][-110:], r["reason"][:40], r["mtime"]))
    print("SCAN-DONE")

if __name__ == "__main__":
    main()
