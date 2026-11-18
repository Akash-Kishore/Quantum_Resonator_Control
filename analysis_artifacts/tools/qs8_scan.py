#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs8_scan.py (v8) - READ-ONLY static audit of the project code. Changes nothing.
Finds: hard-coded absolute paths and fixed seeds, swallowed exceptions, very long functions, TODO/FIXME comments,
duplicate files, and python modules nothing imports (dead-code candidates). Writes a CSV and a markdown report.
Usage: python qs8_scan.py --roots R1 R2 --queue WORK\\queue --csv out.csv --md out.md"""
import os, re, ast, sys, csv, json, glob, hashlib, argparse

SKIP = {"__pycache__", ".git", ".pytest_cache", "site-packages", "venv", ".venv", "graveyard", "quarantine", "backup",
        "_LEGACY_MiniProject_Sem4", "node_modules", "runs", "results", "queue", "jobs_json", "logs", "audit", "audit8"}
WIN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?!/)")
UNIX = re.compile(r"^/(home|Users|mnt|usr|var|tmp)/")
TODO = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
TEXTX = {".ps1", ".cfg", ".yaml", ".yml", ".toml", ".ini", ".bat", ".cmd", ".json"}
LONG = 80

def walk(roots, exts):
    for r in roots:
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if d not in SKIP]
            for n in fn:
                if os.path.splitext(n)[1].lower() in exts:
                    yield os.path.join(dp, n)

def rd(p):
    raw = open(p, "rb").read()
    return raw.decode("utf-16", "ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig", "ignore")

class V(ast.NodeVisitor):
    def __init__(self):
        self.f, self.imports, self.main = [], set(), False
    def add(self, n, kind, d):
        self.f.append((getattr(n, "lineno", 0), kind, str(d)[:90]))
    def visit_Import(self, n):
        for x in n.names:
            self.imports.update(x.name.split("."))
    def visit_ImportFrom(self, n):
        if n.module:
            self.imports.update(n.module.split("."))
        for x in n.names:
            self.imports.add(x.name)
    def visit_Constant(self, n):
        if isinstance(n.value, str) and (WIN.search(n.value) or UNIX.match(n.value)):
            self.add(n, "ABS_PATH", n.value)
    def visit_Assign(self, n):
        for t in n.targets:
            nm = t.id if isinstance(t, ast.Name) else getattr(t, "attr", "")
            if "seed" in nm.lower() and isinstance(n.value, ast.Constant) and isinstance(n.value.value, int):
                self.add(n, "HARD_SEED", "%s = %s" % (nm, n.value.value))
        self.generic_visit(n)
    def visit_Call(self, n):
        for k in n.keywords:
            if k.arg and "seed" in k.arg.lower() and isinstance(k.value, ast.Constant) and isinstance(k.value.value, int):
                self.add(n, "HARD_SEED", "%s=%s" % (k.arg, k.value.value))
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr == "chdir":
            self.add(n, "CHDIR", "os.chdir")
        self.generic_visit(n)
    def visit_ExceptHandler(self, n):
        if n.type is None:
            self.add(n, "BARE_EXCEPT", "except:")
        elif all(isinstance(b, (ast.Pass, ast.Continue)) for b in n.body):
            self.add(n, "SWALLOWED_EXCEPT", "except ...: pass")
        self.generic_visit(n)
    def visit_FunctionDef(self, n):
        ln = (getattr(n, "end_lineno", n.lineno) or n.lineno) - n.lineno + 1
        if ln > LONG:
            self.add(n, "LONG_FUNC", "%s (%d lines)" % (n.name, ln))
        self.generic_visit(n)
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_If(self, n):
        t = n.test
        if isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__":
            self.main = True
        self.generic_visit(n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--queue", default="")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    files = sorted(set(walk(a.roots, {".py"})))
    rows, info, byhash = [], {}, {}
    for p in files:
        try:
            txt = rd(p)
            tree = ast.parse(txt)
        except (SyntaxError, ValueError) as e:
            rows.append((p, 0, "UNPARSEABLE", type(e).__name__))
            continue
        v = V()
        v.visit(tree)
        for ln, k, d in v.f:
            rows.append((p, ln, k, d))
        for i, line in enumerate(txt.split("\n"), 1):
            if TODO.search(line):
                rows.append((p, i, "TODO", line.strip()[:90]))
        info[p] = dict(loc=txt.count("\n") + 1, imports=v.imports, main=v.main)
        byhash.setdefault(hashlib.sha256(txt.encode("utf-8", "ignore")).hexdigest(), []).append(p)
    for p in walk(a.roots, TEXTX):
        if p.lower().endswith(".json") and not re.search(r"[\\/]configs?[\\/]", p):
            continue
        try:
            if os.path.getsize(p) > 2 * 1024 * 1024:
                continue
            for i, line in enumerate(rd(p).split("\n"), 1):
                if WIN.search(line):
                    rows.append((p, i, "ABS_PATH_CFG", line.strip()[:90]))
        except OSError:
            pass
    # reachability
    entries = set()
    if a.queue:
        for jp in glob.glob(os.path.join(a.queue, "*.job.json")):
            try:
                cmd = json.load(open(jp, encoding="utf-8-sig")).get("cmd", "")
            except Exception:
                continue
            for m in re.findall(r"([\w\-.]+)\.py\b", cmd):
                entries.add(m.lower())
    names = {}
    for p in info:
        names.setdefault(os.path.splitext(os.path.basename(p))[0].lower(), []).append(p)
    seen, stack = set(), []
    for p, i in info.items():
        b = os.path.splitext(os.path.basename(p))[0].lower()
        if i["main"] or b in entries or b.startswith("test") or "tests" in p.replace("\\", "/").lower().split("/"):
            stack.append(p)
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        for imp in info[p]["imports"]:
            for q in names.get(imp.lower(), []):
                if q not in seen:
                    stack.append(q)
    for p in info:
        if p not in seen:
            rows.append((p, 0, "UNREACHABLE", "no entry point reaches this module"))
    for h, ps in byhash.items():
        if len(ps) > 1:
            for p in ps[1:]:
                rows.append((p, 0, "DUPLICATE_OF", ps[0]))
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "line", "kind", "detail"])
        w.writerows(rows)
    cnt = {}
    for r in rows:
        cnt[r[2]] = cnt.get(r[2], 0) + 1
    loc = sum(i["loc"] for i in info.values())
    print("QS8 python_files=%d lines=%d findings=%d" % (len(info), loc, len(rows)))
    for k in sorted(cnt):
        print("  %-18s %d" % (k, cnt[k]))
    for k in ("ABS_PATH", "ABS_PATH_CFG", "UNREACHABLE", "DUPLICATE_OF", "UNPARSEABLE"):
        for r in [x for x in rows if x[2] == k][:5]:
            print("  e.g. %-13s %s:%d %s" % (k, r[0][-70:], r[1], r[3][:40]))
    if a.md:
        L = ["# Code audit (static, read-only)", "", "Python files: %d | lines: %d | findings: %d" % (len(info), loc, len(rows)), ""]
        L += ["| finding | count |", "|---|---|"] + ["| %s | %d |" % (k, cnt[k]) for k in sorted(cnt)] + [""]
        expl = {"ABS_PATH": "hard-coded absolute path in a string (must become a path relative to the project root)",
                "ABS_PATH_CFG": "absolute path inside a script/config", "HARD_SEED": "seed fixed in code (check it belongs in a config)",
                "BARE_EXCEPT": "except: with no type", "SWALLOWED_EXCEPT": "error caught and ignored",
                "LONG_FUNC": "function longer than %d lines (simplification candidate)" % LONG, "TODO": "open TODO/FIXME",
                "UNREACHABLE": "no entry point, test or job imports it (dead-code candidate: human check)",
                "DUPLICATE_OF": "identical copy of another file", "CHDIR": "os.chdir changes global state", "UNPARSEABLE": "syntax error"}
        for k in sorted(cnt):
            L += ["## %s - %s" % (k, expl.get(k, "")), ""]
            for r in [x for x in rows if x[2] == k][:40]:
                L.append("- `%s:%d` %s" % (r[0], r[1], r[3].replace("|", "/")))
            if cnt[k] > 40:
                L.append("- ... %d more in the CSV" % (cnt[k] - 40))
            L.append("")
        L += ["## Lines of code per file", "", "| file | lines | findings |", "|---|---|---|"]
        per = {}
        for r in rows:
            per[r[0]] = per.get(r[0], 0) + 1
        for p, i in sorted(info.items(), key=lambda kv: -kv[1]["loc"])[:60]:
            L.append("| %s | %d | %d |" % (p, i["loc"], per.get(p, 0)))
        open(a.md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("QS8-SCAN-DONE")

if __name__ == "__main__":
    main()
