"""qstudy docs_check — Appendix C document linter.
A docs\\approach_*.md FAILS if: a numbered heading (1..11) is missing, section 5 or 7 has no table,
the text contains TODO/TBD/XXX/"to be filled", a referenced file path does not exist, or a numeric
table cell in section 7 cannot be found in the database or a results file (tolerance 1e-6)."""
import glob, json, os, re, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "db", "qstudy.sqlite")
HEADINGS = ["1 Purpose and hypothesis", "2 Physics and mathematics", "3 Implementation",
            "4 Parameters and defaults", "5 Validation tests", "6 Experiments", "7 Results",
            "8 Interpretation", "9 Limitations and threats", "10 Reproduction commands",
            "11 Raw data paths with sha256 and database tables"]
BAD = re.compile(r"TODO|TBD|XXX|to be filled", re.I)


def sections(text):
    out = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^(\d+)\s", line.strip())
        if m and int(m.group(1)) in range(1, 12) and any(line.strip().startswith(h) for h in HEADINGS):
            cur = int(m.group(1))
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def check_doc(path):
    fails = []
    text = open(path, encoding="utf-8").read()
    secs = sections(text)
    for h in HEADINGS:
        n = int(h[0])
        if n not in secs:
            fails.append("missing heading: %s" % h)
    for n in (5, 7):
        body = "\n".join(secs.get(n, []))
        if "|" not in body:
            fails.append("section %d has no table" % n)
    if BAD.search(text):
        fails.append("contains TODO/TBD/XXX/'to be filled'")
    # referenced file paths must exist
    for m in re.finditer(r"`([A-Za-z]:\\[^`\n]+|[\w\-\\.]*results[/\\][^\s`\n|]+|[\w\-\\.]*qstudy[/\\][^\s`\n|]+)`", text):
        p = m.group(1).strip()
        if os.path.isabs(p):
            if not os.path.exists(p):
                fails.append("path does not exist: %s" % p)
    # numeric cells in section 7 must be findable in DB or results files (tolerance 1e-6)
    body7 = "\n".join(secs.get(7, []))
    con = sqlite3.connect(DB) if os.path.exists(DB) else None
    db_vals = []
    if con:
        for (v,) in con.execute("SELECT value FROM validation"):
            try:
                db_vals.append(float(v))
            except Exception:
                pass
        for row in con.execute("SELECT * FROM evals"):
            for v in row:
                if isinstance(v, (int, float)):
                    db_vals.append(float(v))
        con.close()
    file_vals = []
    for p in glob.glob(os.path.join(HERE, "results", "**", "*.json"), recursive=True):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue

        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                file_vals.append(float(o))
        walk(j)
    for p in glob.glob(os.path.join(HERE, "results", "tables", "*.csv")) + glob.glob(os.path.join(HERE, "results", "tables", "*.txt")):
        for line in open(p, encoding="utf-8", errors="replace"):
            for tok in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", line):
                file_vals.append(float(tok))
    pool = db_vals + file_vals
    for line in body7.splitlines():
        if "|" not in line or line.strip().startswith(("|-", "| ")) and set(line.strip()) <= set("|-: "):
            continue
        for cell in line.split("|"):
            cell = cell.strip()
            if re.fullmatch(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", cell):
                x = float(cell)
                if not any(abs(x - v) <= 1e-6 * max(1.0, abs(x)) for v in pool):
                    fails.append("numeric cell %s not found in db/results (section 7)" % cell)
    return fails


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    docs = args or sorted(glob.glob(os.path.join(HERE, "docs", "approach_*.md")))
    all_ok = True
    for d in docs:
        fails = check_doc(d)
        status = "PASS" if not fails else "FAIL"
        print("%s %s" % (status, d))
        for f in fails:
            print("   -", f)
            all_ok = False
        if not fails:
            all_ok = all_ok and True
    print("DOCS CHECK %s" % ("PASS" if all_ok else "FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
