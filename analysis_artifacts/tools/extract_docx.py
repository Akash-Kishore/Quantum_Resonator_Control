import os, re, sys, zipfile
ROOT = r"C:\Projects\MiniProject_Sem4"
OUT = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\understanding\docs\extracted"
os.makedirs(OUT, exist_ok=True)

def docx_text(p):
    try:
        with zipfile.ZipFile(p) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    except Exception as e:
        return None, "ERR %s" % e
    xml = re.sub(r"<w:p[ >]", "\n<w:p ", xml)
    xml = re.sub(r"<w:tab/>", " ", xml)
    txt = re.sub(r"<[^>]+>", "", xml)
    lines = []
    for i, para in enumerate(txt.split("\n")):
        para = para.replace("&", "&").replace("<", "<").replace(">", ">").strip()
        if para:
            lines.append("[%d] %s" % (i, para))
    return "\n".join(lines), None

n_ok, n_err = 0, 0
for d, ds, fs in os.walk(ROOT):
    ds[:] = [x for x in ds if x not in {".git", "__pycache__", ".opencode", "qstudy"}]
    for f in fs:
        if not f.lower().endswith(".docx"):
            continue
        rel = os.path.relpath(os.path.join(d, f), ROOT)
        txt, err = docx_text(os.path.join(d, f))
        safe = rel.replace("\\", "__").replace("/", "__") + ".txt"
        if err:
            open(os.path.join(OUT, safe), "w", encoding="utf-8").write(err + "\n")
            n_err += 1
            continue
        open(os.path.join(OUT, safe), "w", encoding="utf-8").write(txt or "(EMPTY)")
        n_ok += 1
print("DOCX EXTRACTED ok=%d err=%d -> %s" % (n_ok, n_err, OUT))
