"""Extract a readable digest dump from opencode chat-export JSON files.
Dumps per message: role, time, files written (from info.summary.diffs names),
full text parts, reasoning parts (full), tool calls as (tool, input, output tail 800).
Usage: python extract_chat.py <export.json> <out_dump.txt>
"""
import datetime, json, sys

def compact(obj, limit=300):
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "..."

def main(src, out):
    d = json.load(open(src, encoding="utf-8"))
    ms = d["messages"]
    n = len(ms)
    lines = []
    lines.append("SESSION: %s" % d["info"].get("title", ""))
    lines.append("MODEL: %s" % json.dumps(d["info"].get("model", {}))[:200])
    lines.append("N_MESSAGES: %d" % n)
    for i, m in enumerate(ms):
        info = m["info"]
        t = info.get("time", {}).get("created")
        ts = datetime.datetime.fromtimestamp(t / 1000).isoformat(timespec="seconds") if t else "?"
        role = info.get("role", "?")
        diffs = info.get("summary", {}).get("diffs") or []
        files = []
        for df in diffs:
            fn = df.get("file")
            if fn:
                files.append(fn)
        head = "=== MSG %d/%d %s %s" % (i + 1, n, role, ts)
        if files:
            head += " | files_changed: %s" % ",".join(files)
        lines.append(head)
        for pt in m["parts"]:
            ty = pt.get("type")
            if ty == "text":
                lines.append("[TEXT] " + (pt.get("text") or ""))
            elif ty == "reasoning":
                lines.append("[REASONING] " + (pt.get("text") or ""))
            elif ty == "tool":
                st = pt.get("state", {}) or {}
                inp = compact(st.get("input", {}), 400)
                outp = st.get("output", "")
                if not isinstance(outp, str):
                    outp = json.dumps(outp, ensure_ascii=False)
                tail = outp[-800:] if len(outp) > 800 else outp
                tail = tail.replace("\r", " ").replace("\n", " | ")
                lines.append("[TOOL %s] input=%s | out_tail=%s" % (pt.get("tool"), inp.replace("\n", " "), tail))
        lines.append("")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", out, "lines:", len(lines), "chars:", sum(len(x) for x in lines))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
