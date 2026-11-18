import glob, json, os, re, datetime
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
Q = os.path.join(W, "queue"); ST = os.path.join(W, "smoke", "train_watch.json")
DEF = {"busy_ts": "15:32:00", "baselines": {"rl__dither__DR": 1785.7, "rl__dither__DRcurr": 1687.8, "rl__rabi__plain": 2114.4},
       "global_proxy": 2114.4, "first_drop": None, "reported": False}
st = json.load(open(ST, encoding="utf-8")) if os.path.exists(ST) else DEF
if "baselines" not in st: st.update(DEF)
busy = datetime.datetime.strptime(st["busy_ts"], "%H:%M:%S")
def vals(p):
    out = []
    for l in open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"\[(\d\d:\d\d:\d\d)\] VAL steps=(\d+)", l)
        if m: out.append((datetime.datetime.strptime(m.group(1), "%H:%M:%S"), int(m.group(2))))
    return out
def slope(pts):
    return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]).total_seconds() if len(pts) > 1 else None
run = []
busy_cpu = 0
for p in glob.glob(os.path.join(Q, "*.status.json")):
    try: s = json.load(open(p, encoding="utf-8"))
    except Exception: continue
    if s.get("state") != "RUNNING": continue
    jid = os.path.basename(p)[:-12]
    j = json.load(open(os.path.join(Q, jid + ".job.json"), encoding="utf-8"))
    if j["lane"] == "gpu": run.append((jid, j))
    elif j["lane"].startswith("cpu"): busy_cpu += 1
now = datetime.datetime.now()
done_msg = ""
if not run:
    print("train_rate: no gpu job RUNNING; busy_cpu=%d" % busy_cpu)
elif len(run) > 1:
    print("train_rate: MULTIPLE gpu RUNNING: %s" % ",".join(r[0] for r in run))
else:
    jid, j = run[0]
    m = re.search(r'--out "([^"]+)"', j["cmd"]); prog = os.path.join(m.group(1), "progress.log")
    cfg = jid.split("_", 2)[2]
    v = vals(prog)
    pre = slope([x for x in v if x[0] < busy]); post = slope([x for x in v if x[0] >= busy])
    base, kind = (pre, "own-pre") if pre else (st["baselines"].get(cfg, st["global_proxy"]), "cfg-ref" if cfg in st["baselines"] else "proxy")
    if pre and cfg not in st["baselines"]:
        st["baselines"][cfg] = pre; done_msg = " (+new baseline %s=%.1f)" % (cfg, pre)
    drop = post is not None and base is not None and post < 0.7 * base
    if drop and busy_cpu == 4 and kind != "proxy":
        if st["first_drop"] is None: st["first_drop"] = now.strftime("%H:%M:%S")
        age = (now - datetime.datetime.strptime(st["first_drop"], "%H:%M:%S")).total_seconds()
        if age > 1200 and not st["reported"]:
            st["reported"] = True
            print("TRAIN-SLOWDOWN %s baseline(%s)=%.1f st/s current=%.1f st/s (drop %.0f%%, sustained %.0f min, 4 cpu lanes busy)" % (jid, kind, base, post, 100*(1-post/base), age/60))
        else: print("train_rate: %s DROP ongoing base(%s)=%.1f post=%.1f age=%.0fs busy_cpu=%d" % (jid, kind, base, post, age, busy_cpu))
    else:
        if not (drop and busy_cpu == 4): st["first_drop"] = None
        print("train_rate: %s base(%s)=%s post=%s busy_cpu=%d%s" % (jid, kind, ("%.1f" % base) if base else "n/a", ("%.1f" % post) if post else "waiting", busy_cpu, done_msg))
json.dump(st, open(ST, "w"), indent=1)
