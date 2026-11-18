import datetime, glob, json, os, re, subprocess, sys, time, traceback
WORK, LANE = sys.argv[1], sys.argv[2]
Q, L = os.path.join(WORK, "queue"), os.path.join(WORK, "logs")
os.makedirs(Q, exist_ok=True); os.makedirs(L, exist_ok=True)
TRANSIENT = re.compile(r"CUDA out of memory|PermissionError|WinError 32|WinError 5|Access is denied|Resource temporarily|Connection reset|MemoryError|cudaError|DefaultCPUAllocator", re.I)

def now(): return datetime.datetime.now().isoformat(timespec="seconds")

def rd(p):
    for _ in range(6):
        try:
            with open(p, encoding="utf-8") as f: return json.load(f)
        except Exception: time.sleep(0.5)
    return None

def wr(p, d):
    with open(p + ".tmp", "w", encoding="utf-8") as f: json.dump(d, f, indent=1)
    os.replace(p + ".tmp", p)

def status(jid): return rd(os.path.join(Q, jid + ".status.json")) or {"state": "PENDING", "attempts": 0}

def setstat(jid, **kw):
    s = status(jid); s.update(kw); s["updated"] = now(); wr(os.path.join(Q, jid + ".status.json"), s)

def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % int(pid), "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return False
    for line in out.splitlines():
        parts = line.strip().strip('"').split('","')
        if len(parts) > 1 and parts[1] == str(int(pid)): return True
    return False

def beat(msg=""):
    with open(os.path.join(WORK, "heartbeat_%s.txt" % LANE), "w") as f: f.write(now() + " " + msg)

def nap(sec):
    end = time.time() + sec
    while time.time() < end:
        beat("backoff"); time.sleep(max(0, min(10, end - time.time())))

def kill(pid): subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)

def all_jobs():
    out = {}
    for p in sorted(glob.glob(os.path.join(Q, "*.job.json"))):
        j = rd(p)
        if j: out[j["id"]] = j
    return out

def finish(jid, j, rc, reason, log, att):
    ok = os.path.join(Q, jid + ".ok")
    tail = open(log, errors="replace").read()[-3000:] if log and os.path.exists(log) else ""
    good = os.path.exists(ok) and all(os.path.exists(x) for x in j.get("expected_outputs", []))
    if good:
        setstat(jid, state="DONE", rc=rc, finished=now()); return
    retry = (reason or "").startswith("STALL") or bool(TRANSIENT.search(tail))
    if retry and att <= j.get("max_retries", 2):
        setstat(jid, state="PENDING", rc=rc, note="retry %d after %s" % (att, reason or "transient error"), tail=tail[-500:])
        nap(min(600, 60 * att))
    else:
        setstat(jid, state="FAILED", rc=rc, finished=now(), reason=reason, tail=tail[-1500:])

def watch(jid, j, pid, log, att, p=None):
    t0, reason = time.time(), None
    stall, maxw = j.get("stall_minutes", 60) * 60, j.get("max_wall_hours", 24) * 3600
    wf = j.get("watch", [])
    while (p.poll() is None) if p else alive(pid):
        time.sleep(10); beat(jid)
        ts = [os.path.getmtime(x) for x in [log] + wf if x and os.path.exists(x)]
        age = time.time() - (max(ts) if ts else t0)
        if age > stall: reason = "STALL no output for %d min" % (age // 60)
        elif time.time() - t0 > maxw: reason = "MAXWALL %s h" % j.get("max_wall_hours", 24)
        if reason:
            kill(pid); time.sleep(5); break
    finish(jid, j, (p.poll() if p else None), reason, log, att)

def run_job(j):
    jid = j["id"]; att = status(jid).get("attempts", 0) + 1
    log = os.path.join(L, "%s.attempt%d.log" % (jid, att)); ok = os.path.join(Q, jid + ".ok")
    if os.path.exists(ok): os.remove(ok)
    env = dict(os.environ); env.update(j.get("env", {})); env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    with open(log, "wb") as lf:
        p = subprocess.Popen(j["cmd"] + ' && echo OK> "%s"' % ok, shell=True, cwd=j["cwd"], env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        setstat(jid, state="RUNNING", pid=p.pid, attempts=att, log=log, started=now())
        watch(jid, j, p.pid, log, att, p)

def _qs7_parse_tasklist(text):
    """image name (lower case) from the CSV output of tasklist; "" if no process matched"""
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith('"'):
            return line.split('","')[0].strip('"').lower()
    return ""

def _qs7_cmdline_of(pid):
    """command line of a process: text, "" if it does not exist, None if it cannot be determined"""
    import subprocess
    try:
        if os.name == "nt":
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d').CommandLine" % int(pid)],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return None
            return (r.stdout or "").strip()
        with open("/proc/%d/cmdline" % int(pid), "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "ignore").strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return None

def runner_pid_ok(pid, lane):
    """False = the pid is NOT a job_runner process (stale pid file). True or None = a runner or unknown: stay out."""
    if os.name == "nt":
        import subprocess
        try:
            r = subprocess.run(["tasklist", "/FI", "PID eq %d" % int(pid), "/FO", "CSV", "/NH"],
                               capture_output=True, text=True, timeout=30)
            img = _qs7_parse_tasklist(r.stdout)
            if not img or not img.startswith("python"):
                return False
        except Exception:
            pass
    c = _qs7_cmdline_of(pid)
    if c is None:
        return None
    if not c:
        return False
    return "job_runner" in c.lower()

def single_instance():
    pf = os.path.join(WORK, "runner_%s.pid" % LANE)
    try:
        old = int(open(pf).read().strip())
    except Exception:
        old = None
    if old and old != os.getpid() and alive(old) and runner_pid_ok(old, LANE) is not False:
        sys.exit(0)
    with open(pf, "w") as f: f.write(str(os.getpid()))

def main():
    for jid, j in all_jobs().items():
        s = status(jid)
        if j["lane"] == LANE and s["state"] == "RUNNING":
            if alive(s.get("pid", -1)): watch(jid, j, s["pid"], s.get("log", ""), s.get("attempts", 1))
            else: finish(jid, j, None, None, s.get("log", ""), s.get("attempts", 1))
    while not os.path.exists(os.path.join(WORK, "STOP")):
        beat("idle"); ran = False
        for jid, j in all_jobs().items():
            if j["lane"] != LANE or status(jid)["state"] != "PENDING": continue
            dep = [status(d)["state"] for d in j.get("depends_on", [])]
            if any(d in ("FAILED", "BLOCKED") for d in dep):
                setstat(jid, state="BLOCKED", note="dependency failed"); continue
            if all(d == "DONE" for d in dep):
                run_job(j); ran = True; break
        if not ran: time.sleep(15)

single_instance()
while True:
    try:
        main(); break
    except SystemExit:
        raise
    except Exception:
        with open(os.path.join(L, "runner_%s.err" % LANE), "a") as f: f.write(now() + "\n" + traceback.format_exc() + "\n")
        time.sleep(30)
