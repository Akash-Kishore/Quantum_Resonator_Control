#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs10_lanes.py (v10) - make sure the six lane runners and the watchdog are running. Safe to run at any time:
a runner exits by itself when its lane is already served, and the watchdog is only started when none exists.
  status                           one line per lane + watchdog (read-only)
  ensure [--wait 420] [--gap 10]   start missing lanes ONE AT A TIME, then the watchdog; waits until every lane beats (a cold boot can take minutes)
  install-autostart                run 'ensure' automatically 60 s after every Windows logon (Startup folder, no admin needed)
  uninstall-autostart
  pause                            RAM relief without damage: stops the watchdog and writes the STOP file; every runner finishes its CURRENT job and then
                                   exits (no new jobs start). Training jobs can take 20-35 min to finish.
  resume                           removes the STOP file and runs 'ensure'
  --lanes gpu,cpu1,cpu2            work on a subset of lanes only (fewer lanes = less RAM); the watchdog is started for that subset
Exit code 0 = all lanes active, 1 = not. WORK is the folder above tools\\ (override with env QS_WORK)."""
import os, sys, time, subprocess, argparse

WORK = os.environ.get("QS_WORK") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANES = ["gpu", "aer", "cpu1", "cpu2", "cpu3", "cpu4"]
PY = sys.executable
FRESH = 120      # seconds: a heartbeat younger than this is fresh
LOADED = 900     # seconds: a live runner under heavy load may beat this slowly

def hb(lane):
    return os.path.join(WORK, "heartbeat_%s.txt" % lane)

def age(lane):
    try:
        return time.time() - os.path.getmtime(hb(lane))
    except OSError:
        return float("inf")

def powershell(script, timeout=120):
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                       capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "").strip()

def image_of(pid):
    """lower-case program name of a pid ('' = no such process)"""
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""
    for line in out.splitlines():
        if line.strip().startswith('"'):
            return line.strip().split('","')[0].strip('"').lower()
    return ""

def runner_alive(lane):
    try:
        pid = int(open(os.path.join(WORK, "runner_%s.pid" % lane)).read().strip())
    except Exception:
        return False
    return image_of(pid).startswith("python")

def launch_lane(lane):
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(WORK, "tools", "launch_runner.ps1"),
                    "-Py", PY, "-Work", WORK, "-Lane", lane], capture_output=True, text=True, timeout=180)

def watchdog_alive():
    try:
        out = powershell("(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'watchdog\\.ps1' } | Measure-Object).Count")
        return int(out or 0) > 0
    except Exception:
        return False

def start_watchdog():
    cl = 'powershell -NoProfile -ExecutionPolicy Bypass -File "%s" -Py "%s" -Work "%s" -Lanes "%s"' % (
        os.path.join(WORK, "tools", "watchdog.ps1"), PY, WORK, ",".join(LANES))
    powershell("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='%s'; CurrentDirectory='%s'}; \"RC=$($r.ReturnValue)\"" % (cl.replace("'", "''"), WORK.replace("'", "''")))

def stop_file():
    return os.path.join(WORK, "STOP")

def stop_watchdog():
    try:
        powershell("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'watchdog\\.ps1' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    except Exception:
        pass

def lane_state(lane, t0):
    """ACTIVE = beating; STARTING = a live runner that has not beaten yet (slow after a cold boot); DEAD = no runner"""
    a = age(lane)
    if a < FRESH or (time.time() - a) > t0:
        return "ACTIVE"
    if runner_alive(lane):
        return "ACTIVE" if a < LOADED else "STARTING"
    return "DEAD"

def lane_ok(lane, t0):
    return lane_state(lane, t0) == "ACTIVE"

def show(t0):
    for l in LANES:
        a = age(l)
        print("  %-5s heartbeat %s  %s" % (l, ("%d s" % a) if a != float("inf") else "none", lane_state(l, t0)))
    print("  watchdog %s" % ("running" if watchdog_alive() else "NOT running"))

def ensure(wait, gap):
    if os.path.exists(stop_file()):
        print("STOP file exists (pause is active): run 'resume' instead. Nothing started.")
        return 1
    t0 = time.time()
    todo = [l for l in LANES if age(l) >= FRESH and not runner_alive(l)]   # a lane with a live runner is only waited for
    print("LANES to start: " + (", ".join(todo) or "none"))
    for l in todo:
        launch_lane(l)
        time.sleep(gap)
    if not watchdog_alive():
        print("watchdog missing: starting it")
        start_watchdog()
    end = time.time() + wait
    while time.time() < end and not all(lane_ok(l, t0) for l in LANES):
        time.sleep(5)
    bad = [l for l in LANES if not lane_ok(l, t0)]
    show(t0)
    if bad and all(lane_state(l, t0) == "STARTING" for l in bad):
        print("STILL STARTING (runners alive, first heartbeat pending; slow after a cold boot): " + ", ".join(bad) + " - run 'status' again in a few minutes")
    print("ALL LANES ACTIVE" if not bad else "NOT ACTIVE: " + ", ".join(bad))
    return 0 if not bad else 1

def startup_cmd():
    return os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs", "Startup", "qstudy_lanes.cmd")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "ensure", "pause", "resume", "install-autostart", "uninstall-autostart"])
    ap.add_argument("--lanes", default="")
    ap.add_argument("--wait", type=float, default=420)
    ap.add_argument("--gap", type=float, default=10)
    a = ap.parse_args()
    if a.lanes:
        LANES[:] = [x.strip() for x in a.lanes.split(",") if x.strip()]
    if a.cmd == "pause":
        stop_watchdog()
        open(stop_file(), "w").write("paused %s\n" % time.strftime("%Y-%m-%d %H:%M"))
        print("PAUSED: watchdog stopped, STOP file written. Each runner exits after its current job. Use 'resume' to continue.")
        sys.exit(0)
    if a.cmd == "resume":
        if os.path.exists(stop_file()):
            os.remove(stop_file())
        sys.exit(ensure(a.wait, a.gap))
    if a.cmd == "status":
        never = float("inf")      # status has no launch time: only a fresh heartbeat (or a loaded live runner) counts as active
        show(never)
        sys.exit(0 if all(lane_ok(l, never) for l in LANES) else 1)
    if a.cmd == "ensure":
        sys.exit(ensure(a.wait, a.gap))
    p = startup_cmd()
    if a.cmd == "install-autostart":
        os.makedirs(os.path.dirname(p), exist_ok=True)
        log = os.path.join(WORK, "logs", "autostart.log")
        open(p, "w", newline="\r\n").write('@echo off\nping -n 61 127.0.0.1 >nul\n"%s" "%s" ensure --wait 420 >> "%s" 2>&1\n' % (PY, os.path.abspath(__file__), log))
        print("INSTALLED " + p)
    else:
        if os.path.exists(p):
            os.remove(p)
        print("REMOVED " + p)

if __name__ == "__main__":
    main()
