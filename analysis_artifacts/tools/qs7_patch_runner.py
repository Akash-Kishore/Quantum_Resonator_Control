#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs7_patch_runner.py (v7) - fixes the stale-pid bug in WORK\\tools\\job_runner.py.
Bug: single_instance() exits when the pid in runner_<lane>.pid is ALIVE, even if Windows reused that pid for an unrelated program
(for example UDCUserAgent.exe). The lane then never starts. Fix: the pid only counts as a runner if its command line contains 'job_runner'.
If the check itself fails (cannot read the command line) the old behaviour is kept (stay out), so two runners can never share a lane.
Only newly started runners use the new code; running runners are not affected.
Usage:
  python qs7_patch_runner.py --work WORK --selftest LANE        (read-only; reads runner_<LANE>.pid and prints the check result)
  python qs7_patch_runner.py --work WORK --selftest-pid PID     (read-only; checks one pid)
  python qs7_patch_runner.py --work WORK --apply                (backs up and patches job_runner.py)
  python qs7_patch_runner.py --work WORK --revert               (restores the backup)"""
import os, sys, shutil, time, argparse, glob

HELPER = '''def _qs7_cmdline_of(pid):
    """command line of a process: text, "" if the process does not exist, None if it cannot be determined"""
    import subprocess
    try:
        if os.name == "nt":
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d').CommandLine" % int(pid)],
                               capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                return None
            return (r.stdout or "").strip()
        with open("/proc/%d/cmdline" % int(pid), "rb") as f:
            return f.read().replace(b"\\0", b" ").decode("utf-8", "ignore").strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return None

def runner_pid_ok(pid, lane):
    """False = the pid is NOT a job_runner process (stale pid file: pid dead or reused by another program).
    True or None = a runner (or unknown): stay out."""
    c = _qs7_cmdline_of(pid)
    if c is None:
        return None
    if not c:
        return False
    return "job_runner" in c.lower()

'''
OLD = "    if old and old != os.getpid() and alive(old):\n        sys.exit(0)\n"
NEW = "    if old and old != os.getpid() and alive(old) and runner_pid_ok(old, LANE) is not False:\n        sys.exit(0)\n"
ANCHOR = "def single_instance():"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--selftest")
    g.add_argument("--selftest-pid", type=int)
    a = ap.parse_args()
    jr = os.path.join(a.work, "tools", "job_runner.py")
    if a.selftest or a.selftest_pid:
        ns = {"os": os}
        exec(HELPER, ns)
        if a.selftest:
            pf = os.path.join(a.work, "runner_%s.pid" % a.selftest)
            pid = int(open(pf).read().strip())
        else:
            pid = a.selftest_pid
        cl = ns["_qs7_cmdline_of"](pid)
        print("pid=%d cmdline=%r" % (pid, (cl or "")[:120] if cl is not None else None))
        print("runner_pid_ok = %s   (True = a job_runner process; False = stale/other program; None = unknown)" % ns["runner_pid_ok"](pid, a.selftest or ""))
        return
    bdir = os.path.join(a.work, "backup", time.strftime("%Y%m%d"))
    bak = os.path.join(bdir, "job_runner.py.bak.pre_pidfix")
    if a.revert:
        if not os.path.exists(bak):
            print("NO BACKUP FOUND: " + bak)
            sys.exit(2)
        shutil.copy2(bak, jr)
        print("REVERTED from " + bak)
        return
    with open(jr, "r", encoding="utf-8", newline="") as f:
        txt = f.read()
    if "runner_pid_ok" in txt:
        print("ALREADY PATCHED. Nothing changed.")
        return
    nl = "\r\n" if "\r\n" in txt else "\n"
    old = OLD.replace("\n", nl)
    new = NEW.replace("\n", nl)
    helper = HELPER.replace("\n", nl)
    if txt.count(old) != 1 or txt.count(ANCHOR) != 1:
        print("REFUSED: expected text not found exactly once (old=%d anchor=%d). Nothing changed." % (txt.count(old), txt.count(ANCHOR)))
        sys.exit(2)
    out = txt.replace(ANCHOR, helper + ANCHOR).replace(old, new)
    try:
        compile(out, jr, "exec")
    except SyntaxError as e:
        print("REFUSED: patched text does not compile (%s). Nothing changed." % e)
        sys.exit(2)
    os.makedirs(bdir, exist_ok=True)
    shutil.copy2(jr, bak)
    with open(jr, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print("PATCHED %s (backup %s)" % (jr, bak))
    print("PATCH-DONE")

if __name__ == "__main__":
    main()
