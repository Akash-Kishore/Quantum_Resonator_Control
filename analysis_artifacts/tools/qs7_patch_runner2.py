#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs7_patch_runner2.py (v2) - upgrades the stale-pid fix in WORK\\tools\\job_runner.py.
v1 asked PowerShell for the command line of the saved pid. Right after a restart PowerShell can be slow, the check timed out
and the runner stayed out (the lane never started). v2 first asks tasklist (fast): if the pid belongs to a program that is not python,
or to no program, the pid file is stale at once. Only a python process is checked further with a long (120 s) command-line query.
Works on a job_runner.py that has v1, that has nothing, or that already has v2.
Usage: python qs7_patch_runner2.py --work WORK --apply | --revert | --selftest LANE | --selftest-pid PID"""
import os, sys, shutil, time, argparse

HELPER = '''def _qs7_parse_tasklist(text):
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
            return f.read().replace(b"\\0", b" ").decode("utf-8", "ignore").strip()
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

'''
OLD = "    if old and old != os.getpid() and alive(old):\n        sys.exit(0)\n"
NEW = "    if old and old != os.getpid() and alive(old) and runner_pid_ok(old, LANE) is not False:\n        sys.exit(0)\n"
ANCHOR = "def single_instance():"
START = "def _qs7_cmdline_of("

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
        pid = int(open(os.path.join(a.work, "runner_%s.pid" % a.selftest)).read().strip()) if a.selftest else a.selftest_pid
        t0 = time.time()
        res = ns["runner_pid_ok"](pid, a.selftest or "")
        print("pid=%d runner_pid_ok=%s  (%.1f s)   True=runner  False=stale/other program  None=unknown" % (pid, res, time.time() - t0))
        return
    bdir = os.path.join(a.work, "backup", time.strftime("%Y%m%d"))
    bak = os.path.join(bdir, "job_runner.py.bak.pre_pidfix2")
    if a.revert:
        if not os.path.exists(bak):
            print("NO BACKUP FOUND: " + bak)
            sys.exit(2)
        shutil.copy2(bak, jr)
        print("REVERTED from " + bak)
        return
    with open(jr, "r", encoding="utf-8", newline="") as f:
        txt = f.read()
    if "_qs7_parse_tasklist" in txt:
        print("ALREADY PATCHED (v2). Nothing changed.")
        return
    nl = "\r\n" if "\r\n" in txt else "\n"
    helper = HELPER.replace("\n", nl)
    if txt.count(ANCHOR) != 1:
        print("REFUSED: anchor 'def single_instance():' found %d times. Nothing changed." % txt.count(ANCHOR))
        sys.exit(2)
    if START in txt:                                   # v1 present: replace its helper block
        i, j = txt.index(START), txt.index(ANCHOR)
        if not i < j:
            print("REFUSED: unexpected layout. Nothing changed.")
            sys.exit(2)
        out = txt[:i] + helper + txt[j:]
    else:                                              # nothing present: insert helper and change the condition
        old, new = OLD.replace("\n", nl), NEW.replace("\n", nl)
        if txt.count(old) != 1:
            print("REFUSED: expected condition not found exactly once. Nothing changed.")
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
    print("PATCH2-DONE")

if __name__ == "__main__":
    main()
