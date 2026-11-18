"""Fix-up: emit the 16 missing m2f7 bayes re-run jobs (dither + rabi probes) with
probe-qualified ids (the original ids collided across probes)."""
import json, os

S = r"C:\Projects\MiniProject_Sem4\qstudy"
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
PY = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"
CPUS = ["cpu1", "cpu2", "cpu3", "cpu4"]

moved = json.load(open(os.path.join(W, "results", "f1f2_moved_manifest.json"), encoding="utf-8"))
jobs, i = [], 0
for m in moved:
    if not m["method"].startswith("ctrl:bayes"):
        continue
    probe = "ramsey_iq" if "__ramsey_iq__" in m["json"] else ("dither" if "__dither__" in m["json"] else "rabi")
    if probe == "ramsey_iq":
        continue  # covered by the 8 already-submitted jobs
    jid = "m2f7_bayes2_%s_%s__%s" % (probe, m["method"].replace("ctrl:", "").replace(":", "_"), m["cond"])
    out_abs = os.path.join(S, "results", "val", m["json"])
    jobs.append(dict(id=jid, lane=CPUS[i % 4], cwd=S,
                     cmd='"%s" %s > "%s" 2>&1' % (PY, m["cmd"], os.path.join(W, "results", jid + ".log")),
                     depends_on=[], expected_outputs=[out_abs],
                     env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                     max_retries=2, stall_minutes=90, max_wall_hours=24, watch=[]))
    i += 1
json.dump(jobs, open(os.path.join(W, "jobs_json", "m2f7_bayes2.json"), "w"), indent=1)
print("bayes2 jobs:", len(jobs))
