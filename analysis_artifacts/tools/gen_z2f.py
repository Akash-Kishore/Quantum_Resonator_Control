"""F4: gen_z2f.py — clone z2s eval jobs for FTB last-step models.
Reads jobs_json\z2s_eval.json, writes jobs_json\z2f_eval.json with:
  - id prefix z2f_ (vs z2s_)
  - model dir = runs_snap\s15__FTB_<budget>_last\seed_<n> (staged in F2)
  - tag z2f__... (vs z2s__)
  - outputs extended by "_last" in tag
  - 4 budgets x 5 seeds x 5 cells = 100 jobs
New file only (jobs_json\). validate_jobs.py must PASS.
"""
import json, os, re, sys

JOBS_Z2S = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\jobs_json\z2s_eval.json"
OUT = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\jobs_json\z2f_eval.json"
PY = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"
QSTUDY = r"C:\Projects\MiniProject_Sem4\qstudy"


def parse_id(job_id):
    """z2s_nominal_s15__E4_s200__LG__s50000 -> dict"""
    parts = job_id.split("__")
    # parts[0] = z2s_nominal_s15, parts[1] = E4_s200, parts[2] = probe, parts[3] = s50000
    p0 = parts[0]  # z2s_nominal_s15
    cell_type = p0.split("_")[1]  # nominal | ofat
    p1 = parts[1]  # E4_s200
    cfg_name = p1.split("_")[0]  # E4
    seed = int(p1.split("_")[1][1:])  # strip leading 's'
    probe = parts[2]
    step = int(parts[3][1:])  # strip leading 's'
    return dict(cell_type=cell_type, cfg_name=cfg_name, seed=seed, probe=probe, step=step)


def budget_for_step(step):
    """E4 step number == FTB budget number."""
    return step


def main():
    jobs = json.load(open(JOBS_Z2S, encoding="utf-8"))
    z2f = []
    for j in jobs:
        info = parse_id(j["id"])
        b = budget_for_step(info["step"])
        s = info["seed"]
        probe = info["probe"]
        cell = info["cell_type"]
        # new id: z2f_<cell>_s15__FTB_<b>_last_s<s>__<probe>__s<b>  (seed included for uniqueness)
        new_id = "z2f_%s_s15__FTB_%d_last_s%d__%s__s%d" % (cell, b, s, probe, b)
        # new tag: z2f__s15__FTB_<b>_last_s<s>__<probe>__s<b>
        new_tag = "z2f__s15__FTB_%d_last_s%d__%s__s%d" % (b, s, probe, b)
        old_model = "runs_snap\\s15__E4_s%d\\seed_%d" % (info["step"], s)
        new_model = "runs_snap\\s15__FTB_%d_last\\seed_%d" % (b, s)
        old_tag = "z2s__s15__E4_s%d__%s__s%d" % (s, probe, info["step"])
        # cmd replacements
        cmd = j["cmd"]
        cmd = cmd.replace(old_model, new_model)
        cmd = cmd.replace(old_tag, new_tag)
        cmd = cmd.replace("z2s_", "z2f_", 1)  # log path prefix
        # expected_outputs
        new_outputs = []
        for out in j["expected_outputs"]:
            new_out = out.replace(old_tag, new_tag).replace("z2s__", "z2f__")
            new_outputs.append(new_out)
        new_job = dict(j)
        new_job["id"] = new_id
        new_job["cmd"] = cmd
        new_job["expected_outputs"] = new_outputs
        z2f.append(new_job)

    # round-robin lanes to match z2s distribution (25 per lane)
    lanes = ["cpu1", "cpu2", "cpu3", "cpu4"]
    for i, j in enumerate(z2f):
        j["lane"] = lanes[i % 4]

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(z2f, f, indent=1)
    print("wrote", OUT, len(z2f), "jobs")
    from collections import Counter
    c = Counter(j["lane"] for j in z2f)
    print("lanes:", dict(c))
    # print sample
    print("sample id:", z2f[0]["id"])
    print("sample cmd:", z2f[0]["cmd"][:200])
    print("sample outputs:", z2f[0]["expected_outputs"])


if __name__ == "__main__":
    main()
