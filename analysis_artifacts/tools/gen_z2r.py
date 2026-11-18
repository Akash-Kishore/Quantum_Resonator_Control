"""SENS2R-B: gen_z2r.py — generate sealed eval jobs for FTB_200000 raw snapshot models.
Reads jobs_json\z2s_eval.json, writes jobs_json\z2r_eval.json with:
  - id prefix z2r_ (vs z2s_)
  - model dir = runs_snap\s15__FTB_200000_raw\seed_<n> (staged in step 3)
  - tag z2r__... (vs z2s__)
  - outputs extended by "_raw" in tag
  - 2 seeds (202, 204) x 5 cells (nominal LG, LB, RG, RB + OFAT RB) = 10 jobs
  - same sealed split base 920000, same episode counts
New file only (jobs_json\). validate_jobs.py must PASS.
NOTE: The old tool gen_z2f.py copied best_model.zip (best-val checkpoint). This tool uses the raw snapshot_200000.zip as best_model.zip (staged in step 3).
"""
import json, os, re, sys

JOBS_Z2S = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\jobs_json\z2s_eval.json"
OUT = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\jobs_json\z2r_eval.json"
QSTUDY = r"C:\Projects\MiniProject_Sem4\qstudy"

TARGET_SEEDS = [202, 204]
TARGET_STEP = 200000


def parse_id(job_id):
    """z2s_nominal_s15__E4_s200__LG__s50000 -> dict"""
    parts = job_id.split("__")
    p0 = parts[0]  # z2s_nominal_s15
    cell_type = p0.split("_")[1]  # nominal | ofat
    p1 = parts[1]  # E4_s200
    cfg_name = p1.split("_")[0]  # E4
    seed = int(p1.split("_")[1][1:])  # strip leading 's'
    probe = parts[2]
    step = int(parts[3][1:])  # strip leading 's'
    return dict(cell_type=cell_type, cfg_name=cfg_name, seed=seed, probe=probe, step=step)


def main():
    jobs = json.load(open(JOBS_Z2S, encoding="utf-8"))
    z2r = []
    for j in jobs:
        info = parse_id(j["id"])
        if info["seed"] not in TARGET_SEEDS:
            continue
        if info["step"] != TARGET_STEP:
            continue
        # Only keep LG, LB, RG, RB for nominal; RB for ofat
        if info["cell_type"] == "nominal" and info["probe"] not in ["LG", "LB", "RG", "RB"]:
            continue
        if info["cell_type"] == "ofat" and info["probe"] != "RB":
            continue
        
        s = info["seed"]
        probe = info["probe"]
        cell = info["cell_type"]
        step = info["step"]
        
        # new id: z2r_<cell>_s15__FTB_200000_raw_s202__LG__s200000
        new_id = "z2r_%s_s15__FTB_200000_raw_s%d__%s__s%d" % (cell, s, probe, step)
        # new tag: z2r__s15__FTB_200000_raw_s202__LG__s200000
        new_tag = "z2r__s15__FTB_200000_raw_s%d__%s__s%d" % (s, probe, step)
        old_model = "runs_snap\\s15__E4_s%d\\seed_%d" % (step, s)
        new_model = "runs_snap\\s15__FTB_200000_raw\\seed_%d" % s
        old_tag = "z2s__s15__E4_s%d__%s__s%d" % (s, probe, step)
        
        # cmd replacements
        cmd = j["cmd"]
        cmd = cmd.replace(old_model, new_model)
        cmd = cmd.replace(old_tag, new_tag)
        cmd = cmd.replace("z2s_", "z2r_", 1)  # log path prefix
        
        # expected_outputs
        new_outputs = []
        for out in j["expected_outputs"]:
            new_out = out.replace(old_tag, new_tag).replace("z2s__", "z2r__")
            new_outputs.append(new_out)
        
        new_job = dict(j)
        new_job["id"] = new_id
        new_job["cmd"] = cmd
        new_job["expected_outputs"] = new_outputs
        z2r.append(new_job)

    # round-robin lanes to match z2s distribution
    lanes = ["cpu1", "cpu2", "cpu3", "cpu4"]
    for i, j in enumerate(z2r):
        j["lane"] = lanes[i % 4]

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(z2r, f, indent=1)
    print("wrote", OUT, len(z2r), "jobs")
    from collections import Counter
    c = Counter(j["lane"] for j in z2r)
    print("lanes:", dict(c))
    # print sample
    if z2r:
        print("sample id:", z2r[0]["id"])
        print("sample cmd:", z2r[0]["cmd"][:200])
        print("sample outputs:", z2r[0]["expected_outputs"])


if __name__ == "__main__":
    main()