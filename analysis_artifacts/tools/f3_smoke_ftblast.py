"""F3: Smoke test staged FTB_200000_last seed_200 on split val, 5 episodes.
Loads via evaluate.py, checks status=ok + no NaN, compares SHA of staged
best_model.zip vs source best_model.zip (should match - faithful copy).
"""
import hashlib, json, os, subprocess, sys

PY = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"
EVAL = r"C:\Projects\MiniProject_Sem4\qstudy\evaluate.py"
SRC_MODEL = r"C:\Projects\MiniProject_Sem4\qstudy\runs\s15__FTB_200000\seed_200\best_model.zip"
STG_MODEL = r"C:\Projects\MiniProject_Sem4\qstudy\runs_snap\s15__FTB_200000_last\seed_200\best_model.zip"
OUT = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\smoke\s3f_ftblast200k_seed200.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def has_nan(o):
    if isinstance(o, float):
        return o != o
    if isinstance(o, dict):
        return any(has_nan(v) for v in o.values())
    if isinstance(o, list):
        return any(has_nan(v) for v in o)
    return False


def main():
    # SHA compare (faithful copy check)
    src_sha = sha(SRC_MODEL)
    stg_sha = sha(STG_MODEL)
    print("source best_model.zip SHA:", src_sha)
    print("staged  best_model.zip SHA:", stg_sha)
    print("SHA identical:", src_sha == stg_sha)

    # Run evaluate.py on staged dir
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = [
        PY, EVAL,
        "--method", "rl:runs_snap\\s15__FTB_200000_last\\seed_200",
        "--cfg", "configs\\eval__rabi.json",
        "--cond", "nom", "--split", "val", "--episodes", "5",
        "--seed_base", "500000", "--out", OUT, "--engine", "surrogate",
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=r"C:\Projects\MiniProject_Sem4\qstudy",
                       capture_output=True, text=True, timeout=300)
    print("rc:", r.returncode)
    if r.stdout:
        print("STDOUT:", r.stdout[-2000:])
    if r.stderr:
        print("STDERR:", r.stderr[-2000:])

    with open(OUT) as f:
        d = json.load(f)
    print("status:", d.get("status"))
    print("lock_w1000_mean:", d.get("lock_w1000_mean"))
    print("any NaN in output:", has_nan(d))
    if d.get("status") != "ok":
        print("SMOKE FAILED: status != ok")
        sys.exit(1)
    if has_nan(d):
        print("SMOKE FAILED: NaN present")
        sys.exit(1)
    print("SMOKE OK: loads, status=ok, no NaN")
    print("SHA identical (copy faithful):", src_sha == stg_sha)


if __name__ == "__main__":
    main()
