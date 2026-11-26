import json, os
J = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK\jobs_json"
def load(n): return json.load(open(os.path.join(J, n), encoding="utf-8"))
def dump(n, jobs):
    p = os.path.join(J, n)
    assert not os.path.exists(p), "exists: " + n
    json.dump(jobs, open(p, "w", encoding="utf-8"), indent=1)
tr = load("z1_train.json"); ev = load("z1_eval.json")
a = [j for j in tr if j["id"].startswith("z1a_")]
b = [j for j in tr if j["id"].startswith("z1b_")]
ea = [j for j in ev if any(d.startswith("z1a_") for d in j.get("depends_on", []))]
eb = [j for j in ev if any(d.startswith("z1b_") for d in j.get("depends_on", []))]
dump("z1_train_a.json", a); dump("z1_train_b.json", b)
dump("z1_eval_a.json", ea); dump("z1_eval_b.json", eb)
print("z1_train_a", len(a), "z1_train_b", len(b), "z1_eval_a", len(ea), "z1_eval_b", len(eb))
print("unmatched train", len(tr) - len(a) - len(b), "unmatched eval", len(ev) - len(ea) - len(eb))
