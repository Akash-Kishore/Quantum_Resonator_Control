import json
PY = "C:\\Users\\akash\\anaconda3\\envs\\quantum_control\\python.exe"
S = "C:\\Projects\\MiniProject_Sem4\\qstudy"
W = "C:\\Projects\\MiniProject_Sem4_QSTUDY_WORK"
job = {
    "id": "s17_suite_changepoint",
    "lane": "cpu1",
    "cwd": "C:\\Projects\\MiniProject_Sem4",
    "cmd": '"%s" -m unittest discover qstudy\\tests -p "test_*.py" -v > "%s\\results\\s17_suite_changepoint.txt" 2>&1' % (PY, W),
    "depends_on": [],
    "expected_outputs": ["%s\\results\\s17_suite_changepoint.txt" % W],
    "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    "stall_minutes": 45,
}
with open(W + "\\jobs_json\\s17_suite_changepoint.json", "w") as f:
    json.dump([job], f, indent=1)
print("wrote suite job")
