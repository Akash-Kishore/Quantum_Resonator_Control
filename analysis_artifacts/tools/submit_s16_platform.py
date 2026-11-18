import json
import os
import sys

w = sys.argv[1]
py = sys.argv[2]
s = r"C:\Projects\MiniProject_Sem4\qstudy"
sys.path.insert(0, os.path.join(w, "tools"))
import jobtools

out = os.path.join(w, "results", "s16_platform_ext.txt")
cmd = '"%s" tests\\test_platform_ext.py > "%s" 2>&1' % (py, out)
print(jobtools.submit(w, "s16_platform_ext", "aer", s, cmd,
                      expected_outputs=[out], env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                      max_retries=1, stall_minutes=60, max_wall_hours=2))
