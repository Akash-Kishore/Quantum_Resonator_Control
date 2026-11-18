#!/usr/bin/env python
"""Insert TR-4 into db.contrasts using the same schema as TR-1/TR-2."""
import sqlite3, datetime

DB = r"C:\Projects\MiniProject_Sem4\qstudy\db\qstudy.sqlite"

# TR-4 values from tierb3_analysis.py
tr4 = {
    "contrast_id": "TR-4",
    "endpoint": "P2T",
    "method_a": "E2_LB_zero_shot",
    "method_b": "E1_LG_zero_shot",
    "n_a": 5,
    "n_b": 5,
    "statistic": 24.0,          # U statistic
    "p_value": 0.015873015873015872,
    "effect_size": 0.92,
    "ci_low": 0.08120098039215681,
    "ci_high": 0.23214869281045755,
    "holm_p": 0.047619047619047616,
    "exploratory": 0,
}

def main():
    con = sqlite3.connect(DB)
    c = con.cursor()
    
    # Check if TR-4 already exists
    c.execute("SELECT * FROM contrasts WHERE contrast_id = ?", (tr4["contrast_id"],))
    existing = c.fetchone()
    if existing:
        print("TR-4 already exists in contrasts:", existing)
        con.close()
        return
    
    # Insert TR-4
    recorded_at = datetime.datetime.now().isoformat(timespec="seconds")
    c.execute("""
        INSERT INTO contrasts (contrast_id, endpoint, method_a, method_b, n_a, n_b,
                               statistic, p_value, effect_size, ci_low, ci_high,
                               holm_p, exploratory, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (tr4["contrast_id"], tr4["endpoint"], tr4["method_a"], tr4["method_b"],
          tr4["n_a"], tr4["n_b"], tr4["statistic"], tr4["p_value"],
          tr4["effect_size"], tr4["ci_low"], tr4["ci_high"],
          tr4["holm_p"], tr4["exploratory"], recorded_at))
    con.commit()
    
    # Verify insertion
    c.execute("SELECT * FROM contrasts WHERE contrast_id = ?", (tr4["contrast_id"],))
    row = c.fetchone()
    print("Inserted TR-4 row:", row)
    
    # Also show all contrasts
    c.execute("SELECT * FROM contrasts")
    for r in c.fetchall():
        print("All contrasts:", r)
    
    con.close()

if __name__ == "__main__":
    main()