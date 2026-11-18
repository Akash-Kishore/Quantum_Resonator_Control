#!/usr/bin/env python
"""Update TR-4 in db.contrasts with computed values."""
import sqlite3, datetime

DB = r"C:\Projects\MiniProject_Sem4\qstudy\db\qstudy.sqlite"

tr4 = {
    "contrast_id": "TR-4",
    "endpoint": "P2T",
    "method_a": "E2_LB_zero_shot",
    "method_b": "E1_LG_zero_shot",
    "n_a": 5,
    "n_b": 5,
    "statistic": 24.0,
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
    
    # Show existing row before update
    c.execute("SELECT * FROM contrasts WHERE contrast_id = ?", (tr4["contrast_id"],))
    existing = c.fetchone()
    print("BEFORE update:", existing)
    
    # Update TR-4 with computed values
    recorded_at = datetime.datetime.now().isoformat(timespec="seconds")
    c.execute("""
        UPDATE contrasts SET
            endpoint = ?, method_a = ?, method_b = ?, n_a = ?, n_b = ?,
            statistic = ?, p_value = ?, effect_size = ?, ci_low = ?, ci_high = ?,
            holm_p = ?, exploratory = ?, recorded_at = ?
        WHERE contrast_id = ?
    """, (tr4["endpoint"], tr4["method_a"], tr4["method_b"], tr4["n_a"], tr4["n_b"],
          tr4["statistic"], tr4["p_value"], tr4["effect_size"], tr4["ci_low"], tr4["ci_high"],
          tr4["holm_p"], tr4["exploratory"], recorded_at, tr4["contrast_id"]))
    con.commit()
    
    # Verify update
    c.execute("SELECT * FROM contrasts WHERE contrast_id = ?", (tr4["contrast_id"],))
    row = c.fetchone()
    print("AFTER update:", row)
    
    # Also show all contrasts
    c.execute("SELECT * FROM contrasts")
    for r in c.fetchall():
        print("All contrasts:", r)
    
    con.close()

if __name__ == "__main__":
    main()