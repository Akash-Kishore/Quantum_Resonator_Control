#!/usr/bin/env python
"""Update TR-4 status to FINAL in db.contrasts."""
import sqlite3, datetime

DB = r"C:\Projects\MiniProject_Sem4\qstudy\db\qstudy.sqlite"

def main():
    con = sqlite3.connect(DB)
    c = con.cursor()
    
    # Show existing row before update
    c.execute("SELECT * FROM contrasts WHERE contrast_id = ?", ("TR-4",))
    existing = c.fetchone()
    print("BEFORE update:", existing)
    
    # Update TR-4 status to FINAL
    recorded_at = datetime.datetime.now().isoformat(timespec="seconds")
    c.execute("""
        UPDATE contrasts SET
            recorded_at = ?,
            status = 'FINAL'
        WHERE contrast_id = ?
    """, (recorded_at, "TR-4"))
    con.commit()
    
    # Verify update
    c.execute("SELECT * FROM contrasts WHERE contrast_id = ?", ("TR-4",))
    row = c.fetchone()
    print("AFTER update:", row)
    
    con.close()

if __name__ == "__main__":
    main()