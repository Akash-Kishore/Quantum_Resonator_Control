import sqlite3
db = r'C:\Projects\MiniProject_Sem4\qstudy\db\qstudy.sqlite'
conn = sqlite3.connect(db)
c = conn.cursor()

c.execute("SELECT run_id, config_name, seed, finished FROM runs WHERE config_name LIKE ?", ('%ramsey_iq%',))
for row in c.fetchall():
    print(row)

conn.close()