import sqlite3
db = r'C:\Projects\MiniProject_Sem4\qstudy\qstudy.sqlite'
conn = sqlite3.connect(db)
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
for row in c.fetchall():
    print(row)
conn.close()