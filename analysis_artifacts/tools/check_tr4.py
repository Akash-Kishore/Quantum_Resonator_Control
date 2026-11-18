import sqlite3
db = r'C:\Projects\MiniProject_Sem4\qstudy\qstudy.sqlite'
conn = sqlite3.connect(db)
c = conn.cursor()
c.execute('SELECT * FROM contrasts WHERE contrast_id LIKE ?', ('TR-%',))
for row in c.fetchall():
    print(row)
conn.close()