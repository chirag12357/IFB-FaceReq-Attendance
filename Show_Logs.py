import sqlite3 as sql

con = sql.connect("Rekognise_Logs.db")
cur = con.cursor()

for i in cur.execute("Select * from Logs"):
    print(i)

