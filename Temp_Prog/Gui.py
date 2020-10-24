import pandas as pd

df = pd.read_csv("Database.csv")
known_face_names = list(df["NAME"])
known_face_ids = list(df["ID Number"])

print(known_face_names)
print(known_face_ids)
