import pandas as pd
from db import get_engine

df = pd.read_csv("students.csv")
df.to_sql("students", get_engine(), if_exists = "append", index = False)
print("Loaded", len(df), "rows")