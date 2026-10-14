import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500

attendance = np.clip(rng.normal(75, 15, N), 30, 100).round(2)
study_hours = np.clip(rng.normal(3, 1.5, N),0 , 10).round(2)
internal = np.clip(0.4*attendance + 6 * study_hours + rng.normal(20, 10, N), 0, 100).round(2)
assignment = np.clip(0.5*internal + rng.normal(35, 12, N), 0, 100).round(2)

score = 0.35*attendance + 4*study_hours + 0.4*internal + 0.1*assignment + rng.normal(0,6,N)
result = np.where(score >= np.percentile(score, 30),"Pass", "Fail")

df = pd.DataFrame({
    "student_id" : range(1, N+1),
    "name": [f"Student_{i}" for i in range(1, N+1)],
    "attendance_pct": attendance, "study_hours" : study_hours,
    "internal_marks": internal, "assignment_score":assignment,
    "result": result,
})
df.to_csv("students.csv", index=False)
print(df.head)