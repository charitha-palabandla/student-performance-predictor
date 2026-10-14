import pandas as pd
from db import get_engine
FEATURES = ["attendance_pct","study_hours","internal_marks","assignment_score"]

query = """
SELECT attendance_pct, study_hours, internal_marks, assignment_score, result FROM students
"""

df = pd.read_sql(query, get_engine())
print("Rows loaded:" ,len(df))

df = df.drop_duplicates()
df = df.dropna(subset=FEATURES + ["result"])
df = df[df["attendance_pct"].between(0, 100)]
df["passed"] = (df["result"] == "Pass").astype(int)

print("After cleaning: ", len(df))
print(df["result"].value_counts())

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt

 
print("\nAverages by result:")
print(df.groupby("result")[FEATURES].mean().round(1))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for label, grp in df.groupby("result"):
    axes[0].scatter(grp["attendance_pct"], grp["study_hours"], alpha=0.5, label=label)
axes[0].set(xlabel="Attendance %", ylabel="Study hours/day", title="Attendance vs Study Hours")
axes[0].legend()

df.groupby("result")["internal_marks"].mean().plot(kind="bar", ax=axes[1], color=["tomato", "seagreen"])
axes[1].set(title="Avg internal marks by result", ylabel="Marks")

plt.tight_layout()
plt.savefig("trends.png", dpi=150)
print("Saved trends.png")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


X = df[FEATURES]      
y = df["passed"]     

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining students:", len(X_train), "| Test students:", len(X_test))

scaler = StandardScaler().fit(X_train)    
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(scaler.transform(X_train), y_train)
print("Model trained!")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


pred = model.predict(scaler.transform(X_test))

print(f"\nAccuracy: {accuracy_score(y_test, pred):.2%}")
print(classification_report(y_test, pred, target_names=["Fail", "Pass"]))


ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Fail", "Pass"]).plot()
plt.savefig("confusion_matrix.png", dpi=150)


print("Feature influence (bigger number = matters more):")
for f, c in sorted(zip(FEATURES, model.coef_[0]), key=lambda t: -abs(t[1])):
    print(f"  {f:18s} {c:+.2f}")


import joblib


joblib.dump((scaler, model), "model.joblib")
print("\nSaved model.joblib")


new_student = pd.DataFrame([{
    "attendance_pct": 62,
    "study_hours": 1.5,
    "internal_marks": 45,
    "assignment_score": 55,
}])
prob = model.predict_proba(scaler.transform(new_student))[0][1]
print(f"New student -> chance of passing: {prob:.0%} ({'Pass' if prob >= 0.5 else 'Fail'})")

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score

candidates = {
    "Logistic Regression":            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Logistic Reg (balanced)":        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest":                  RandomForestClassifier(n_estimators=200, random_state=42),
}

print("\n--- Model comparison ---")
for name, m in candidates.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    acc = accuracy_score(y_test, p)
    fail_recall = recall_score(y_test, p, pos_label=0)   
    cv = cross_val_score(m, X, y, cv=5).mean()           
    print(f"{name:25s} accuracy={acc:.0%}  fail_recall={fail_recall:.0%}  5-fold CV={cv:.0%}")