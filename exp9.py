import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r"C:/Users/24ucs127/Downloads/student_attendance_dataset.csv")
print("DATASET:")
print(df.head(40).to_string(index=False))

X = df[["Attendance", "StudyHours", "InternalMarks", "Participation"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy * 100, "%")
print("\n--- STUDENT PREDICTION ---")
attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours: "))
internal_marks = float(input("Enter Internal Marks: "))
participation = float(input("Enter Participation (1-10): "))

student = pd.DataFrame({
    "Attendance": [attendance],
    "StudyHours": [study_hours],
    "InternalMarks": [internal_marks],
    "Participation": [participation]
})

prediction = model.predict(student)
print("\nStudent Result:", prediction[0])
