import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv(r"C:/Users/24ucs127/Downloads/exp10.csv")
print("DIABETES DATA")
print(df.head(40).to_string(index=False))
features = ["Age", "BMI", "BloodPressure"]
X = df[features]
y = df["Diabetes"]
weak_learner = DecisionTreeClassifier(max_depth=1)

boost = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=5,
    random_state=42
)

boost.fit(X, y)

print("\nENTER PATIENT DETAILS")

user_age = float(input("Age: "))
user_bmi = float(input("BMI: "))
user_bp = float(input("Blood Pressure: "))

patient_data = pd.DataFrame(
    {
        "Age": [user_age],
        "BMI": [user_bmi],
        "BloodPressure": [user_bp]
    }
)
answer = boost.predict(patient_data)
if answer[0] == 1:
    print("Prediction : Diabetes Risk")
else:
    print("Prediction : No Diabetes Risk")

