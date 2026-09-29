import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "Age": [19, 21, 24, 27, 31, 23],
    "Browsing History": [1, 2, 4, 7, 8, 3],
    "Time Spent": [3, 6, 12, 18, 22, 9],
    "Click": [0, 0, 1, 1, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

print("----- DATASET -----")
print(df)

# Features
X = df[["Age", "Browsing History", "Time Spent"]]

# Target
y = df["Click"]

# Create Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(X, y)

# User input
age = float(input("\nEnter Age: "))
history = float(input("Enter Browsing History: "))
time = float(input("Enter Time Spent on Website: "))

# New user
new_user = [[age, history, time]]

# Prediction
prediction = model.predict(new_user)

print("\nPredicted Click:", prediction[0])

if prediction[0] == 1:
    print("User is likely to CLICK the advertisement")
else:
    print("User is NOT likely to CLICK the advertisement")

# Feature Importance
print("\n----- FEATURE IMPORTANCE -----")

features = ["Age", "Browsing History", "Time Spent"]

for feature, coefficient in zip(features, model.coef_[0]):
    print(feature, ":", coefficient)

# Graph - Feature Importance
plt.bar(features, model.coef_[0])
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.title("Feature Importance - Logistic Regression")
plt.xticks(rotation=20)
plt.grid(True)
plt.show()
