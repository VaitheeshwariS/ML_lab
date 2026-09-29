import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = {
    'bedroom': [2,7,9,4, 3, 4, 5, 3],
    'size': [1000,2000,1944,1604, 1500, 1800, 2500, 1600],
    'age': [10, 5, 8, 6, 3, 6,8, 6],
    'price': [30, 55,65,70, 45, 55, 75, 50]
}
df = pd.DataFrame(data)
X = df[['bedroom', 'size', 'age']]
y = df['price']
model = LinearRegression()
model.fit(X, y)
bedroom = int(input("Enter bedroom: "))
size = int(input("Enter house size: "))
age = int(input("Enter house age: "))
new_house = pd.DataFrame({
    'bedroom': [bedroom],
    'size': [size],
    'age': [age]
})
pred = model.predict(new_house)
print("Predicted price:", pred[0])
y_pred = model.predict(X)
plt.scatter(y, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression")
plt.show()
