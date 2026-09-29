import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
data={
    'Age':[25,30,35,40,45,50,55,60],
    'Gender':[0,0,1,0,1,0,1,1],
    'BMI':[22,24,28,27,30,32,34,36],
    'BP':[120,125,140,135,150,160,170,180],
    'Cholesterol':[180,190,220,210,240,260,280,300],
    'Disease':[0,0,1,0,1,1,1,1]
}
df=pd.DataFrame(data)
x=df[['Age']]
y=df['Disease']
model=LogisticRegression()
model.fit(x,y)
Age=int(input("Enter Age:"))
Gender=int(input("Enter Gender (Male=1 Female=0):"))
BMI=float(input("Enter BMI:"))
BP=int(input("Enter Blood Pressure:"))
Cholesterol=int(input("Enter Cholesterol:"))
new_data=pd.DataFrame({'Age':[Age]})
Prediction=model.predict(new_data)
if Prediction[0]==1:
    print("Disease Detected")
else:
    print("No Disease")
x_values=np.linspace(df['Age'].min(),df['Age'].max(),200)
x_df=pd.DataFrame({'Age':x_values})
y_prob=model.predict_proba(x_df)[:,1]
plt.figure(figsize=(6,4))
plt.scatter(df['Age'],y,color='blue',label='Actual Data')
plt.plot(x_values,y_prob,color='red',linewidth=2,label='Logistic Regression Curve')
plt.xlabel('Age')
plt.ylabel("Probability")
plt.title("Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()
