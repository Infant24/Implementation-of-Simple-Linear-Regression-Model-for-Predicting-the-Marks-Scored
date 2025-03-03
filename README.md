# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 2.Set variables for assigning dataset values.

2.Import linear regression from sklearn.

3.Assign the points for representing in the graph.

4.Predict the regression for marks by using the representation of the graph.

5.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Infant Maria Stefanie .F
RegisterNumber: 212224230095

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 


*/
```

## Output:

![Screenshot 2025-03-03 222240](https://github.com/user-attachments/assets/61346c83-2829-45ed-bb8a-7c4dc651b7e8)
![Screenshot 2025-03-03 222316](https://github.com/user-attachments/assets/5e7df2b6-d4ae-41b9-81b9-8fd4cea415e4)
![Screenshot 2025-03-03 222325](https://github.com/user-attachments/assets/dd624c2f-c678-4d79-a518-c5e0d425cd47)
![Screenshot 2025-03-03 222344](https://github.com/user-attachments/assets/2b31ff18-04e7-4666-93dd-8dff915df4f7)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
