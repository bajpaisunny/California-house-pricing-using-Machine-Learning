#This project is created by Bajpai Sunny, might put on git hub soon! :)
#I have explained the project line by line below to help you out.


#Make sure you have this libaries installed

from tkinter import N
import pandas as pd
import numpy as np
import sklearn


"""
Now we will load the data that we will be using to train our model.
Here we are using california house pricing dataset which is available in the skleanr libary itself.
"""

#Little context about dataset, it has 14 variables based on which we have to predict pricing


from sklearn.datasets import fetch_california_housing
df = fetch_california_housing()

df.keys()

''' 
print(df.data)
'''
#Remove comment below to check dataset of project
#print(df.DESCR) #Information about datasets

#Convert dict into dataframe using pandas
california = pd.DataFrame(df.data, columns=df.feature_names)
print("Dataset")
print(california.head(5))


print("Dataset with target values")
#Adding column of target values
california['MEDV'] = df.target
print(california.head())

print("To check if null value is present or not")
#Must check if there is null value or not,if it null = true else false
print(california.isnull())
print("Check null value if 0 is non & if 1 it is there")
print(california.isnull().sum())
print()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = california.drop('MEDV', axis =1)
Y = california['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 1)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

print()
print()
print("Model accuracy")
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))

print("The model performance for training set")
print('RMSE is {}'.format(rmse))






