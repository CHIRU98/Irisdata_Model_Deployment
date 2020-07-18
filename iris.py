import pandas as pd
import numpy as np
import pickle
data = pd.read_csv(r"C:\Users\user\Downloads\iris (1).data")
data.columns = ["sepal length","sepal width","petal length","petal width","class"]

x = data.iloc[:,:-1].values
y = data["class"]

from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
y = number.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.30)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

print(lr.score(x_train,y_train))

pickle.dump(lr,open('model.pickle','wb'))