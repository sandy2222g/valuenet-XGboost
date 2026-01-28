import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)


df = df.rename(columns={"medv": "price"})

df.isnull().sum()


df.describe()

co=df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(co,cbar=True, square=True,fmt=".2f",annot=True,annot_kws={'size':10},cmap="Greens")


x=df.drop(['price'],axis=1)
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
model=XGBRegressor()
model.fit(x_train,y_train)



train_prediction=model.predict(x_train)


s1=metrics.r2_score(y_train,train_prediction)
s2=metrics.mean_absolute_error(y_train,train_prediction)
print("R squared error:",s1)
print("mean absolute error:",s1)

test_prediction=model.predict(x_test)
s1=metrics.r2_score(y_test,test_prediction)
s2=metrics.mean_absolute_error(y_test,test_prediction)
print("R squared error",s1)
print("mean absolute error",s1)


plt.scatter(y_train,train_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted prices")