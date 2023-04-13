import sklearn
from sklearn.tree import DecisionTreeClassifier as dtc
import pandas as pd
boo=pd.read_csv("seattle-weather.csv")

x=boo.drop(columns=['weather','date'])
y=boo['weather']
model=dtc()
model.fit(x,y)
predictions=model.predict([[5,6.1,-20.6,45.8]])
print(predictions)