import pandas as pd 
import numpy as np
dataset=pd.read_csv("supplement.csv")
dataset
dataset.info()
dataset.isnull().sum()
dataset.describe()
#Pip install plotly
import plotly.express as px
pie=dataset["Store_Type"].value_counts()
store=pie.index
orders=pie.values
fig=px.pie(dataset,values=orders,names=store)
fig.show()
pie2=dataset["Location_Type"].value_counts()
location=pie2.index
orders=pie2.values
fig=px.pie(dataset,values=orders,names=location)
fig.show()
pie3=dataset["Discount"].value_counts()
discount=pie3.index
orders=pie3.values
fig=px.pie(dataset,values=orders,names=discount)
fig.show()
pie4=dataset["Holiday"].value_counts()
holiday=pie4.index
orders=pie4.values
fig=px.pie(dataset,values=orders,names=holiday)
fig.show()
#Replace the Discount value
dataset["Discount"]=dataset["Discount"].map({"No":0,"Yes":1})
dataset
#Replace the Store_Type value
dataset["Store_Type"]=dataset["Store_Type"].map({"S1":1,"S2":2,"S3":3,"S4":4})
dataset
#Replace the Location_Type value
dataset["Location_Type"]=dataset["Location_Type"].map({"L1":1,"L2":2,"L3":3,"L4":4,"L5":5})
dataset
X=np.array(dataset[["Store_Type","Location_Type","Holiday","Discount"]])
y=np.array(dataset["#Order"])
X
y
#Building the ML model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
len(X_train)
#pip install lightgbm
import lightgbm as ltb
model=ltb.LGBMRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_pred
y_test
data=pd.DataFrame(data=("Predicted Orders",y_pred.flatten()))
data
