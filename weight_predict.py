import pandas as pd   
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import model_selection


label_enc = preprocessing.LabelEncoder()
data=pd.read_csv("weight-height.csv")

print(data.columns)

data1=data['Gender']

label_enc.fit(data1)

data['Gender']=label_enc.transform(data['Gender'])


print(data.head())

Y=data['Weight']
data.drop(['Weight'],axis=1,inplace=True)

X_train,X_val,Y_train,Y_val=model_selection.train_test_split(data,Y,test_size=0.20)

lr=LinearRegression()
lr.fit(X_train,Y_train)

print(lr.score(X_val,Y_val))
