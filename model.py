import pandas as pd

df=pd.read_csv('diabetes_data.csv')

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

df['gender']=labelencoder.fit_transform(df['gender'])
df['smoking_history']=labelencoder.fit_transform(df['smoking_history'])

x=df.drop(['diabetes'],axis=1)
y=df['diabetes']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()

logmodel.fit(x_train,y_train)
print(logmodel.score(x,y))

import pickle

pickle.dump(logmodel,open('model1.pkl','wb'))

