import numpy as np
from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
import pandas as pd 
from sklearn.model_selection import train_test_split

def predict(X, dat):
    
    df=pd.read_csv('./static/{}'.format(dat))
    #selecting index 
    x= df.iloc[:,0]
    y=df.iloc[:,1]
    #testing training values

     
    x_train,x_test,y_train,y_test = train_test_split (x,y, test_size=1/3.0, random_state=0)
    #we need to reshape the array as it exceeds 2d 
    x_train =x_train.values.reshape(-1,1)

    
    regressor = LinearRegression()
    regressor.fit(x_train,y_train)


    a = [[X]]
    result = regressor.predict(a)
    result1 = result[0]
    return result1