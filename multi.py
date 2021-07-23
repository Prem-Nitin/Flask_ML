import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def predict(x1,x2,x3, dat):
    df= pd.read_csv('./static/{}'.format(dat))

    x=df.iloc[:,:-1]
    y=df.iloc[:,3]

    x_train,x_test,y_train,y_test = train_test_split (x,y, test_size=1/10, random_state=0)
   
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    youtube = x1
    facebook = x2
    newspaper = x3
    X_test = [[youtube, facebook, newspaper]]

    
    res = regressor.predict(X_test)
    result = res[0]
    print(res[0])
    return result
