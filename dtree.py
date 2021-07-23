import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def dec_predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,dat):
    df = pd.read_csv('./static/{}'.format(dat))
    a = df.columns.to_list()
    print(df)
    categor_condn=[ (df['quality']<=5)]
    rating=['worst']
    df['rating'] = np.select(categor_condn,rating,default='best')
    df = df.drop(['quality'], axis='columns')

    
    
    
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=42)
    
    model = DecisionTreeClassifier(criterion='entropy',max_depth=10)
    model.fit(X_train, Y_train)

    x = [[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]]
    y_pred = model.predict(x)
    result = y_pred[0]
    print(result)
    
    return result
