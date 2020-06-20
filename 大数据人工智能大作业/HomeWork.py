import pandas as pd
import os
from math import sqrt

from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LinearRegression
if __name__=='__main__':
    dataset_path=os.path.join('~/s3data/dataset','housing.csv')#KAGGLE上的boston房价数据
    df=pd.read_csv(dataset_path)
    # pd.set_option('display.max_columns',None)
    # print(df.head(10))

    y=df.get('MEDV')
    X=df.drop('MEDV',axis=1)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    lr=LinearRegression()#线性回归
    model=lr.fit(X_train,y_train)
    # model_path=os.path.join('../model','lr.pkl')
    # joblib.dump(model,model_path,compress=3)

    y_pred=model.predict(X_test)
    MSE=metrics.mean_squared_error(y_test,y_pred)
    RMSE=sqrt(MSE)
    print(f'RMSE={RMSE}')
    y_pred_pd=pd.DataFrame(y_pred)
    y_pred_pd.to_csv("prediction.csv",index=False)
