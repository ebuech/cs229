# Lily Buechler, CS 229 (Fall 2018)

import pandas
import os
import glob
import re
from sklearn.cluster import KMeans
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.svm
import statsmodels.api as sm
from sklearn.feature_selection import RFE
import numpy as np
import sklearn.neighbors


def LR_feature_select(x_train,x_test,x_dev,y_train,y_test,y_dev,num_features):
    '''Linear regression with recursive feature selection'''

    
    estimator=sklearn.linear_model.LinearRegression()
    selector=RFE(estimator,num_features,step=1)
    selector=selector.fit(x_train,np.ndarray.flatten(y_train))
    
    x_train_new=x_train[:,selector.support_]
    x_test_new=x_test[:,selector.support_]
    x_dev_new=x_dev[:,selector.support_]
    
    lr=sklearn.linear_model.LinearRegression().fit(x_train_new,y_train)
    pred_dev=lr.predict(x_dev_new)
    pred_train=lr.predict(x_train_new)
    pred_test=lr.predict(x_test_new)

    MSE_train=np.dot(np.ndarray.flatten(pred_train-y_train),np.ndarray.flatten(pred_train-y_train))*(1/len(pred_train))    
    MSE_dev=np.dot(np.ndarray.flatten(pred_dev-y_dev),np.ndarray.flatten(pred_dev-y_dev))*(1/len(pred_dev))
    MSE_test=np.dot(np.ndarray.flatten(pred_test-y_test),np.ndarray.flatten(pred_test-y_test))*(1/len(pred_test))
        
    return MSE_test,MSE_train,MSE_dev

def knn_fit(x_train,x_test,x_dev,y_train,y_test,y_dev,num_neighbors):
    '''KNN regression'''

    knn=sklearn.neighbors.KNeighborsRegressor(n_neighbors=num_neighbors).fit(x_train,y_train)
    pred_test=knn.predict(x_test)
    pred_train=knn.predict(x_train)
    pred_dev=knn.predict(x_dev)
    
    MSE_train=np.dot(np.ndarray.flatten(pred_train-y_train),np.ndarray.flatten(pred_train-y_train))*(1/len(pred_train))
    MSE_dev=np.dot(np.ndarray.flatten(pred_dev-y_dev),np.ndarray.flatten(pred_dev-y_dev))*(1/len(pred_dev))
    MSE_test=np.dot(np.ndarray.flatten(pred_test-y_test),np.ndarray.flatten(pred_test-y_test))*(1/len(pred_test))

    return MSE_test,MSE_train,MSE_dev
    
def RF_feature_select(x_train,x_test,x_dev,y_train,y_test,y_dev,num_features,n_est,max_depth_rf):
    '''Random forests with recursive feature selection'''
    
    
    estimator=sklearn.ensemble.RandomForestRegressor(n_estimators=n_est,max_depth=max_depth_rf)
    selector=RFE(estimator,num_features,step=1)
    selector=selector.fit(x_train,np.ndarray.flatten(y_train))
    
    x_train_new=x_train[:,selector.support_]
    x_test_new=x_test[:,selector.support_]
    x_dev_new=x_dev[:,selector.support_]
    
    rf=sklearn.ensemble.RandomForestRegressor(n_estimators=n_est,max_depth=max_depth_rf).fit(x_train_new,np.ndarray.flatten(y_train))
    
    pred_dev=rf.predict(x_dev_new)
    pred_test=rf.predict(x_test_new)
    pred_train=rf.predict(x_train_new)
    
    MSE_train=np.dot(np.ndarray.flatten(pred_train-y_train),np.ndarray.flatten(pred_train-y_train))*(1/len(pred_train))
    MSE_dev=np.dot(np.ndarray.flatten(pred_dev-y_dev),np.ndarray.flatten(pred_dev-y_dev))*(1/len(pred_dev))
    MSE_test=np.dot(np.ndarray.flatten(pred_test-y_test),np.ndarray.flatten(pred_test-y_test))*(1/len(pred_test))

    return MSE_test,MSE_train,MSE_dev


