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
import flexibility_models_functions
import sklearn.neighbors

#%% Load data
os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features')
x_vec_pred=pandas.read_csv('flexibility_features_v2.csv',header=None)
x_vec_pred=np.array(x_vec_pred)
x_vec_pred=x_vec_pred[:,1:]


flex_vec=np.loadtxt('flexibility_values_v2.txt', dtype=float).reshape(-1,1)

x_train=x_vec_pred[0:20,:]
y_train=flex_vec[0:20,:]
x_dev=x_vec_pred[20:26,:]
y_dev=flex_vec[20:26,:]
x_test=x_vec_pred[26:32,:]
y_test=flex_vec[26:32,:]


#%% plain linear regression with feature selection

num_features=5
MSE_test,MSE_train,MSE_dev=flexibility_models_functions.LR_feature_select(x_train,x_test,x_dev,y_train,y_test,y_dev,num_features)


#%% KNN

num_neighbors=5
MSE_test,MSE_train,MSE_dev=flexibility_models_functions.knn_fit(x_train,x_test,x_dev,y_train,y_test,y_dev,num_neighbors)


#%% Random Forest with feature selection

n_est=1000
max_depth_rf=10
num_features=12
MSE_test,MSE_train,MSE_dev=flexibility_models_functions.RF_feature_select(x_train,x_test,x_dev,y_train,y_test,y_dev,num_features,n_est,max_depth_rf)
