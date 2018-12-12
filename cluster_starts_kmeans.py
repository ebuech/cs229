#%%

import numpy as np
import pandas
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy
import scipy.cluster
import cluster_starts_functions
import networkx as nx
from networkx.algorithms import bipartite
import copy

#%% Load data

data_train_drye1,data_dev_drye1,data_train_dishwasher1,data_dev_dishwasher1,data_train_clotheswasher1,data_dev_clotheswasher1,data_train_car1,data_dev_car1,data_train_waterheater1,data_dev_waterheater1,data_train_clotheswasher_dryg1,data_dev_clotheswasher_dryg1=cluster_starts_functions.load_start_distributions()


#%% run Kmeans on both training and dev set

os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\start_clusters_kmeans')

load_type_list=['clotheswasher1','dishwasher1','drye1','car1','waterheater1','clotheswasher_dryg1']
label_list=['Clotheswasher','Dishwasher','Dryer','EV','Waterheater','Clotheswasher + dryer']
save_clusters=False
num_clust_vec=np.arange(2,15)

cluster_labels_train, cluster_labels_dev=cluster_starts_functions.Kmeans_TrainOnTrain_TrainOnDev(load_type_list,label_list,save_clusters,num_clust_vec,data_train_drye1,data_dev_drye1,data_train_dishwasher1,data_dev_dishwasher1,data_train_clotheswasher1,data_dev_clotheswasher1,data_train_car1,data_dev_car1,data_train_waterheater1,data_dev_waterheater1,data_train_clotheswasher_dryg1,data_dev_clotheswasher_dryg1)

    
#%% Run kmeans on training set and predict clusters on dev set

os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\start_clusters_kmeans')

load_type_list=['clotheswasher1','dishwasher1','drye1','car1','waterheater1','clotheswasher_dryg1']
label_list=['Clotheswasher','Dishwasher','Dryer','EV','Waterheater','Clotheswasher + dryer']
num_clust_vec=np.arange(2,15)
save_clusters=False

cluster_lanels_train_df,cluster_labels_dev_df=cluster_starts_functions.Kmeans_TrainOnTrain_TestOnDev(load_type_list,label_list,save_clusters,num_clust_vec,data_train_drye1,data_dev_drye1,data_train_dishwasher1,data_dev_dishwasher1,data_train_clotheswasher1,data_dev_clotheswasher1,data_train_car1,data_dev_car1,data_train_waterheater1,data_dev_waterheater1,data_train_clotheswasher_dryg1,data_dev_clotheswasher_dryg1)

    