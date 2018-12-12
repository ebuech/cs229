# Lily Buechler, CS 229 (Fall 2018)

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


#%% Perform hierachical clustering on both training set and dev set

os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\start_clusters_hier')

load_type_list=['clotheswasher1','dishwasher1','drye1','car1','waterheater1','clotheswasher_dryg1']
label_list=['Clotheswasher','Dishwasher','Dryer','EV','Waterheater','Clotheswasher + dryer']
num_clust_vec=np.arange(2,15)
save_clusters=False

cluster_labels_train_df,cluster_labels_dev_df=cluster_starts_functions.Hier_TrainOnTrain_TrainOnDev(load_type_list,label_list,num_clust_vec,save_clusters,data_train_drye1,data_dev_drye1,data_train_dishwasher1,data_dev_dishwasher1,data_train_clotheswasher1,data_dev_clotheswasher1,data_train_car1,data_dev_car1,data_train_waterheater1,data_dev_waterheater1,data_train_clotheswasher_dryg1,data_dev_clotheswasher_dryg1)

    
#%% Perform hierachical clustering on training set and use to predict on dev set

os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\start_clusters_hier')

load_type_list=['clotheswasher1','dishwasher1','drye1','car1','waterheater1','clotheswasher_dryg1']
label_list=['Clotheswasher','Dishwasher','Dryer','EV','Waterheater','Clotheswasher + dryer']
num_clust_vec=np.arange(2,15)
save_clusters=False

cluster_labels_train_df,cluster_labels_dev_df=cluster_starts_functions.Hier_TrainOnTrain_TestOnDev(load_type_list,label_list,num_clust_vec,save_clusters,data_train_drye1,data_dev_drye1,data_train_dishwasher1,data_dev_dishwasher1,data_train_clotheswasher1,data_dev_clotheswasher1,data_train_car1,data_dev_car1,data_train_waterheater1,data_dev_waterheater1,data_train_clotheswasher_dryg1,data_dev_clotheswasher_dryg1)

