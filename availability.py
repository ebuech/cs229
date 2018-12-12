#%%

import numpy as np
import pandas
import os
import matplotlib.pyplot as plt
import scipy
import scipy.cluster
import networkx as nx
from networkx.algorithms import bipartite
import availability_functions

#%% plot increase in availability based on increase in percentile of load duration curve
set_type='train'

perc=0.95
load_type_list=['clotheswasher1','dishwasher1','drye1','car1','waterheater1','clotheswasher_dryg1']
load_type_list_nice=['Clotheswasher','Dishwasher','Dryer','EV','Waterheater','Clotheswasher+dryer']

availability_functions.plot_percLDC(perc,load_type_list,load_type_list_nice,set_type)



#%% plot increase in availability based on increase in hourly mean load from consumer segmentation
set_type='train'


load_type_list=['clotheswasher1','dishwasher1','drye1','car1','waterheater1','clotheswasher_dryg1']
load_type_list_nice=['Clotheswasher','Dishwasher','Dryer','EV','Waterheater','Clotheswasher+dryer']


availability_functions.plot_max_mean_hr_load(load_type_list,load_type_list_nice,set_type)

    
#%% Plot the variance of cluster size for different numbers of clusters for different algorithms

set_type='train'
load_type='clotheswasher1'

availability_functions.plot_cluster_size_var(set_type,load_type)
