# Lily Buechler, CS 229 (Fall 2018)

#%%

import glob
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import re
import calculate_start_distributions_functions

   
#%% Get IDs of homes with data for 3 different datasets

load_type='clotheswasher1'
int_list=calculate_start_distributions_functions.get_house_list_intersection(load_type)

#%% Calculate probability distribution of start times with laplace smoothing for intersection of data sets


set_type='dev'
#load_type_list=['clotheswasher1','clotheswasher_dryg1','drye1','dishwasher1','waterheater1','car1']
load_type_list=['waterheater1']
save_data=False

calculate_start_distributions_functions.get_start_prob_dist_intersection(set_type,load_type_list,save_data,int_list)

