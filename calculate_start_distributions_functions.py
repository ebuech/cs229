# Lily Buechler, CS 229 (Fall 2018)

import glob
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import re

def get_house_list_intersection(load_type):
    os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features')
   
    file_list=glob.glob("starts_test_*_"+load_type+".csv")
    dataid_list_test=list()
    for i in range(len(file_list)):
        dataid=re.search('starts_test_(.+?)_'+load_type+'.csv', file_list[i])[1]
        dataid_list_test.append(dataid)
    
    file_list=glob.glob("starts_train_*_"+load_type+".csv")
    dataid_list_train=list()
    for i in range(len(file_list)):
        dataid=re.search('starts_train_(.+?)_'+load_type+'.csv', file_list[i])[1]
        dataid_list_train.append(dataid)
    
    file_list=glob.glob("starts_dev_*_"+load_type+".csv")
    dataid_list_dev=list()
    for i in range(len(file_list)):
        dataid=re.search('starts_dev_(.+?)_'+load_type+'.csv', file_list[i])[1]
        dataid_list_dev.append(dataid)
    
    # get intersection of data for 3 lists
    
    int_list=list(set(dataid_list_train).intersection(set(dataid_list_dev).intersection(set(dataid_list_test))))
    return int_list

def get_start_prob_dist_intersection(set_type,load_type_list,save_data,int_list):
    os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features')
    for load_type in load_type_list:
        
        file_list=glob.glob("starts_"+set_type+"_*_"+load_type+".csv")
        
        dist=np.zeros((24,1))
        
        k=0
        for i in range(len(file_list)):
            dataid=re.search('starts_'+set_type+'_(.+?)_'+load_type+'.csv', file_list[i])[1]
            data_df = pandas.read_csv(file_list[i],header=0)
            if dataid in int_list:
            
                for h in range(24):
                    dist[h,0]=(len(data_df[data_df.hour==h].hour)+1)/(len(data_df.hour)+24)
                if k==0:
                    start_dist=pandas.DataFrame({dataid:np.ndarray.flatten(dist)})
                else:
                    start_dist[dataid]=dist
                k=k+1
        if save_data==True:
            start_dist.to_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',index=False)