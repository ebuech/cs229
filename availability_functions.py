import numpy as np
import pandas
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy
import scipy.cluster
import copy


def get_percinc_LDC(set_type,load_type,clust_method,perc,num_clust_vec):
    '''Calculate increase in percentile of hourly load duration curve as a result of consumer 
    segmentation
    
    set_type='train','dev', or 'test'
    load_type = 'clotheswasher1','dishwasher1','drye1','car1','waterheater1', or 'clotheswasher_dryg1'
    clust_method = 'kmeans','hier','GMM', or 'LDA'
    perc = percentile of load duration curve
    num_clust_vec = vector of number of clusters to evaluate
    
    '''
    print(load_type)
    print(clust_method)
    os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features')
    
    power_data = pandas.read_csv(str(load_type)+'_data_'+'test'+'.csv',header=0)
    
    #Replace any NANs in raw data
    null_elem=power_data.isnull().any()
    
    for i in range(len(null_elem)):
        if null_elem[i]==True:
            power_data[power_data.columns[i]][power_data[power_data.columns[i]].isnull()==True]=0
    
    if clust_method=='kmeans'or clust_method=='hier':
        os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\start_clusters_'+clust_method)
        clust_labels_matrix = pandas.read_csv('start_clusters_'+set_type+'_'+str(load_type)+'_'+clust_method+'.csv',header=0)
    elif clust_method=='GMM':
        os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\\trainedOntrainAndDev_2\GMM')
        clust_labels_matrix = pandas.read_csv('GMM_'+set_type+'_'+load_type[0:len(load_type)-1]+'.csv',header=None)
    elif clust_method=='LDA':
        os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\\trainedOntrainAndDev_2\LDA')
        clust_labels_matrix = pandas.read_csv(set_type+'_'+load_type[0:len(load_type)-1]+'.csv',header=None)        
    
    os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features\start_distributions_final')
    house_id_vec=pandas.read_csv('dist_'+set_type+'_'+load_type+'_ls.csv',header=0)
    
    clust_labels_matrix_np=np.array(clust_labels_matrix)
    
        
    #Reorder the columns for the timeseries data according to the ordering of the cluster assignments
    power_data_new=power_data[['month','day','hour','minute','weekday','timestamp']]
    for id_x in house_id_vec.columns:
        power_data_new[id_x]=power_data[id_x]
        
    power_data_array=np.array(power_data_new[power_data_new.columns[6:power_data_new.shape[1]]])
    
    #Get total load for all homes in the dataset
    agg_power_net=np.sum(power_data_array,axis=1).reshape(-1,1)

    
    it=0
    
    perc_inc_LDC=np.zeros((len(num_clust_vec),24))
    
    for num_clust in num_clust_vec:
        print(num_clust)
        clust_label_temp=clust_labels_matrix_np[it,:]
        
        agg_power=np.zeros((power_data_array.shape[0],num_clust))
        
        clust_id=0
        k=0
        for clust_id in range(num_clust):
            agg_power[:,k]=np.sum(power_data_array[:,clust_label_temp==clust_id],axis=1)
            k=k+1
    
        for hr in range(24):
            agg_power_hour=agg_power[power_data_new.hour==hr]
            agg_power_net_hour=agg_power_net[power_data_new.hour==hr]        
            for cl in range(num_clust): 
                if len(clust_label_temp[clust_label_temp==cl])>0.1*len(clust_label_temp):
                    sorted_power_net=np.array(sorted(agg_power_net_hour[:,0],reverse=True))/len(clust_label_temp)
                    sorted_power=np.array(sorted(agg_power_hour[:,cl],reverse=True))/len(clust_label_temp[clust_label_temp==cl])
                                          
                    perc_inc_temp=(sorted_power[int(len(sorted_power)*(1-perc))]-sorted_power_net[int(len(sorted_power_net)*(1-perc))])/sorted_power_net[int(len(sorted_power_net)*(1-perc))]
                
                    if perc_inc_temp>perc_inc_LDC[it,hr]:
                        perc_inc_LDC[it,hr]=perc_inc_temp
        it=it+1
                   
    return perc_inc_LDC


def plot_percLDC(perc,load_type_list,load_type_list_nice,set_type):
    '''Plot results from get_perc_inc_LDC function'''

    plt.figure(figsize=(8,7))
    l_num=0
    for load_type in load_type_list:
    
        plt.subplot(3,2,l_num+1)    
        
        clust_method='kmeans'
        num_clust_vec=np.arange(2,15)
        perc_inc_LDC=get_percinc_LDC(set_type,load_type,clust_method,perc,num_clust_vec)
                   
        for i in range(16,18):
            if i==16:
                c='b'
            elif i==17:
                c='r'
            plt.plot(num_clust_vec,perc_inc_LDC[:,i],'.-',label=i,c=c)
    
        clust_method='hier'
        num_clust_vec=np.arange(2,15)
        perc_inc_LDC=get_percinc_LDC(set_type,load_type,clust_method,perc,num_clust_vec)
                        
        for i in range(16,18):
            if i==16:
                c='b'
            elif i==17:
                c='r'
            plt.plot(num_clust_vec,perc_inc_LDC[:,i],'-.',label=i,c=c)
        
        
        clust_method='GMM'
        num_clust_vec=np.arange(2,8)
        perc_inc_LDC=get_percinc_LDC(set_type,load_type,clust_method,perc,num_clust_vec)
                        
        for i in range(16,18):
            if i==16:
                c='b'
            elif i==17:
                c='r'
            plt.plot(num_clust_vec,perc_inc_LDC[:,i],'--',label=i,c=c)
    
        
        
        clust_method='LDA'
        num_clust_vec=np.arange(2,8)
        perc_inc_LDC=get_percinc_LDC(set_type,load_type,clust_method,perc,num_clust_vec)
                        
        for i in range(16,18):
            if i==16:
                c='b'
            elif i==17:
                c='r'
            plt.plot(num_clust_vec,perc_inc_LDC[:,i],':',label=i,c=c)
            
        if l_num==4 or l_num==5:
            plt.xlabel('Number of clusters',fontsize=14)
        if l_num==0 or l_num==2 or l_num==4:
            plt.ylabel('Factor increase in \n hourly '+'{:.0f}'.format((perc)*100)+'%ile LDC',fontsize=14)
        ax=plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        #plt.legend()
        plt.title(load_type_list_nice[l_num],fontsize=14)    
    
        l_num=l_num+1
    
    #ax.legend(loc=2,fontsize=14,bbox_to_anchor=(1.2,0.3))
        
    plt.tight_layout()




def get_max_mean_hr_load(set_type,load_type,clust_method,num_clust_vec):
    '''
    Calculate the increase in mean load for each cluster and appliance type and hour as a result 
    of consumer segmentation
    
    set_type='train','dev', or 'test'
    load_type='clotheswasher1','dishwasher1','drye1','car1','waterheater1', or 'clotheswasher_dryg1'
    clust_method = 'kmeans','hier','GMM', or 'LDA'
    perc = percentile of load duration curve
    num_clust_vec = vector of number of clusters to evaluate
    '''

    print(load_type)
    print(clust_method)
    os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features')
    
    power_data = pandas.read_csv(str(load_type)+'_data_'+'test'+'.csv',header=0)
    
    #Replace any NANs in timeseries data
    null_elem=power_data.isnull().any()
    
    for i in range(len(null_elem)):
        if null_elem[i]==True:
            power_data[power_data.columns[i]][power_data[power_data.columns[i]].isnull()==True]=0
    
    #Load cluster assignements
    if clust_method=='kmeans'or clust_method=='hier':
        os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\start_clusters_'+clust_method)
        clust_labels_matrix = pandas.read_csv('start_clusters_'+set_type+'_'+str(load_type)+'_'+clust_method+'.csv',header=0)
    elif clust_method=='GMM':
        os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\\trainedOntrainAndDev_2\GMM')
        clust_labels_matrix = pandas.read_csv('GMM_'+set_type+'_'+load_type[0:len(load_type)-1]+'.csv',header=None)
    elif clust_method=='LDA':
        os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\\trainedOntrainAndDev_2\LDA')
        clust_labels_matrix = pandas.read_csv(set_type+'_'+load_type[0:len(load_type)-1]+'.csv',header=None)        
    
    os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features\start_distributions_final')
    house_id_vec=pandas.read_csv('dist_'+set_type+'_'+load_type+'_ls.csv',header=0)
    
    clust_labels_matrix_np=np.array(clust_labels_matrix)
    
        
    #Put timeseries data in same order as cluster label data
    power_data_new=power_data[['month','day','hour','minute','weekday','timestamp']]
    #for id_x in clust_labels_matrix.columns:
    for id_x in house_id_vec.columns:
        power_data_new[id_x]=power_data[id_x]
        
    power_data_array=np.array(power_data_new[power_data_new.columns[6:power_data_new.shape[1]]])
    
    #Get total timeseries power consumption for all homes in dataset
    agg_power_net=np.sum(power_data_array,axis=1).reshape(-1,1)
    
    
    it=0
    
    avail=np.zeros((len(num_clust_vec),24))
    
    for num_clust in num_clust_vec:
        print(num_clust)
        clust_label_temp=clust_labels_matrix_np[it,:]
        
        agg_power=np.zeros((power_data_array.shape[0],num_clust))
        
        clust_id=0
        k=0
        for clust_id in range(num_clust):
            agg_power[:,k]=np.sum(power_data_array[:,clust_label_temp==clust_id],axis=1)
            k=k+1
    
        for hr in range(24):
            agg_power_hour=agg_power[power_data_new.hour==hr]
            agg_power_net_hour=agg_power_net[power_data_new.hour==hr]        
            for cl in range(num_clust): 
                if len(clust_label_temp[clust_label_temp==cl])>0.1*len(clust_label_temp):
                    
                    mean_net=np.mean(agg_power_net_hour/len(clust_label_temp))
                    mean_clust=np.mean(agg_power_hour[:,cl]/len(clust_label_temp[clust_label_temp==cl]))
                    avail_temp=mean_clust/mean_net
                    if avail_temp>avail[it,hr]:
                        avail[it,hr]=avail_temp                   

        it=it+1
                   
    return avail

def plot_max_mean_hr_load(load_type_list,load_type_list_nice,set_type):
    '''Plot results from get_max_mean_hr_load() function'''

    plt.figure(figsize=(8,7))
    l_num=0
    for load_type in load_type_list:
    
        plt.subplot(3,2,l_num+1)    
        
        clust_method='kmeans'
        num_clust_vec=np.arange(2,13)
        perc_inc_LDC=get_max_mean_hr_load(set_type,load_type,clust_method,num_clust_vec)
                   
        for i in range(16,18):
            if i==16:
                c='b'
            elif i==17:
                c='r'
            plt.plot(num_clust_vec,perc_inc_LDC[:,i],'.-',label=i,c=c)
    
        clust_method='hier'
        num_clust_vec=np.arange(2,13)
        perc_inc_LDC=get_max_mean_hr_load(set_type,load_type,clust_method,num_clust_vec)
                        
        for i in range(16,18):
            if i==16:
                c='b'
            elif i==17:
                c='r'
            plt.plot(num_clust_vec,perc_inc_LDC[:,i],'-.',label=i,c=c)
        
        
        clust_method='GMM'
        num_clust_vec=np.arange(2,8)
        perc_inc_LDC=get_max_mean_hr_load(set_type,load_type,clust_method,num_clust_vec)
                        
        for i in range(16,18):
            if i==16:
                c='b'
            elif i==17:
                c='r'
            plt.plot(num_clust_vec,perc_inc_LDC[:,i],'--',label=i,c=c)
    
        
        
        clust_method='LDA'
        num_clust_vec=np.arange(2,8)
        perc_inc_LDC=get_max_mean_hr_load(set_type,load_type,clust_method,num_clust_vec)
                        
        for i in range(16,18):
            if i==16:
                c='b'
            elif i==17:
                c='r'
            plt.plot(num_clust_vec,perc_inc_LDC[:,i],':',label=i,c=c)
            
        if l_num==4 or l_num==5:
            plt.xlabel('Number of clusters',fontsize=14)
        if l_num==0 or l_num==2 or l_num==4:
            plt.ylabel('Factor increase \n in mean load',fontsize=14)
        ax=plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        #plt.legend()
        plt.title(load_type_list_nice[l_num],fontsize=14)    
        plt.xticks(np.arange(2,13,2))
        if l_num==4:
            plt.ylim([0.5,2])
    
        l_num=l_num+1
    
    #ax.legend(loc=2,fontsize=14,bbox_to_anchor=(1.2,0.3))
        
    plt.tight_layout()

def count_cluster_size(set_type,load_type,clust_method,num_clust_vec):
    '''
    Get cluster sizes for different methods 
    
    set_type='train','dev', or 'test'
    load_type='clotheswasher1','dishwasher1','drye1','car1','waterheater1', or 'clotheswasher_dryg1'
    clust_method = 'kmeans','hier','GMM', or 'LDA'
    num_clust_vec = vector of number of clusters to evaluate
    
    Return list of list of cluster sizes, where each sublist corresponds with an different number of 
    clusters in num_clust_vec
    '''
    
    print(load_type)
    print(clust_method)
    os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features')
    
    if clust_method=='kmeans'or clust_method=='hier':
        os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\start_clusters_'+clust_method)
        clust_labels_matrix = pandas.read_csv('start_clusters_'+set_type+'_'+str(load_type)+'_'+clust_method+'.csv',header=0)
    elif clust_method=='GMM':
        os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\\trainedOntrainAndDev_2\GMM')
        clust_labels_matrix = pandas.read_csv('GMM_'+set_type+'_'+load_type[0:len(load_type)-1]+'.csv',header=None)
    elif clust_method=='LDA':
        os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\\trainedOntrainAndDev_2\LDA')
        clust_labels_matrix = pandas.read_csv(set_type+'_'+load_type[0:len(load_type)-1]+'.csv',header=None)        
    
    os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features\start_distributions_final')
    house_id_vec=pandas.read_csv('dist_'+set_type+'_'+load_type+'_ls.csv',header=0)
    
    clust_labels_matrix_np=np.array(clust_labels_matrix)
    
    size_clust_list=list()
    
    k=0
    for num_clust in num_clust_vec:
        clust_label_temp=clust_labels_matrix_np[k,:]

        size_clust_list.append(list())
        for i in range(num_clust):
            size_clust_list[k].append(len(clust_label_temp[clust_label_temp==i]))
            
        k=k+1
    m=len(clust_label_temp)
    return size_clust_list,m


def plot_cluster_size_var(set_type,load_type):
    '''Plot the variance of cluster size with the number of clusters for different methods'''
    
    clust_method='hier'
    num_clust_vec=np.arange(2,15)
    size_clust_list,m=count_cluster_size(set_type,load_type,clust_method,num_clust_vec)
    
    k=0
    for num_clust in num_clust_vec:
        if num_clust==2:
            plt.plot(num_clust,np.var(np.array(size_clust_list[k])),'x',c='b',label='Hierarchical')
        else:
            plt.plot(num_clust,np.var(np.array(size_clust_list[k])),'x',c='b')
            
        k=k+1
        
        
        
    clust_method='kmeans'
    num_clust_vec=np.arange(2,15)
    size_clust_list,m=count_cluster_size(set_type,load_type,clust_method,num_clust_vec)
    
    k=0
    for num_clust in num_clust_vec:
        if num_clust==2:
            plt.plot(num_clust,np.var(np.array(size_clust_list[k])),'x',c='r',label='K-means')
        else:
            plt.plot(num_clust,np.var(np.array(size_clust_list[k])),'x',c='r')
            
    
        k=k+1
        
    
    clust_method='LDA'
    num_clust_vec=np.arange(2,8)
    size_clust_list,m=count_cluster_size(set_type,load_type,clust_method,num_clust_vec)
    
    k=0
    for num_clust in num_clust_vec:
        if num_clust==2:
            plt.plot(num_clust,np.var(np.array(size_clust_list[k])),'x',c='g',label='LDA')
        else:
            plt.plot(num_clust,np.var(np.array(size_clust_list[k])),'x',c='g')        
        k=k+1
        
        
        
    clust_method='GMM'
    num_clust_vec=np.arange(2,8)
    size_clust_list,m=count_cluster_size(set_type,load_type,clust_method,num_clust_vec)
    
    k=0
    for num_clust in num_clust_vec:
        if num_clust==2:
            plt.plot(num_clust,np.var(np.array(size_clust_list[k])),'x',c='k',label='GMM')
        else:
            plt.plot(num_clust,np.var(np.array(size_clust_list[k])),'x',c='k')
    
        k=k+1
        
    plt.xlabel('Number of clusters',fontsize=14)
    plt.ylabel('Variance of cluster size',fontsize=14)
    ax=plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)