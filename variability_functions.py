# Lily Buechler, CS 229 (Fall 2018)

import pandas
import os
import numpy as np
from sklearn.cluster import KMeans
import datetime
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics


def get_profiles(load_type,set_type):
    os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features')

    if set_type=='train':
        power_data_train = pandas.read_csv(load_type+'_data_train.csv',header=0)
        # clean up timestamps
        for i in range(len(power_data_train.timestamp)):
            if power_data_train.timestamp[i]=='0':
                year_temp=2015
                temp_date=datetime.datetime(year_temp,power_data_train.month[i],power_data_train.day[i],power_data_train.hour[i],
                                            power_data_train.minute[i])
                timestamp_temp=datetime.datetime.strftime(temp_date,"%m-%d-%Y %H:%M:%S")
                power_data_train.timestamp[i]=timestamp_temp

        # remove columns with any NA elements
        k=0
        na_vec=power_data_train.isnull().any()
        first_na_flag=0
        for i in power_data_train.columns:
            if na_vec[k]==True:
                if first_na_flag==0:
                    power_data_train=power_data_train.drop(i,axis=1)  
                    first_na_flag=1
                else:
                    power_data_train=power_data_train.drop(i,axis=1)  
            k=k+1
            
        # reindex with date
        power_data_train=power_data_train.set_index(pandas.DatetimeIndex(power_data_train['timestamp']))

        #% resample to hourly interval
        power_data_train_resampled=power_data_train.resample('H').sum()/60
        
        # extract all daily profiles
        house_id=list()
        for i in range(5,len(power_data_train_resampled.columns)):
            if i==5:
                profiles=power_data_train_resampled[power_data_train_resampled.columns[i]].reshape(-1,24)
                for j in range(profiles.shape[0]):
                    house_id.append(power_data_train_resampled.columns[i])
            else:
                profiles_temp=power_data_train_resampled[power_data_train_resampled.columns[i]].reshape(-1,24)
                profiles=np.concatenate((profiles,profiles_temp),axis=0)
                for j in range(profiles_temp.shape[0]):
                    house_id.append(power_data_train_resampled.columns[i])
                    
        #normalize using total daily energy consumption
        flag_norm=0
        house_id_norm=list()
        for i in range(profiles.shape[0]):
            if np.sum(profiles[i,:])>0.2:
                profiles_norm_temp=profiles[i,:]/np.sum(profiles[i,:])
                profiles_norm_temp=profiles_norm_temp.reshape(1,-1)
                if flag_norm==0:
                    profiles_norm=profiles_norm_temp
                    flag_norm=1
                else:
                    profiles_norm=np.concatenate((profiles_norm,profiles_norm_temp),axis=0)
                house_id_norm.append(house_id[i])


    elif set_type=='dev':    
        power_data_dev = pandas.read_csv(load_type+'_data_dev.csv',header=0)

        if load_type=='use':
            power_data_dev['timestamp'][281723]='07-15-2014 15:23:00'
            power_data_dev['timestamp'][281724]='07-15-2014 15:24:00'
            
        # clean up timestamps
        for i in range(len(power_data_dev.timestamp)):
            if power_data_dev.timestamp[i]=='0':
                year_temp=2014
                temp_date=datetime.datetime(year_temp,power_data_dev.month[i],power_data_dev.day[i],power_data_dev.hour[i],
                                            power_data_dev.minute[i])
                timestamp_temp=datetime.datetime.strftime(temp_date,"%m-%d-%Y %H:%M:%S")
                power_data_dev.timestamp[i]=timestamp_temp
        
        # remove columns with any NA elements
        k=0
        na_vec=power_data_dev.isnull().any()
        first_na_flag=0
        for i in power_data_dev.columns:
            if na_vec[k]==True:
                if first_na_flag==0:
                    power_data_dev=power_data_dev.drop(i,axis=1)  
                    first_na_flag=1
                else:
                    power_data_dev=power_data_dev.drop(i,axis=1)  
            k=k+1
                        
        # reindex with date
        power_data_dev=power_data_dev.set_index(pandas.DatetimeIndex(power_data_dev['timestamp']))
        
        # resample to hourly interval
        power_data_dev_resampled=power_data_dev.resample('H').sum()/60
        
        # extract all daily profiles
        house_id=list()
        for i in range(5,len(power_data_dev_resampled.columns)):
            if i==5:
                profiles=power_data_dev_resampled[power_data_dev_resampled.columns[i]].reshape(-1,24)
                for j in range(profiles.shape[0]):
                    house_id.append(power_data_dev_resampled.columns[i])
            else:
                profiles_temp=power_data_dev_resampled[power_data_dev_resampled.columns[i]].reshape(-1,24)
                profiles=np.concatenate((profiles,profiles_temp),axis=0)
                for j in range(profiles_temp.shape[0]):
                    house_id.append(power_data_dev_resampled.columns[i])
        #normalize using total daily energy consumption
        flag_norm=0
        house_id_norm=list()
        for i in range(profiles.shape[0]):
            if np.sum(profiles[i,:])>0.2:
                profiles_norm_temp=profiles[i,:]/np.sum(profiles[i,:])
                profiles_norm_temp=profiles_norm_temp.reshape(1,-1)
                if flag_norm==0:
                    profiles_norm=profiles_norm_temp
                    flag_norm=1
                else:
                    profiles_norm=np.concatenate((profiles_norm,profiles_norm_temp),axis=0)
                house_id_norm.append(house_id[i])
    return profiles_norm,profiles,house_id_norm,house_id

def plot_cluster_centers(cluster_center_list,load_type_list):
    plt.figure(figsize=(15,8))
    for a in range(len(cluster_center_list)):
        plt.subplot(2,5,a+1)
        for c in range(len(cluster_center_list[a])):
            plt.plot(cluster_center_list[a][c,:])
            plt.title(load_type_list[a])
        if a==0 or a==5:
            plt.ylabel('Normalized power',fontsize=14)
        if a==6 or a==7 or a==8 or a==9 or a==10:
            plt.xlabel('Hour',fontsize=14)
        plt.xticks(np.arange(0,30,8))
        ax=plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14) 
    plt.tight_layout()
    
def calculate_entropy(cluster_label_list,house_id_list,num_clusters):
    prob_entropy=list()
    entropy=list()
    house_id_entropy=list()
    for a in range(len(cluster_label_list)):
        prob_entropy.append([])
        entropy.append([])
        house_id_entropy.append([])
        house_vec=np.array([int(m) for m in house_id_list[a]])
        cluster_labels_temp=cluster_label_list[a]
        
        k=0
        for h in np.unique(house_vec):
            house_id_entropy[a].append(h)
            cluster_labels_house=cluster_labels_temp[house_vec==h]
            prob_entropy[a].append([])
            entropy_temp=0
            for i in range(num_clusters):
                prob_temp=(len(cluster_labels_house[cluster_labels_house==i])+1)/(len(cluster_labels_house)+num_clusters)
                prob_entropy[a][k].append(prob_temp)
                entropy_temp=entropy_temp+prob_temp*np.log(prob_temp)
            entropy[a].append(-entropy_temp)
            k=k+1
    return prob_entropy, entropy, house_id_entropy

def plot_entropy_hist(entropy,load_type_list,load_type_list_nice,xlim_val):
    plt.figure(figsize=(12,5))
    for a in range(len(entropy)):
        plt.subplot(2,5,a+1)
        plt.plot(np.mean(entropy[a])*np.array([1,1]),np.array([0,100]),'--',c='r')
        n=plt.hist(entropy[a],range=[0,xlim_val],bins=40)
        plt.text(0.1,max(n[0])*1.05,'Mean='+str("{0:.2f}".format(np.mean(entropy[a]))),fontsize=10)
        ax=plt.gca()
        plt.ylim([0,max(n[0])*1.2])
        if a==0 or a==5:
            plt.ylabel('Frequency',fontsize=14)
        if a==6 or a==7 or a==8 or a==9 or a==5:
            plt.xlabel('Entropy',fontsize=14)
        
        ax.tick_params(axis='both', which='major', labelsize=14) 
        plt.title(load_type_list_nice[a])
        plt.xticks(np.arange(np.ceil(xlim_val)))
    plt.tight_layout()
    
def plot_entropy_bar(cluster_label_list,house_id_entropy,load_type_list,entropy):
    entropy_dict={}
    house_list_total=list()
    for a in range(len(cluster_label_list)):
        for h in range(len(house_id_entropy[a])):
            if house_id_entropy[a][h] in house_list_total:
                entropy_dict[house_id_entropy[a][h]][load_type_list[a]]=entropy[a][h]
            else:
                house_list_total.append(house_id_entropy[a][h])
                entropy_dict[house_id_entropy[a][h]]={load_type_list[a]:entropy[a][h]}
        
    
    
    
    plt.figure(figsize=(15,10))
    k=0
    
    for h in entropy_dict.keys():
        entropy_plot_temp=list()
        app_plot_temp=list()
        for a in load_type_list[0:]:
        #for a in ['drye1','air1']:
            if a in entropy_dict[h].keys():
                entropy_plot_temp.append(entropy_dict[h][a])
                app_plot_temp.append(a)
        plt.plot(np.ones((len(app_plot_temp),1))*k,entropy_plot_temp,'.-',markersize=10,linewidth=1)
        #plt.plot(k,np.mean(entropy_plot_temp),'.',color='k',markersize=20)
        plt.ylim([0,3.5])
        plt.xlabel('House',fontsize=14)
        plt.ylabel('Entropy',fontsize=14)
        ax=plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14) 
        k=k+1

def plot_perc_entropy_bar(cluster_label_list,house_id_entropy,load_type_list,entropy):
    entropy_dict={}
    house_list_total=list()
    for a in range(len(cluster_label_list)):
        for h in range(len(house_id_entropy[a])):
            if house_id_entropy[a][h] in house_list_total:
                entropy_dict[house_id_entropy[a][h]][load_type_list[a]]=entropy[a][h]
            else:
                house_list_total.append(house_id_entropy[a][h])
                entropy_dict[house_id_entropy[a][h]]={load_type_list[a]:entropy[a][h]}
        
    
    
    
    plt.figure(figsize=(15,10))
    k=0
    
    for h in entropy_dict.keys():
        entropy_plot_temp=list()
        app_plot_temp=list()
        for a in load_type_list[0:]:
        #for a in ['drye1','air1']:
            if a in entropy_dict[h].keys():
                entropy_plot_temp.append(entropy_dict[h][a])
                app_plot_temp.append(a)
        plt.plot(np.ones((len(app_plot_temp),1))*k,entropy_plot_temp,'.-',markersize=10,linewidth=2)
        #plt.plot(k,np.mean(entropy_plot_temp),'.',color='k',markersize=20)
        #plt.ylim([0,3.5])
        plt.xlabel('House',fontsize=14)
        plt.ylabel('Percentile entropy',fontsize=14)
        ax=plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14) 
        k=k+1

    
def compare_labels(cluster_labels_A,cluster_labels_B,num_clust,m):
    
    lab=np.arange(m)
    count_mat=np.zeros((num_clust,num_clust))
    for i in range(num_clust):
        for j in range(num_clust):
            count_mat[i,j]=len(set(lab[cluster_labels_A==i]).intersection(set(lab[cluster_labels_B==j])))
    
    A_node_list=list()
    B_node_list=list()
    for i in range(num_clust):
        A_node_list.append('A'+str(i))
        B_node_list.append('B'+str(i))
        
    
    B=nx.Graph()
    B.add_nodes_from(A_node_list,bipartite=0)
    B.add_nodes_from(B_node_list,bipartite=1)
    for i in range(num_clust):
        for j in range(num_clust):
            B.add_edges_from([(A_node_list[i],B_node_list[j])],weight=count_mat[i,j])
    
    matching=nx.max_weight_matching(B,maxcardinality=True)
    
    score=0
    for i in range(num_clust):
        score=score+count_mat[i,B_node_list.index(matching[A_node_list[i]])]
    print(matching)
    score=score/m
    return score