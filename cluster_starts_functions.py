import numpy as np
import pandas
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy
import scipy.cluster


def load_start_distributions():
    os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features\start_distributions_final')

    load_type='drye1'
    set_type='train'
    data_train_drye1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    set_type='dev'
    data_dev_drye1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    
    load_type='dishwasher1'
    set_type='train'
    data_train_dishwasher1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    set_type='dev'
    data_dev_dishwasher1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    
    load_type='clotheswasher1'
    set_type='train'
    data_train_clotheswasher1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    set_type='dev'
    data_dev_clotheswasher1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    
    load_type='car1'
    set_type='train'
    data_train_car1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    set_type='dev'
    data_dev_car1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    
    load_type='waterheater1'
    set_type='train'
    data_train_waterheater1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    set_type='dev'
    data_dev_waterheater1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    
    load_type='clotheswasher_dryg1'
    set_type='train'
    data_train_clotheswasher_dryg1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)
    set_type='dev'
    data_dev_clotheswasher_dryg1 = pandas.read_csv('dist_'+set_type+'_'+str(load_type)+'_ls.csv',header=0)

    return data_train_drye1,data_dev_drye1,data_train_dishwasher1,data_dev_dishwasher1,data_train_clotheswasher1,data_dev_clotheswasher1,data_train_car1,data_dev_car1,data_train_waterheater1,data_dev_waterheater1,data_train_clotheswasher_dryg1,data_dev_clotheswasher_dryg1


def kmeans_cluster(data,num_clust,plot_fig):
    '''K-means clustering on the train set'''
    m=data.shape[1]

    kmeans=KMeans(n_clusters=num_clust,n_init=10).fit(np.transpose(np.array(data)))
    cluster_labels=kmeans.labels_
    cluster_centers=kmeans.cluster_centers_
    
    #% plot results
    
    if plot_fig==True:
        plt.figure(figsize=(6,num_clust*1.5))
        rank=np.array(range(m))
        for k in range(num_clust):
            plt.subplot(num_clust,1,k+1)
            ax=plt.gca()
            for i in range(len(rank[cluster_labels==k])):
                plt.plot(data[data.columns[rank[cluster_labels==k][i]]])
                #print(data.columns[rank[cluster_labels==k][i]])
            plt.ylabel('Probability',fontsize=14)
            #plt.xlim([0,23])
            if k==num_clust-1:
                plt.setp(plt.gca(), xticks=(0, 6
                         ,12,18),yticks=(0,0.2,0.4))
                plt.xlabel('Hour',fontsize=14)
            else:
                plt.setp(plt.gca(), xticks=(0, 6, 12,18), xticklabels=[],yticks=(0,0.2,0.4))        
            ax.tick_params(axis='both', which='major', labelsize=14)
            plt.xlim([0,23])
            plt.ylim([0,np.max(np.max(data))*1.1])
            plt.title('Cluster '+str(k+1)+', '+str(len(rank[cluster_labels==k]))+' samples',fontsize=14)
        plt.tight_layout()
    return cluster_labels,cluster_centers
    
    #plt.savefig('hier_dishwasher_k5.png')
    
def Kmeans_TrainOnTrain_TrainOnDev(load_type_list,label_list,save_clusters,num_clust_vec,data_train_drye1,data_dev_drye1,data_train_dishwasher1,data_dev_dishwasher1,data_train_clotheswasher1,data_dev_clotheswasher1,data_train_car1,data_dev_car1,data_train_waterheater1,data_dev_waterheater1,data_train_clotheswasher_dryg1,data_dev_clotheswasher_dryg1):
    '''Train Kmeans clustering on training set and on dev set'''
    
    load_iter=0
    for load_type in load_type_list:
        print(load_type)
        if load_type=='dishwasher1':
            data_train=data_train_dishwasher1
            data_dev=data_dev_dishwasher1
        elif load_type=='drye1':
            data_train=data_train_drye1
            data_dev=data_dev_drye1
        elif load_type=='clotheswasher1':
            data_train=data_train_clotheswasher1
            data_dev=data_dev_clotheswasher1
        elif load_type=='car1':
            data_train=data_train_car1
            data_dev=data_dev_car1
        elif load_type=='waterheater1':
            data_train=data_train_waterheater1
            data_dev=data_dev_waterheater1        
        elif load_type=='clotheswasher_dryg1':
            data_train=data_train_clotheswasher_dryg1
            data_dev=data_dev_clotheswasher_dryg1
            
            
        m=data_train.shape[1]
        n=data_train.shape[0]
        k=0
        for num_clust in num_clust_vec:
            print(k)
            plot_fig=False
            cluster_labels_train_temp,cluster_centers_train_temp=kmeans_cluster(data_train,num_clust,plot_fig)
            cluster_labels_dev_temp,cluster_centers_dev_temp=kmeans_cluster(data_dev,num_clust,plot_fig)
            
            if k==0:
                cluster_labels_train=cluster_labels_train_temp.reshape(1,-1)
                cluster_labels_dev=cluster_labels_dev_temp.reshape(1,-1)
            else:
                cluster_labels_train=np.concatenate((cluster_labels_train,cluster_labels_train_temp.reshape(1,-1)),axis=0)
                cluster_labels_dev=np.concatenate((cluster_labels_dev,cluster_labels_dev_temp.reshape(1,-1)),axis=0)
            
            k=k+1
            
            
        for i in range(cluster_labels_train.shape[1]):
            if i==0:
                cluster_labels_train_df=pandas.DataFrame({data_train.columns[i]:cluster_labels_train[:,i]})
                cluster_labels_dev_df=pandas.DataFrame({data_dev.columns[i]:cluster_labels_dev[:,i]})
            else:
                cluster_labels_train_df[data_train.columns[i]]=cluster_labels_train[:,i]
                cluster_labels_dev_df[data_dev.columns[i]]=cluster_labels_dev[:,i]
                
        if save_clusters==True:
            cluster_labels_train_df.to_csv('start_clusters_train_'+str(load_type)+'_kmeans.csv',index=False)
            cluster_labels_dev_df.to_csv('start_clusters_dev_'+str(load_type)+'_kmeans.csv',index=False)
    return cluster_labels_train, cluster_labels_dev
    
    
def kmeans_cluster_pred(data_train,data_dev,num_clust):
    '''Predict K-means clusters on the dev set'''
    m=data_train.shape[1]

    kmeans=KMeans(n_clusters=num_clust,n_init=10).fit(np.transpose(np.array(data_train)))
    cluster_labels_dev=kmeans.predict(np.transpose(data_dev))
    cluster_labels_train=kmeans.labels_
    cluster_centers=kmeans.cluster_centers_
    
    return cluster_labels_dev,cluster_labels_train,cluster_centers
    
def Kmeans_TrainOnTrain_TestOnDev(load_type_list,label_list,save_clusters,num_clust_vec,data_train_drye1,data_dev_drye1,data_train_dishwasher1,data_dev_dishwasher1,data_train_clotheswasher1,data_dev_clotheswasher1,data_train_car1,data_dev_car1,data_train_waterheater1,data_dev_waterheater1,data_train_clotheswasher_dryg1,data_dev_clotheswasher_dryg1):
    '''Perform kmeans clustering, training on the training set and predicting on the dev set'''
    load_iter=0
    for load_type in load_type_list:
        print(load_type)
        if load_type=='dishwasher1':
            data_train=data_train_dishwasher1
            data_dev=data_dev_dishwasher1
        elif load_type=='drye1':
            data_train=data_train_drye1
            data_dev=data_dev_drye1
        elif load_type=='clotheswasher1':
            data_train=data_train_clotheswasher1
            data_dev=data_dev_clotheswasher1
        elif load_type=='car1':
            data_train=data_train_car1
            data_dev=data_dev_car1
        elif load_type=='waterheater1':
            data_train=data_train_waterheater1
            data_dev=data_dev_waterheater1        
        elif load_type=='clotheswasher_dryg1':
            data_train=data_train_clotheswasher_dryg1
            data_dev=data_dev_clotheswasher_dryg1
            
            
        m=data_train.shape[1]
        n=data_train.shape[0]
        k=0
        for num_clust in num_clust_vec:
            print(k)
            plot_fig=False
            #cluster_labels_train_temp,cluster_centers_train_temp=cluster_starts_functions.kmeans_cluster(data_train,num_clust,plot_fig)
            #cluster_labels_dev_temp,cluster_centers_dev_temp=cluster_starts_functions.kmeans_cluster(data_dev,num_clust,plot_fig)
            
            cluster_labels_dev_temp,cluster_labels_train_temp,cluster_centers=kmeans_cluster_pred(data_train,data_dev,num_clust)
            
            if k==0:
                cluster_labels_train=cluster_labels_train_temp.reshape(1,-1)
                cluster_labels_dev=cluster_labels_dev_temp.reshape(1,-1)
            else:
                cluster_labels_train=np.concatenate((cluster_labels_train,cluster_labels_train_temp.reshape(1,-1)),axis=0)
                cluster_labels_dev=np.concatenate((cluster_labels_dev,cluster_labels_dev_temp.reshape(1,-1)),axis=0)
            
            k=k+1
            
            
        for i in range(cluster_labels_train.shape[1]):
            if i==0:
                cluster_labels_train_df=pandas.DataFrame({data_train.columns[i]:cluster_labels_train[:,i]})
                cluster_labels_dev_df=pandas.DataFrame({data_dev.columns[i]:cluster_labels_dev[:,i]})
            else:
                cluster_labels_train_df[data_train.columns[i]]=cluster_labels_train[:,i]
                cluster_labels_dev_df[data_dev.columns[i]]=cluster_labels_dev[:,i]
                
        if save_clusters==True:
            cluster_labels_train_df.to_csv('start_clusters_train_v2_'+str(load_type)+'_kmeans.csv',index=False)
            cluster_labels_dev_df.to_csv('start_clusters_dev_v2_'+str(load_type)+'_kmeans.csv',index=False)
            
    return cluster_labels_train_df,cluster_labels_dev_df



def KL_similarity(data):
    '''Get KL divergence to do hierachical clustering on train set'''
    m=data.shape[1]
    n=data.shape[0]
    d=np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            dist_A=data[data.columns[i]]
            dist_B=data[data.columns[j]]
            KL_temp=0
            for p in range(n):
                KL_temp=KL_temp+dist_A[p]*np.log(dist_A[p]/dist_B[p])+dist_B[p]*np.log(dist_B[p]/dist_A[p])
            d[i,j]=KL_temp
    return d

def KL_similarity_compare(data_train,data_test,cluster_labels):
    '''Get KL divergence to perform hierachical clustering on dev set'''
    m=data_train.shape[1]
    n=data_train.shape[0]
    d=np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            dist_A=data_test[data_test.columns[i]]
            dist_B=data_train[data_train.columns[j]]
            KL_temp=0
            for p in range(n):
                KL_temp=KL_temp+dist_A[p]*np.log(dist_A[p]/dist_B[p])+dist_B[p]*np.log(dist_B[p]/dist_A[p])
            d[i,j]=KL_temp
    num_clust=len(np.unique(cluster_labels))
    dist_clust=np.zeros((m,num_clust))
    for j in range(m):
        for i in range(num_clust):
            d_temp=d[j,:]
            dist_clust[j,i]=np.mean(d_temp[cluster_labels==i])
    cluster_labels_test=np.argmin(dist_clust,axis=1)
    
    return cluster_labels_test,dist_clust

def hier_clust(d,linkage_method,plot_data):
    '''Perform hierachical clustering'''
    dv=scipy.spatial.distance.squareform(d,checks=False)
    clust=scipy.cluster.hierarchy.linkage(dv,method=linkage_method)
    if plot_data==True:
        plt.figure(figsize=(15,5))
        dendro=scipy.cluster.hierarchy.dendrogram(clust,leaf_font_size=10)
        plt.xlabel('Event number',fontsize=14)
        plt.ylabel('Similarity',fontsize=14)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14) 
    return clust

def cut_tree_num_clust(num_clust,clust):
    '''Cut dendrogram for hierachical clustering for a certain number of clusters'''
    
    num_clust_temp=-100
    height_cut=10
    delta=0.05
    while num_clust_temp!=num_clust:
        height_cut=height_cut-delta
        cluster_labels=np.ndarray.flatten(scipy.cluster.hierarchy.cut_tree(clust,height=height_cut))
        num_clust_temp=len(np.unique(cluster_labels))
        if height_cut<0:
            height_cut=5
            delta=delta/2
    print(num_clust_temp)
    return cluster_labels

def Hier_TrainOnTrain_TrainOnDev(load_type_list,label_list,num_clust_vec,save_clusters,data_train_drye1,data_dev_drye1,data_train_dishwasher1,data_dev_dishwasher1,data_train_clotheswasher1,data_dev_clotheswasher1,data_train_car1,data_dev_car1,data_train_waterheater1,data_dev_waterheater1,data_train_clotheswasher_dryg1,data_dev_clotheswasher_dryg1):
    '''Perform hierachical clustering on training and dev set'''
    load_iter=0
    for load_type in load_type_list:
        print(load_type)
        if load_type=='dishwasher1':
            data_train=data_train_dishwasher1
            data_dev=data_dev_dishwasher1
        elif load_type=='drye1':
            data_train=data_train_drye1
            data_dev=data_dev_drye1
        elif load_type=='clotheswasher1':
            data_train=data_train_clotheswasher1
            data_dev=data_dev_clotheswasher1
        elif load_type=='car1':
            data_train=data_train_car1
            data_dev=data_dev_car1
        elif load_type=='waterheater1':
            data_train=data_train_waterheater1
            data_dev=data_dev_waterheater1        
        elif load_type=='clotheswasher_dryg1':
            data_train=data_train_clotheswasher_dryg1
            data_dev=data_dev_clotheswasher_dryg1
            
            
        m=data_train.shape[1]
        n=data_train.shape[0]
        
        d_dev=KL_similarity(data_dev)
        d_train=KL_similarity(data_train)
    
        # hierarchical clustering
        linkage_method='ward'
        plot_data=False
        clust_train=hier_clust(d_train,linkage_method,plot_data)
        clust_dev=hier_clust(d_dev,linkage_method,plot_data)
        
        k=0
        for num_clust in num_clust_vec:
            #print(k)
            plot_fig=False
            
            cluster_labels_train_temp=cut_tree_num_clust(num_clust,clust_train)
            cluster_labels_dev_temp=cut_tree_num_clust(num_clust,clust_dev)
    
            if k==0:
                cluster_labels_train=cluster_labels_train_temp.reshape(1,-1)
                cluster_labels_dev=cluster_labels_dev_temp.reshape(1,-1)
            else:
                cluster_labels_train=np.concatenate((cluster_labels_train,cluster_labels_train_temp.reshape(1,-1)),axis=0)
                cluster_labels_dev=np.concatenate((cluster_labels_dev,cluster_labels_dev_temp.reshape(1,-1)),axis=0)
            
            k=k+1
            
            
        for i in range(cluster_labels_train.shape[1]):
            if i==0:
                cluster_labels_train_df=pandas.DataFrame({data_train.columns[i]:cluster_labels_train[:,i]})
                cluster_labels_dev_df=pandas.DataFrame({data_dev.columns[i]:cluster_labels_dev[:,i]})
            else:
                cluster_labels_train_df[data_train.columns[i]]=cluster_labels_train[:,i]
                cluster_labels_dev_df[data_dev.columns[i]]=cluster_labels_dev[:,i]
                
        if save_clusters==True:
            cluster_labels_train_df.to_csv('start_clusters_train_'+str(load_type)+'_hier.csv',index=False)
            cluster_labels_dev_df.to_csv('start_clusters_dev_'+str(load_type)+'_hier.csv',index=False)
    return cluster_labels_train_df,cluster_labels_dev_df


def Hier_TrainOnTrain_TestOnDev(load_type_list,label_list,num_clust_vec,save_clusters,data_train_drye1,data_dev_drye1,data_train_dishwasher1,data_dev_dishwasher1,data_train_clotheswasher1,data_dev_clotheswasher1,data_train_car1,data_dev_car1,data_train_waterheater1,data_dev_waterheater1,data_train_clotheswasher_dryg1,data_dev_clotheswasher_dryg1):
    '''Perform hierachical clustering on training set, use to predict on dev set'''

    load_iter=0
    for load_type in load_type_list:
        print(load_type)
        if load_type=='dishwasher1':
            data_train=data_train_dishwasher1
            data_dev=data_dev_dishwasher1
        elif load_type=='drye1':
            data_train=data_train_drye1
            data_dev=data_dev_drye1
        elif load_type=='clotheswasher1':
            data_train=data_train_clotheswasher1
            data_dev=data_dev_clotheswasher1
        elif load_type=='car1':
            data_train=data_train_car1
            data_dev=data_dev_car1
        elif load_type=='waterheater1':
            data_train=data_train_waterheater1
            data_dev=data_dev_waterheater1        
        elif load_type=='clotheswasher_dryg1':
            data_train=data_train_clotheswasher_dryg1
            data_dev=data_dev_clotheswasher_dryg1
            
            
        m=data_train.shape[1]
        n=data_train.shape[0]
        
        d_train=KL_similarity(data_train)
    
        linkage_method='ward'
        plot_data=False
        clust_train=hier_clust(d_train,linkage_method,plot_data)
        
        k=0
        for num_clust in num_clust_vec:
            #print(k)
            plot_fig=False
            
            cluster_labels_train_temp=cut_tree_num_clust(num_clust,clust_train)
            #cluster_labels_dev_temp=cluster_starts_functions.cut_tree_num_clust(num_clust,clust_dev)
            cluster_labels_dev_temp,dist_clust=KL_similarity_compare(data_train,data_dev,cluster_labels_train_temp)
    
            if k==0:
                cluster_labels_train=cluster_labels_train_temp.reshape(1,-1)
                cluster_labels_dev=cluster_labels_dev_temp.reshape(1,-1)
            else:
                cluster_labels_train=np.concatenate((cluster_labels_train,cluster_labels_train_temp.reshape(1,-1)),axis=0)
                cluster_labels_dev=np.concatenate((cluster_labels_dev,cluster_labels_dev_temp.reshape(1,-1)),axis=0)
            
            k=k+1
            
            
        for i in range(cluster_labels_train.shape[1]):
            if i==0:
                cluster_labels_train_df=pandas.DataFrame({data_train.columns[i]:cluster_labels_train[:,i]})
                cluster_labels_dev_df=pandas.DataFrame({data_dev.columns[i]:cluster_labels_dev[:,i]})
            else:
                cluster_labels_train_df[data_train.columns[i]]=cluster_labels_train[:,i]
                cluster_labels_dev_df[data_dev.columns[i]]=cluster_labels_dev[:,i]
                
        if save_clusters==True:
            cluster_labels_train_df.to_csv('start_clusters_train_v2_'+str(load_type)+'_hier.csv',index=False)
            cluster_labels_dev_df.to_csv('start_clusters_dev_v2_'+str(load_type)+'_hier.csv',index=False)
    return cluster_labels_train_df,cluster_labels_dev_df