# Lily Buechler, CS 229 (Fall 2018)

#%%
import pandas
import os
import numpy as np
from sklearn.cluster import KMeans
import variability_functions

#%% load the load profiles from timeseries data

load_type_list=['use','air1','refrigerator1','waterheater1','clotheswasher1','dishwasher1',
                'drye1','poolpump1','clotheswasher_dryg1','car1']
load_type_list_nice=['Total load','A/C','Refrigerator','Waterheater','Clotheswasher','Dishwasher',
                'Clothes dryer','Poolpump','Clotheswasher+dryer','EV']
os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features')

set_type='train'

for load_type in load_type_list:
    print (load_type)
    if load_type=='use':
        profiles_norm_use,profiles_use,house_id_norm_use,house_id_use=variability_functions.get_profiles(load_type,set_type)
    elif load_type=='air1':
        profiles_norm_air1,profiles_air1,house_id_norm_air1,house_id_air1=variability_functions.get_profiles(load_type,set_type)  
    elif load_type=='refrigerator1':
        profiles_norm_refrigerator1,profiles_refrigerator1,house_id_norm_refrigerator1,house_id_refrigerator1=variability_functions.get_profiles(load_type,set_type)
    elif load_type=='waterheater1':
        profiles_norm_waterheater1,profiles_waterheater1,house_id_norm_waterheater1,house_id_waterheater1=variability_functions.get_profiles(load_type,set_type)
    elif load_type=='clotheswasher1':
        profiles_norm_clotheswasher1,profiles_clotheswasher1,house_id_norm_clotheswasher1,house_id_clotheswasher1=variability_functions.get_profiles(load_type,set_type)
    elif load_type=='dishwasher1':
        profiles_norm_dishwasher1,profiles_dishwasher1,house_id_norm_dishwasher1,house_id_dishwasher1=variability_functions.get_profiles(load_type,set_type)
    elif load_type=='drye1':
        profiles_norm_drye1,profiles_drye1,house_id_norm_drye1,house_id_drye1=variability_functions.get_profiles(load_type,set_type)
    elif load_type=='poolpump1':
        profiles_norm_poolpump1,profiles_poolpump1,house_id_norm_poolpump1,house_id_poolpump1=variability_functions.get_profiles(load_type,set_type)
    elif load_type=='clotheswasher_dryg1':
        profiles_norm_clotheswasher_dryg1,profiles_clotheswasher_dryg1,house_id_norm_clotheswasher_dryg1,house_id_clotheswasher_dryg1=variability_functions.get_profiles(load_type,set_type)
    elif load_type=='car1':
        profiles_norm_car1,profiles_car1,house_id_norm_car1,house_id_car1=variability_functions.get_profiles(load_type,set_type)
        
#%% Kmeans clustering on daily load profiles to get load profile types

num_clusters=20
for load_type in load_type_list:    
    
    print(load_type)
    if load_type=='use':
            kmeans_use=KMeans(n_clusters=num_clusters,n_init=10).fit(profiles_norm_use)
            cluster_labels_use=kmeans_use.labels_
            cluster_centers_use=kmeans_use.cluster_centers_
    elif load_type=='air1':
            kmeans_air1=KMeans(n_clusters=num_clusters,n_init=10).fit(profiles_norm_air1)
            cluster_labels_air1=kmeans_air1.labels_
            cluster_centers_air1=kmeans_air1.cluster_centers_        
    elif load_type=='refrigerator1':
            kmeans_refrigerator1=KMeans(n_clusters=num_clusters,n_init=10).fit(profiles_norm_refrigerator1)
            cluster_labels_refrigerator1=kmeans_refrigerator1.labels_
            cluster_centers_refrigerator1=kmeans_refrigerator1.cluster_centers_        
    elif load_type=='waterheater1':
            kmeans_waterheater1=KMeans(n_clusters=num_clusters,n_init=10).fit(profiles_norm_waterheater1)
            cluster_labels_waterheater1=kmeans_waterheater1.labels_
            cluster_centers_waterheater1=kmeans_waterheater1.cluster_centers_          
    elif load_type=='clotheswasher1':
            kmeans_clotheswasher1=KMeans(n_clusters=num_clusters,n_init=10).fit(profiles_norm_clotheswasher1)
            cluster_labels_clotheswasher1=kmeans_clotheswasher1.labels_
            cluster_centers_clotheswasher1=kmeans_clotheswasher1.cluster_centers_              
    elif load_type=='dishwasher1':
            kmeans_dishwasher1=KMeans(n_clusters=num_clusters,n_init=10).fit(profiles_norm_dishwasher1)
            cluster_labels_dishwasher1=kmeans_dishwasher1.labels_
            cluster_centers_dishwasher1=kmeans_dishwasher1.cluster_centers_            
    elif load_type=='drye1':
            kmeans_drye1=KMeans(n_clusters=num_clusters,n_init=10).fit(profiles_norm_drye1)
            cluster_labels_drye1=kmeans_drye1.labels_
            cluster_centers_drye1=kmeans_drye1.cluster_centers_            
    elif load_type=='poolpump1':
            kmeans_poolpump1=KMeans(n_clusters=num_clusters,n_init=10).fit(profiles_norm_poolpump1)
            cluster_labels_poolpump1=kmeans_poolpump1.labels_
            cluster_centers_poolpump1=kmeans_poolpump1.cluster_centers_         
    elif load_type=='clotheswasher_dryg1':
            kmeans_clotheswasher_dryg1=KMeans(n_clusters=num_clusters,n_init=10).fit(profiles_norm_clotheswasher_dryg1)
            cluster_labels_clotheswasher_dryg1=kmeans_clotheswasher_dryg1.labels_
            cluster_centers_clotheswasher_dryg1=kmeans_clotheswasher_dryg1.cluster_centers_           
    elif load_type=='car1':
            kmeans_car1=KMeans(n_clusters=num_clusters,n_init=10).fit(profiles_norm_car1)
            cluster_labels_car1=kmeans_car1.labels_
            cluster_centers_car1=kmeans_car1.cluster_centers_ 
            
#%% calculate entropy

cluster_label_list=[cluster_labels_use,cluster_labels_air1,cluster_labels_refrigerator1,
                     cluster_labels_waterheater1,cluster_labels_clotheswasher1,
                     cluster_labels_dishwasher1,cluster_labels_drye1,
                     cluster_labels_poolpump1,cluster_labels_clotheswasher_dryg1,
                     cluster_labels_car1]
house_id_list=[house_id_norm_use,house_id_norm_air1,house_id_norm_refrigerator1,
               house_id_norm_waterheater1,house_id_norm_clotheswasher1,
               house_id_norm_dishwasher1,house_id_norm_drye1,house_id_norm_poolpump1,
               house_id_norm_clotheswasher_dryg1,house_id_norm_car1]
prob_entropy,entropy,house_id_entropy=variability_functions.calculate_entropy(cluster_label_list,house_id_list,num_clusters)


#%% save entropy probabilities to csv

os.chdir('C:\\Users\Lily Buechler\Documents\Lily\Stanford\CS_229\Project\Start_features')
cluster_method='kmeans'
for i in range(len(prob_entropy)):
    #prob_entropy_array_temp=np.zeros((len(prob_entropy[i]),num_clusters))
    for j in range(len(prob_entropy[i])):
        if j==0:
            prob_entropy_array=pandas.DataFrame({house_id_entropy[i][j]:prob_entropy[i][j]})
        else:
            prob_entropy_array[house_id_entropy[i][j]]=prob_entropy[i][j]
        #prob_entropy_array[j,:]=prob_entropy[i][j]
    #np.savetxt('load_prof_prob_'+str(load_type_list[i])+'_k'+str(num_clusters)+'.txt', prob_entropy_array, fmt='%f')
    prob_entropy_array.to_csv('load_prof_prob_'+str(load_type_list[i])+'_k'+str(num_clusters)+'_'+cluster_method+'_'+set_type+'.csv',index=False)


#%% plot entropy histogram by appliance
xlim_val=3.5
variability_functions.plot_entropy_hist(entropy,load_type_list,load_type_list_nice,xlim_val)

