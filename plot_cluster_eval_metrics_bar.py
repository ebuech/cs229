# Script to evaluate and visualize performance metrics from CSV files

import numpy as np
import matplotlib.pyplot as plt

#app_name = ['clotheswasher', 'dishwasher', 'drye', 'waterheater', 'car','clotheswasher_dryg']
app_name = ['clotheswasher', 'dishwasher', 'drye', 'waterheater', 'car','clotheswasher_dryg']
eval_set = ['train','dev']
algo_name = ['GMM', 'LDA','kmeans','hier']
selected_cluster_vector = [5,5,5,3,5,3] # this defines the selected cluster number we have chosen for that particular appliance. It must be in the same order as app_name.

# For titles and legends...
algo_nice = ['GMM', 'LDA', 'k-means', 'Hierarchical']
app_nice = ['Clotheswasher', 'Dishwasher', 'Dryer', 'Waterheater', 'Car', 'WasherDryer']

# Initialize our constants and lists
num_appliance = len(app_name)
num_eval_conds = len(eval_set)
num_algos = len(algo_name)
KLD_means = [ [ [ [] for k in range(num_algos)] for j in range(num_eval_conds)] for i in range(num_appliance)]
num_clusters = [ [ [ [] for k in range(num_algos)] for j in range(num_eval_conds)] for i in range(num_appliance)]
C_means = [ [ [ [] for k in range(num_algos)] for j in range(num_eval_conds)] for i in range(num_appliance)]
algo_selected_C_scores = [[] for k in range(num_algos)]

'''BEGINNING OF LOOPS'''

# KLD PLOTS
for j in range(num_eval_conds):
    fig,axs = plt.subplots(1, 4, sharey='row')
    axs.flatten()
    fig.set_size_inches(12,3)
    for k in range(num_algos):
        for i in range(num_appliance):
            file_string = './output/metric_scores/' + app_name[i] + '_' + algo_name[k] + '_' + eval_set[j] + '.csv'
            data = np.loadtxt(file_string, delimiter=',')

            KLD_means[i][j][k] = data[0,:]
            C_means[i][j][k] = data[1,:]
            num_clusters[i][j][k] = np.shape(data[0,:])[0]

            cluster_vector = np.arange(2,num_clusters[i][j][k]+2,1)
            axs[k].plot(cluster_vector,KLD_means[i][j][k])
            axs[k].set_title(algo_name[k] + ', ' + eval_set[j] + ' set')

    fig.legend(app_name,loc='lower center',ncol= num_appliance)
    plt.suptitle('KLD Scores')
    #save_string = './output/metric_scores/plots/KLD_' + algo_name[k] + '_' + eval_set[j] + '.png'
    save_string = './output/metric_scores/plots/KLD_' + eval_set[j] + '.eps'
    plt.savefig(save_string)

# COMPLETENESS SCORE BAR GRAPHS

bar_width = 0.1
opacity = 1
#fig, axs = plt.subplots(1, num_appliance, sharey='row')

for k in range(num_algos):
    for i in range(num_appliance):
        selected_cluster = selected_cluster_vector[i]
        algo_selected_C_scores[k].append(C_means[i][1][k][selected_cluster-2])

figure2 = plt.figure()
figure2.set_size_inches(12, 2)
index = np.arange(num_appliance)
for k in range(num_algos):
    j = 1
    plt.bar(index, np.array(algo_selected_C_scores[k]),bar_width,alpha=opacity,label=algo_name[k])
    index = index+bar_width

index = np.arange(num_appliance)
plt.xticks(index + bar_width, app_nice)
plt.suptitle('Completeness Scores, Train vs Dev')
plt.legend(algo_nice,ncol= num_appliance)
save_string = './output/metric_scores/plots/C_' + algo_name[k] + '_' + app_name[i] + '.png'
plt.savefig(save_string)


plt.show()