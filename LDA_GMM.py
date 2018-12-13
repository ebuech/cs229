import numpy as np
import sklearn.decomposition as skld
import sklearn.mixture as skmx
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import matplotlib.cm as cm
import string
from math import floor

def rounding(original_array,multiplier):
    # This function just rounds numbers such that they

    # round all the numbers down
    output_array = np.floor(original_array)
    diff = multiplier-np.sum(output_array)
    temp_decimal = original_array-output_array
    np.ndarray.sort(temp_decimal)
    for i in range(1,int(diff)+1):
        output_array[-i] = output_array[-i]+1

    return output_array.astype(int)

def import_data(filename):
    # This function imports the dataset specified in filename and then converts it into a frequency distribution that has n = mult counts

    data = np.loadtxt(filename,delimiter=',')
    mult = 100000 # multiplier for to generate frequency distributions
    # remove the headers because we don't need them
    data = np.array(data[1:,:])
    # transpose our data and multiply by factor of 100
    data = mult * data.T
    m, n = data.shape

    for i in range(0,m):
        data[i,:] = rounding(data[i,:],mult)

    return data.astype(int)

def import_data_no_rounding(filename):
    # This function imports the dataset specified in filename WITHOUT converting it into a frequency distribution

    data = np.loadtxt(filename,delimiter=',')

    # remove the headers because we don't need them
    data = np.array(data[1:,:])
    # transpose our data
    data = data.T
    m, n = data.shape

    # data = np.ndarray.astype(1000*data.T,int)

    return data

def compare_distributions(rawData,clusterAssignments,numComponents,nRows,applianceName):
    # This function is a utility function that plots data
    # It saves a figure which shows all probability distributions of the homes grouped into their clusters
    # The PNG plot is saved to the filepath that is stipulated at the end of the function

    #create a new figure
    n = int(np.ceil(numComponents/nRows))
    m = nRows
    fig, axes = plt.subplots(n,m,squeeze=True,sharex=True,sharey=True)
    axes = np.reshape(axes,[1,n*m])

    #create a x vector of 24 hours
    time = np.arange(0,24,1)

    # color rendering: assumes 24 discrete pts per distribution
    r, c = rawData.shape
    N = 24
    num_types = int(c/N) # how many distributions we vectorized = number of colors we want to have
    cmap = plt.cm.get_cmap('nipy_spectral')

    #color_matrix = cmap(np.arange(1,num_types+1,1)/num_types)
    color_matrix = cmap(np.arange(1, r + 1, 1) / r)

    for i in range(numComponents):
        idx = np.where(np.array(clusterAssignments)==i)[0] # this needs to be indexed to 1 and not zero for GMM...
        for j in idx:
            for n in range(0,num_types):
                #axes[0,i].plot(time+24*n, rawData[j,24*n:24*n+24].T,c=color_matrix[n,:])
                #axes[0, i].plot(time + 24 * n, rawData[j, 24 * n:24 * n + 24].T)
                axes[0, i].plot(time, rawData[j,:].T,c=color_matrix[j,:])
                axes[0, i].set_ylim(bottom=0,top=0.25)
                axes[0, i].set_xlabel('Hour')
                axes[0, i].set_ylabel('Probability')

    #fig.suptitle(applianceName)
    #plt.savefig('./output/' + applianceName + '_optimal.png')
    plt.savefig('./output/LDA_'+applianceName+str(numComponents)+'.png')

def runLDA(train,test,dev,numComponents,applianceName):

    # Function that run the Latent Dirichlet Allocation
    # It takes in a train,test and dev dataset that comprises frequency distributions
    # numComponents specifies the number of clusters or topics for the model

    # It returns the perplexity score of the dev set, and the clusters assignements on the train,dev and test set

    model = skld.LatentDirichletAllocation(n_components=numComponents,verbose=0)
    model.fit(train)

    predictions = model.transform(test)
    test_classification = np.argmax(predictions,1)
    test_classification = test_classification.astype(int)

    predictions2 = model.transform(dev)
    dev_classification = np.argmax(predictions2,1)
    dev_classification = dev_classification.astype(int)

    predictions3 = model.transform(train)
    train_classification = np.argmax(predictions3,1)
    train_classification = train_classification.astype(int)

    if numComponents>10:
        nRows = 3
    else:
        nRows = 2

    perplex = model.perplexity(dev)

    #compare_distributions(test,dev_classification,numComponents,nRows,applianceName)
    #compare_distributions(dev, dev_classification, numComponents, nRows, applianceName)

    return perplex,train_classification,dev_classification,test_classification

def runGMM(train,test,dev,numComponents,applianceName):
    # This function trains a GMM model and consequently assigns clusters for the train, test and dev set.
    # It takes in a train,test and dev dataset that comprises probability distributions
    # numComponents specifies the number of clusters or topics for the model

    # It returns the clusters assignements on the train,dev and test set

    model = skmx.GaussianMixture(n_components=numComponents,covariance_type='tied')
    model.fit(train)

    predictions = model.predict(train)
    #compare_distributions(train, predictions, numComponents,2, applianceName)

    predictions2 = model.predict(test)
    #compare_distributions(test, predictions2[0, :], numComponents, 4, applianceName)

    predictions3 = model.predict(dev)

    return predictions, predictions3, predictions2

def calc_KLD(data1,data2):
    # function calculates and returns the Symmetrized KL Divergence between 2 probability distributions defined as data1 and 2

    KLD = np.sum(data1 * np.log(data1 / data2) + data2 * np.log(data2 / data1), 0)
    return KLD

def calc_intracluster_KLD(test_data, assignments):
    # This function calculates the intracluster symmetrized KL Divergence of a set of n assignment lists where n is the.
    # test_data contains the dataset as probability distributions
    # assignments contains a list of list of cluster assignments.
    # The function returns a list of intracluster symmetrized KL Divergence scores

    num_conditions = np.size(assignments,0)  # get the number of clusters we tried. for instance 2,3,4 clusters would be size 3
    KLD = np.zeros(num_conditions)
    for i in range(num_conditions):
        clusters = np.unique(assignments[i])
        for j in clusters:
            idx = np.where(np.array(assignments[i]) == j)  # get the index for the cluster
            arr = test_data[idx]  # slice out all the data in that cluster
            mean = np.mean(arr,0)
            num_dist = np.size(arr, 0)
            for k in range(num_dist):
                        KLD[i] += np.sum(arr[k, :] * np.log(arr[k, :] / mean) + mean * np.log(mean/arr[k, :]), 0)
            '''if num_dist>1:
                KLD[i] = KLD[i]/(num_dist*(num_dist-1))'''  # take average
        # print('Average KLD for ' + str(i+2) + ' clusters  = ' + str(KLD[i]))
    return KLD

def calc_silhouette_scores(data,labels):
    # calculates the silhouette scores for the data based on a given assignment using symmetrized KL Divergence as a distance metric
    # data is the dataset of probability distributions
    # labels is a list of list of cluster assignments

    # returns a list of silhouette scores

    m,n =  data.shape
    KLD_matrix = np.zeros((m,m))
    sil_score_list = []
    L = np.size(labels,0)
    for k in range(L):
        for i in range(m):
            for j in range(m):
                KLD = calc_KLD(data[i, :],data[j, :])
                KLD_matrix[i,j] = KLD

        sil_score_list.append(skmet.silhouette_score(KLD_matrix,labels[k],metric='precomputed'))

        #print('Average sil for ' + str(k + 2) + ' clusters  = ' + str(sil_score_list[k]))

    return np.array(sil_score_list)

def calc_completeness_scores(train_data,eval_data):
    # calculates the completeness score between the training cluster assignments and the evaluation assignments (using the trained clusters)
    # in this case we regard the training cluster assignments as the "TRUE" label and the evaluation assignments as the "PREDICTED" assignments

    # train_data and eval_data are both lists of cluster assignment lists

    # returns a list of completeness scores

    num_clusters = len(train_data)
    c_score_array = np.zeros(num_clusters)
    for i in range(num_clusters):
        c_score_array[i] = skmet.completeness_score(train_data[i], eval_data[i])

    return c_score_array

def calc_adjRand_scores(train_data,eval_data):
    # Calculates the adjusted Rand score

    # train_data and eval_data are once again lists of cluster assignment lists

    # returns a list of adjusted Rand scores.

    num_clusters = len(train_data)
    r_score_array = np.zeros(num_clusters)
    for i in range(num_clusters):
        r_score_array[i] = skmet.adjusted_rand_score(train_data[i], eval_data[i])

    return r_score_array

def plot_silhouette(X,cluster_labels,appliance_name,algo_name):
    # This function calculates the sihouette score for each data point and plots all of the them as a distribution
    # It also calculates and displays the mean silhouette score to the dataset
    # It saves every silhouette plot

    # X is the dataset of probability distributions
    # cluster_labels is the list of list of cluster assignments
    # appliance_name is literally just that
    # algo_name is likewise literally just that

    max_clusters = len(cluster_labels)+1
    range_n_clusters = np.arange(2,max_clusters+1)
    m, n = X.shape
    KLD_matrix = np.zeros((m,m))

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters

        for i in range(m):
            for j in range(m):
                KLD = calc_KLD(X[i, :],X[j, :])
                KLD_matrix[i,j] = KLD

        silhouette_avg = skmet.silhouette_score(KLD_matrix, cluster_labels[n_clusters-2],metric='precomputed')
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = skmet.silhouette_samples(KLD_matrix, cluster_labels[n_clusters-2],metric='precomputed')

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them

            cluster_label_vect = np.array(cluster_labels[n_clusters - 2])

            ith_cluster_silhouette_values = \
                sample_silhouette_values[ cluster_label_vect == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for "+algo_name+" clustering on " + appliance_name + " data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.savefig('./output/sil_plots/' + algo_name + '/sil_plot' + algo_name + '_' + appliance_name + '_' + str(n_clusters) + '.png')

    #plt.show()

def plot_metrics(metric,nCmin,nCmax,appliance_name,metric_name,algo_name,eval_name):
    # plots a metric specified in metric over a number of clusters specified by nCmin,nCmax
    # save the figure as a PNG

    plt.figure()
    legend_list = []
    for count in range(len(appliance_name)):
        plt.plot(np.arange(nCmin, nCmax, 1), metric[count])
        legend_string = metric_name + " for " + appliance_name[count]
        legend_list.append(legend_string)
        plt.xlabel("Number of Components")
        plt.ylabel(metric_name)

        plt.legend(legend_list)

    savefile = './output/' + algo_name + "_" + metric_name + "_" + eval_name + '.png'
    plt.savefig(savefile)

def main(algo):
    # algo is a string that either equals "LDA" or "GMM"

    file_train_prefix = './data_3/dist_train_'
    file_test_prefix = './data_3/dist_test_'
    file_dev_prefix = './data_3/dist_dev_'
    file_suffix = '1_ls.csv'

    #appliance = ['clotheswasher','dishwasher','clotheswasher_dryg','drye','waterheater','car']
    appliance = ['clotheswasher', 'dishwasher', 'drye', 'waterheater', 'car']

    nCmax = 14  # sets the number of clusters (+1) we want
    nCmax = nCmax + 1
    nCmin = 2  # sets the minimum number of clusters we want
    num_runs = 10  # sets the number of runs we want

    # Sets the metric we want to compute the evaluation scores on
    eval_cond = 'dev'

    # initialize some lists to store our variables
    sil_best_assign = [[] for i in range(len(appliance))]
    sil_best = [[] for i in range(len(appliance))]
    sil_means = [[] for i in range(len(appliance))]

    KLD_best = [[] for i in range(len(appliance))]
    KLD_best_assign = [[] for i in range(len(appliance))]
    KLD_means = [[] for i in range(len(appliance))]

    C_best = [[] for i in range(len(appliance))]
    C_best_assign = [[] for i in range(len(appliance))]
    C_means = [[] for i in range(len(appliance))]

    AR_best = [[] for i in range(len(appliance))]
    AR_best_assign = [[] for i in range(len(appliance))]
    AR_means = [[] for i in range(len(appliance))]

    perp_best_assign = [[] for i in range(len(appliance))]
    perp_best = [[] for i in range(len(appliance))]
    perp_means = [[] for i in range(len(appliance))]

    train_list = [[] for i in range(len(appliance))]
    dev_list = [[] for i in range(len(appliance))]
    test_list = [[] for i in range(len(appliance))]
    perplex_list = [[] for i in range(len(appliance))]

    count = 0 # iterator for appliances

    '''BEGIN LOOPS'''
    for s in appliance:

        # Generate appliance specific filepaths
        train_path = file_train_prefix + s + file_suffix
        test_path = file_test_prefix + s + file_suffix
        dev_path = file_dev_prefix+ s + file_suffix

        # This data set contains frequency distributions developed from probability distributions
        train = import_data(train_path)
        test = import_data(test_path)
        dev = import_data(dev_path)

        # This data set contains probability distributions
        train_2 = import_data_no_rounding(train_path)
        dev_2 = import_data_no_rounding(dev_path)
        test_2 = import_data_no_rounding(test_path)

        for run_num in range(num_runs):

            train_list[count] = []
            dev_list[count] = []
            test_list[count] = []
            perplex_list[count] = []

            if algo == 'LDA':
                print("Starting LDA run #" + str(run_num) + " on..." + s)
            if algo =='GMM':
                print("Starting GMM run #" + str(run_num) + " on..." + s)

            for nComponents in range(nCmin,nCmax):
                print('training using n = ' + str(nComponents))

                if algo == 'LDA':
                    perplex,train_pred,dev_pred,test_pred = runLDA(train,test,dev,nComponents,s+'k'+str(nComponents))
                if algo == 'GMM':
                    train_pred,dev_pred,test_pred = runGMM(train_2, test_2,dev_2, nComponents, s)
                    perplex = 0

                train_list[count].append(train_pred)
                dev_list[count].append(dev_pred)
                test_list[count].append(test_pred)
                perplex_list[count].append(perplex)

            if eval_cond == "train":
                eval_list = train_list[count]
                eval_data = train_2
            elif eval_cond == "dev":
                eval_list = dev_list[count]
                eval_data = dev_2
            else:
                eval_list = test_list[count]
                eval_data = test_2

            sil_score = calc_silhouette_scores(eval_data, eval_list)
            KLD_score = calc_intracluster_KLD(eval_data, eval_list)
            C_score = calc_completeness_scores(train_list[count],eval_list)
            AR_score = calc_adjRand_scores(train_list[count],eval_list)

            if run_num == 0:
                sil_means[count] = sil_score
                sil_best[count] = sil_score
                sil_best_assign[count] = eval_list
                C_means[count] = np.array(C_score)
                C_best[count] = C_score
                C_best_assign[count] = eval_list
                KLD_means[count] = KLD_score
                KLD_best[count] = KLD_score
                KLD_best_assign[count] = eval_list
                AR_means[count] = AR_score
                AR_best[count] = AR_score
                AR_best_assign[count] = eval_list
                perp_means[count] = np.array(perplex_list[count])
                perp_best[count] = perplex_list[count]
                perp_best_assign[count] = eval_list
            else:
                sil_means[count] = sil_means[count] + sil_score
                perp_means[count] = perp_means[count] + np.array(perplex_list[count])
                C_means[count] = C_means[count] + np.array(C_score)
                AR_means[count] = AR_means[count] +np.array(AR_score)
                KLD_means[count] = KLD_means[count] + KLD_score
                # This next part just stores our best assignments according to criteria specific to that metric.
                for i in range(len(sil_best[count])):
                    if sil_best[count][i] < sil_score[i]:
                        sil_best[count][i] = sil_score[i]
                        sil_best_assign[count][i] = eval_list[i]
                    if C_best[count][i] < C_score[i]:
                        C_best[count][i] = C_score[i]
                        C_best_assign[count][i] = eval_list[i]
                    if AR_best[count][i] < AR_score[i]:
                        AR_best[count][i] = AR_score[i]
                        AR_best_assign[count][i] = eval_list[i]
                    if KLD_best[count][i] > KLD_score[i]:
                        KLD_best[count][i] = KLD_score[i]
                        KLD_best_assign[count][i] = eval_list[i]
                    if perp_best[count][i] > perplex_list[count][i]:
                        perp_best[count][i] = perplex_list[count][i]
                        perp_best_assign[count][i] = eval_list[i]

        #sil_means[count] = sil_means[count] / num_runs
        KLD_means[count] = KLD_means[count]/ num_runs
        perp_means[count] = perp_means[count]/num_runs
        C_means[count] = C_means[count]/num_runs
        AR_means[count] = AR_means[count]/num_runs

        #for k in range(len(sil_means[count])):
            #print("Averaged Silhouette Coeff for " + str(k + 2) + " clusters = " + str(sil_means[count][k]))
        for k in range(len(KLD_means[count])):
            print("Averaged intracluster sum KLD for " + str(k + 2) + " clusters = " + str(KLD_means[count][k]))
        for k in range(len(perp_means[count])):
            print("Averaged Perplexity for " + str(k + 2) + " clusters = " + str(perp_means[count][k]))
        for k in range(len(perp_means[count])):
            print("Averaged Completeness for " + str(k + 2) + " clusters = " + str(C_means[count][k]))
        for k in range(len(AR_means[count])):
            print("Averaged Adj. Rand for " + str(k + 2) + " clusters = " + str(AR_means[count][k]))

        if eval_cond == 'train':
            plot_silhouette(train_2,sil_best_assign[count],s,algo)

        count += 1

    ''' Save OUR best assignments for the condition we are evaluating '''
    if algo =='LDA':
        for count in range(len(appliance)):
            np.savetxt('./output/LDA_KLD_best_' + appliance[count] + '_' + eval_cond + '.csv', np.array(train_list[count]), delimiter=',')

    if algo == "GMM":
        for count in range(len(appliance)):
            np.savetxt('./output/GMM_KLD_best' + appliance[count] + '_' + eval_cond + '.csv', np.vstack(train_list[count]), delimiter=',')

    if algo == "LDA":
        plot_metrics(perp_means, nCmin, nCmax, appliance, "Perplex", algo,eval_cond)

    count = 0

    # Save all our metrics
    for s in appliance:
        metric = np.vstack((KLD_means[count],C_means[count],AR_means[count]))
        np.savetxt('./output/metric_scores/' + s + '_' + algo + "_" + eval_cond + '.csv',
                   metric, delimiter=',')
        count += 1

    #plot_metrics(sil_means, nCmin, nCmax, appliance, "Sil", algo, eval_cond)
    plot_metrics(KLD_means, nCmin, nCmax, appliance, "KLD", algo, eval_cond)
    plot_metrics(C_means,nCmin,nCmax,appliance, "C_score", algo, eval_cond)
    plot_metrics(AR_means,nCmin,nCmax,appliance,"AR_Score",algo,eval_cond)
    #plt.show()

if __name__ == "__main__":
    main('LDA')