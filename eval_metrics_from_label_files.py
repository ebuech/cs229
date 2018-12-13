import numpy as np
import LDA_GMM as util

# script to process cluster labels from CSV files.

def main():

    eval_cond = ['train','dev']
    num_cond = len(eval_cond)
    app_name = ['clotheswasher_dryg','clotheswasher', 'dishwasher', 'drye', 'waterheater', 'car']

    num_app = len(app_name)
    algo = ['hier','kmeans']
    num_algo = len(algo)

    file_train_prefix = './data_3/dist_train_'
    file_test_prefix = './data_3/dist_test_'
    file_dev_prefix = './data_3/dist_dev_'
    file_suffix = '1_ls.csv'


    for i in range(num_app):

        train_path = file_train_prefix + app_name[i] + file_suffix
        test_path = file_test_prefix + app_name[i] + file_suffix
        dev_path = file_dev_prefix+ app_name[i] + file_suffix

        train_2 = util.import_data_no_rounding(train_path)
        dev_2 = util.import_data_no_rounding(dev_path)

        for j in range(num_cond):
            for k in range(num_algo):

                if eval_cond[j] == 'train':
                    data = train_2
                else:
                    data = dev_2

                print('processing ' + app_name[i] + ', ' + eval_cond[j] + ', ' + algo[k])
                label_train = np.loadtxt('./other_algo_data/start_clusters_' + eval_cond[0] + '_v2_' + app_name[i] + '1_' + algo[k] + '.csv',int,skiprows=1,delimiter=',')
                labels = np.loadtxt('./other_algo_data/start_clusters_' + eval_cond[j] + '_v2_' + app_name[i] + '1_' + algo[k] + '.csv',int,skiprows=1,delimiter=',')

                label_train = np.ndarray.tolist(label_train)
                labels = np.ndarray.tolist(labels)

                KLD = util.calc_intracluster_KLD(data,labels)
                AR = util.calc_adjRand_scores(label_train,labels)
                C = util.calc_completeness_scores(label_train,labels)
                '''
                if eval_cond[j] == 'train':
                    util.plot_silhouette(train_2,label_train,app_name[i],algo[k])
                '''
                nCmin = 2
                nCmax = np.shape(labels)[0]+2

                metric = np.vstack((KLD, C, AR))
                np.savetxt('./output/metric_scores/' + app_name[i] + '_' + algo[k] + "_" + eval_cond[j] + '.csv',
                           metric, delimiter=',')

if __name__ == "__main__":
    main()