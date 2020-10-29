import numpy as np
import logging
import torch
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import soe
from preprocessing_utils.data_select_utils import select_dataset
from preprocessing_utils.TripletData import TripletBatchesDataset
from logging_utils import logging_util
from config_utils.config_eval import load_config
from preprocessing_utils.TripletData import procrustes_disparity
from preprocessing_utils.TripletData import knn_classification_error


def parse_args():
    """
    To run this file use "CUDA_VISIBLE_DEVICES=3 python train_soe.py -config configs/triplet_loss/triplet_loss_baseline.json". See
    the config file in
    the path for an example of how to construct config files.
    """
    parser = argparse.ArgumentParser(description='Run SOE Experiments')
    parser.add_argument('-config', '--config_path', type=str, default='configs/triplet_loss/triplet_loss_baseline.json', required=True,
                        help='Input the Config File Path')
    args = parser.parse_args()
    return args


def main(args):

    config = load_config(args.config_path)
    dataset_name = config['dataset_selected']
    batch_size = config['sz_batch']
    learning_rate = config['optimizer_params']['learning_rate']
    iterations = config['nb_epochs']
    input_dim = config['input_dimension']
    dimensions = config['output_dimension']
    subset_size = config['number_of_points']
    number_of_test_triplets = config['n_test_triplets']
    triplet_multiplier = config['triplets_multiplier']
    log_dir = config['log']['path']
    hyper_search = config['hyper_search']['activation']
    optimizer = config['optimizer']

    if hyper_search:
        run_hyper_search(config=config)
    else:
        vec_data, labels = select_dataset(dataset_name, n_samples=subset_size, input_dim=input_dim)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        n = vec_data.shape[0]
        logn = int(np.log2(n))
        triplet_num = triplet_multiplier * logn * n * dimensions

        bs = min(batch_size, triplet_num)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        experiment_name = 'triplet_loss_' + \
                          'data_' + dataset_name + \
                          '_input_dim_' + str(input_dim) + \
                          '_output_dim_' + str(dimensions) + \
                          '_dimensions_' + str(dimensions) + \
                          '_triplet_num_' + str(triplet_multiplier) + \
                          '_n_pts_' + str(n) + \
                          '_lr_' + str(learning_rate) + \
                          '_optimizer_' + str(optimizer) + \
                          '_bs_' + str(batch_size)

        # create a logging file for extensive logging
        logging_path = os.path.join(log_dir, experiment_name + '.log')
        logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)

        logger.info('Name of Experiments: ' + experiment_name)
        logger.info('Logging Path:' + logging_path)
        logger.info('Dataset Name: ' + dataset_name)
        logger.info('Epochs: ' + str(iterations))
        logger.info('Learning Rate: ' + str(learning_rate))
        logger.info('Number of Points: ' + str(n))
        logger.info('Input Dimension: ' + str(input_dim))
        logger.info('Output Dimension: ' + str(dimensions))
        logger.info('Number of Test Triplets: ' + str(number_of_test_triplets))
        logger.info('Triplet Multiplier: ' + str(triplet_multiplier))
        logger.info('Batch Size: ' + str(batch_size))

        train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, bs, device)

        logger.info('Computing SOE...')
        begin = time.time()

        x, loss_array, remain_loss = soe.triplet_loss_adam(triplets=train_triplets_dataset.trips_data_indices, n=n,
                                                           dim=dimensions, iterations=iterations, bs=bs,
                                                           lr=learning_rate, device=device, logger=logger)

        elapsed_time = time.time() - begin

        logger.info('Evaluating the computed embeddings...')
        # compute triplet error for train and test data
        train_error = train_triplets_dataset.triplet_error(x)
        test_triplets_dataset = TripletBatchesDataset(vec_data, labels, 1000, 1000, device)
        test_error = test_triplets_dataset.triplet_error(x)
        procrustes_error = procrustes_disparity(vec_data, x)
        knn_error_ord_emb, knn_error_true_emb = knn_classification_error(x, vec_data, labels)

        # sample points for tsne visualization
        subsample = np.random.permutation(n)[0:500]
        x = x[subsample, :]
        sub_labels = labels[subsample]

        x_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x)
        fig, ax = plt.subplots(1, 1)

        ax.scatter(x_embedded[:, 0], x_embedded[:, 1], s=3, c=sub_labels)
        fig.savefig(os.path.join(log_dir, experiment_name + '.png'))

        logger.info('Name of Experiments: ' + experiment_name)
        logger.info('Epochs: ' + str(iterations))
        logger.info('Time Taken: ' + str(elapsed_time) + ' seconds.')
        logger.info('Train Error: ' + str(train_error))
        logger.info('Test Error: ' + str(test_error))
        logger.info('Procrustes Disparity: ' + str(procrustes_error))
        logger.info('kNN Classification Error on ground-truth: ' + str(knn_error_true_emb))
        logger.info('kNN Classification Error on embedding: ' + str(knn_error_ord_emb))


def run_hyper_search(config):
    dataset_name = config['dataset_selected']
    batch_size = config['sz_batch']
    iterations = config['nb_epochs']
    input_dim = config['input_dimension']
    subset_size = config['number_of_points']
    number_of_test_triplets = config['n_test_triplets']
    log_dir = config['log']['path']
    triplet_multiplier_range = config['hyper_search']['triplets_multiplier']
    learning_rate_range = config['hyper_search']['learning_rate']
    optimizer = config['optimizer']
    dimensions_range = config['hyper_search']['output_dimension']

    separator = '_'
    experiment_name = 'triplet_loss_hyper_search_' + \
                      'data_' + dataset_name + \
                      '_input_dim_' + str(input_dim) + \
                      '_n_pts_' + str(subset_size) + \
                      '_num_test_trips_' + str(number_of_test_triplets) + \
                      '_output_dim_' + separator.join([str(i) for i in dimensions_range]) + \
                      '_lr_' + separator.join([str(i) for i in learning_rate_range]) + \
                      '_bs_' + str(batch_size) + \
                      '_triplet_number_' + separator.join([str(i) for i in triplet_multiplier_range])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging_path = os.path.join(log_dir, experiment_name + '.log')
    logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)

    best_params_train = []
    best_params_test = []
    for dimensions in dimensions_range:
        best_train_error = 1
        best_test_error = 1
        for learning_rate in learning_rate_range:
            for triplet_multiplier in triplet_multiplier_range:
                # create data
                vec_data, labels = select_dataset(dataset_name, n_samples=subset_size, input_dim=input_dim)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                n = vec_data.shape[0]
                log_n = int(np.log2(n))
                triplet_num = triplet_multiplier * log_n * n * dimensions

                bs = min(batch_size, triplet_num)

                logger.info('Name of Experiment: ' + experiment_name)
                logger.info('Logging Path:' + logging_path)
                logger.info('Dataset Name: ' + dataset_name)
                logger.info('Epochs: ' + str(iterations))
                logger.info('Learning Rate: ' + str(learning_rate))
                logger.info('Number of Points: ' + str(n))
                logger.info('Input Dimension: ' + str(input_dim))
                logger.info('Output Dimension: ' + str(dimensions))
                logger.info('Number of Test Triplets: ' + str(number_of_test_triplets))
                logger.info('Triplet Multiplier: ' + str(triplet_multiplier))
                logger.info('Batch Size: ' + str(batch_size))

                train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, bs, device)

                logger.info('Computing SOE...')
                begin = time.time()

                x, loss_array, remain_loss = soe.triplet_loss_adam(triplets=train_triplets_dataset.trips_data_indices,
                                                                   n=n, dim=dimensions, iterations=iterations,
                                                                   bs=bs, lr=learning_rate, device=device,
                                                                   logger=logger)
                elapsed_time = time.time() - begin

                logger.info('Evaluating the computed embeddings...')
                # compute triplet error for train and test data
                train_error = train_triplets_dataset.triplet_error(x)
                logger.info('Triplet Error on Training Triplets: ' + str(train_error))
                test_triplets_dataset = TripletBatchesDataset(vec_data, labels, number_of_test_triplets, 1000, device)
                test_error = test_triplets_dataset.triplet_error(x)
                logger.info('Number of Test Triplets: ' + str(number_of_test_triplets))
                logger.info('Triplet Error on Test Triplets: ' + str(test_error))

                subsample = np.random.permutation(n)[0:500]
                x_for_tsne = x[subsample, :]
                sub_labels = labels[subsample]

                x_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x_for_tsne)
                fig, ax = plt.subplots(1, 1)

                ax.scatter(x_embedded[:, 0], x_embedded[:, 1], s=3, c=sub_labels)
                fig.savefig(os.path.join(log_dir, experiment_name + str(n) + '_instance' +
                                         '_lr_' + str(learning_rate) +
                                         '_dim_' + str(dimensions) +
                                         '_triplet_multipliers_' + str(triplet_multiplier) + '.png'))

                logger.info('Epochs: ' + str(iterations))
                logger.info('Time Taken: ' + str(elapsed_time) + ' seconds.')
                logger.info('Train Error: ' + str(train_error))
                logger.info('Test Error: ' + str(test_error))


                if test_error < best_test_error:
                    best_params_test.append({'learning_rate': learning_rate, 'optimizer': optimizer,
                                             'triplet_multiplier': triplet_multiplier, 'error': test_error})
                    best_test_error = test_error
                if train_error < best_train_error:
                    best_params_train.append({'learning_rate': learning_rate, 'optimizer': optimizer,
                                              'triplet_multiplier': triplet_multiplier, 'error': train_error})
                    best_train_error = train_error

    for dim_iter, emb_dim in enumerate(dimensions_range):
        logger.info('Best parameters for ' + str(dataset_name) + ' into dimension ' + str(emb_dim))
        best_on_train = best_params_train[dim_iter]
        best_on_test = best_params_test[dim_iter]
        logger.info('achieved ' + str(best_on_train['error']) +
                    ' train error with learning rate: ' + str(best_on_train['learning_rate']))
        logger.info('achieved ' + str(best_on_test['error']) +
                    ' test error with learning rate: ' + str(best_on_test['learning_rate']))


if __name__ == "__main__":
    main(parse_args())
