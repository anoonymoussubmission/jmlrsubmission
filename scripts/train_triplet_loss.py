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

from lib.soe import triplet_loss_adam
from preprocessing_utils.data_select_utils import select_dataset
from preprocessing_utils.TripletData import TripletBatchesDataset
from logging_utils import logging_util


def parse_args():
    parser = argparse.ArgumentParser(description='Run SOE Experiments')
    parser.add_argument('-d', '--dataset_name', type=str, default='mnist', required=False,
                        help='Select the dataset, default is: mnist')
    parser.add_argument('-bs', '--batch_size', type=int, default=1000000, required=False,
                        help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, required=False,
                        help='Learning rate')
    parser.add_argument('-ep', '--epochs', type=int, default=200, required=False,
                        help='Number of epochs or iterations of convergence')
    parser.add_argument('-dim', '--dim', type=int, default=10, required=False,
                        help='Number of dimensions for embedding triplets')
    parser.add_argument('-idim', '--input_dim', type=int, default=10, required=False,
                        help='Dim for the synthetic data generation or in the case of CharPC for PCA dimensions')
    parser.add_argument('-num', '--subset_size', type=int, default=1000, required=False,
                        help='Number of Subsets to be selected from a dataset, only for testing purposes')
    parser.add_argument('--hyper_search', type=bool, default=False, required=False,
                        help='If true, perform a gridsearch over hyperparameters')
    args = parser.parse_args()
    return args


def main(args):

    dataset_name = args.dataset_name
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    iterations = args.epochs
    dimensions = args.dim
    subset_size = args.subset_size
    input_dim = args.input_dim
    vec_data, labels = select_dataset(dataset_name, n_samples=subset_size, input_dim=input_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n = vec_data.shape[0]
    logn = int(np.log2(n))
    triplet_num = 2 * logn * n * dimensions

    bs = min(batch_size, triplet_num)

    train_triplets_dataset = TripletBatchesDataset(vec_data, labels, triplet_num, bs, device)

    begin = time.time()
    x, loss_array, remain_loss = triplet_loss_adam(train_triplets_dataset.trips_data_indices, n, dimensions,
                                                   iterations, bs,
                                                   lr=learning_rate)
    elapsed_time = time.time() - begin

    # compute triplet error for train and test data
    train_error = train_triplets_dataset.triplet_error(x)
    test_triplets_dataset = TripletBatchesDataset(vec_data, labels, 1000, 1000, device)
    test_error = test_triplets_dataset.triplet_error(x)

    subsample = np.random.permutation(n)[0:500]
    x = x[subsample, :]
    sublabel = labels[subsample]

    experiment_name = 'data_' + dataset_name + '_input_dim_' \
                      + str(input_dim) + '_dimensions_' \
                      + str(dimensions) + '_triplet_num_' + str(triplet_num) + '_n_' + str(n)

    X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x)
    fig, ax = plt.subplots(1, 1)

    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=3, c=sublabel)
    fig.savefig('logs/' + experiment_name + '-' + str(n) + '.png')

    logging_path = 'logs/' + experiment_name + '.log'
    logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
    logger.info('Name of Experiments: ' + experiment_name)
    logging.info('Epochs: ' + str(iterations))
    logging.info('Time Taken: ' + str(elapsed_time) + ' seconds.')
    logging.info('Train Error: ' + str(train_error))
    logging.info('Test Error: ' + str(test_error))


if __name__ == "__main__":
    main(parse_args())