import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import torch
import os
import sys
import argparse
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_utils.TripletData import triplet_error_torch
from preprocessing_utils.data_select_utils import select_dataset


def parse_args():
    """
    To run this file use "CUDA_VISIBLE_DEVICES=3 python train_soe.py -config configs/soe/soe_evaluation.json". See
    the config file in
    the path for an example of how to construct config files.
    """
    parser = argparse.ArgumentParser(description='Run Final Evaluation and save it.')
    parser.add_argument('-path', '--data_path', type=str, default='', required=True,
                        help='Input the Data File Path for evaluation')
    parser.add_argument('-equal_d', '--input_equals_output', type=int, default=0,
                        help='For increasing_org_d experiment, we just have equal input and output dimension.')
    parser.add_argument('-data', '--full_dataset_name', type=str, default='not_selected',
                        help='If in this experiment, we look at the whole dataset, we recompute the original kNN error.')
    args = parser.parse_args()
    return args

def compute_evaluations(args):

    with open(args.data_path, 'rb') as f:
        data_file = joblib.load(f)

    y_labels = ['Train Error', 'Test Error', 'Procrustes Error', 'Knn Orig Error', 'Knn Ordinal Error', 'Time']
    experiment_range = data_file[1]
    results = data_file[2]

    values = defaultdict(dict)

    for input_dim_index, input_dim in enumerate(experiment_range['input_dim']):
        for dimensions_index, embedding_dimension in enumerate(experiment_range['output_dim']):
            for subset_index, nmb_points in enumerate(experiment_range['number_of_points']):
                for batch_size_index, batch_size in enumerate(experiment_range['batch_size']):
                    for lr_index, learning_rate in enumerate(experiment_range['learning_rate']):
                        for trip_index, triplet_multiplier in enumerate(experiment_range['triplet_multiplier']):
                            if (not args.input_equals_output) or (args.input_equals_output and input_dim == embedding_dimension):
                                values[input_dim_index, dimensions_index, subset_index,
                                       batch_size_index, lr_index, trip_index, 1] = results[input_dim_index, dimensions_index, subset_index,
                                                                                    batch_size_index, lr_index, trip_index, 1]
                                values[input_dim_index, dimensions_index, subset_index,
                                       batch_size_index, lr_index, trip_index, 2] = results[input_dim_index, dimensions_index, subset_index,
                                                                                    batch_size_index, lr_index, trip_index, 2]
                                values[input_dim_index, dimensions_index, subset_index,
                                       batch_size_index, lr_index, trip_index, 3] = results[input_dim_index, dimensions_index, subset_index,
                                                                                    batch_size_index, lr_index, trip_index, 3]
                                values[input_dim_index, dimensions_index, subset_index,
                                       batch_size_index, lr_index, trip_index, 4] = results[input_dim_index, dimensions_index, subset_index,
                                                                                    batch_size_index, lr_index, trip_index, 4]
                                values[input_dim_index, dimensions_index, subset_index,
                                       batch_size_index, lr_index, trip_index, 5] = results[input_dim_index, dimensions_index, subset_index,
                                                                                    batch_size_index, lr_index, trip_index, 5]

                                x = results[input_dim_index, dimensions_index, subset_index, batch_size_index, lr_index, trip_index, 6]
                                labels = results[input_dim_index, dimensions_index, subset_index, batch_size_index, lr_index, trip_index, 7]

                                values[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_index, 6] = x
                                values[input_dim_index, dimensions_index, subset_index,
                                                 batch_size_index, lr_index, trip_index, 7] = labels

                                # compute knn error on ordinal embedding
                                values[input_dim_index, dimensions_index, subset_index,
                                       batch_size_index, lr_index, trip_index, 4] = knn_classification_error(x, labels)
                                if args.full_dataset_name != 'not_selected':
                                    vec_data, true_labels = select_dataset(args.full_dataset_name, n_samples=-1, input_dim=-1)
                                    values[input_dim_index, dimensions_index, subset_index,
                                           batch_size_index, lr_index, trip_index, 3] = knn_classification_error(vec_data, true_labels)

                                if results[input_dim_index, dimensions_index, subset_index, batch_size_index, lr_index, trip_index, 0] == -1:
                                    print('Computing Training Error')
                                    # compute training error with train triplets
                                    train_triplets = results[input_dim_index, dimensions_index, subset_index, batch_size_index, lr_index, trip_index, 8]
                                    triplet_num = train_triplets.shape[0]
                                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                    train_triplets = torch.tensor(train_triplets).to(device).long()
                                    x = torch.Tensor(x).to(device)

                                    batch_size = 1000000

                                    batches = 1 if batch_size > triplet_num else triplet_num // batch_size
                                    triplet_error = 0
                                    for batch_ind in range(batches):
                                        batch_trips = train_triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets
                                        batch_triplet_error = triplet_error_torch(x, batch_trips)[0].item()
                                        triplet_error += batch_triplet_error
                                    triplet_error = triplet_error / batches
                                    print('Triplet error: ', triplet_error)
                                    values[input_dim_index, dimensions_index, subset_index,
                                           batch_size_index, lr_index, trip_index, 0] = triplet_error

                                else:
                                    values[input_dim_index, dimensions_index, subset_index,
                                           batch_size_index, lr_index, trip_index, 0] = results[input_dim_index, dimensions_index, subset_index,
                                                                                    batch_size_index, lr_index, trip_index, 0]

    data_dump = [y_labels, experiment_range, values]
    joblib.dump(data_dump, args.data_path.replace('.pkl', '_final_evaluation.pkl'))
    print('Finished Final Evaluation')



def knn_classification_error(emb, labels):
    """
    Description: Compute the kNN classification error on a test set (70/30 split)  trained on
    the ground-truth embedding and the ordinal embedding.
    :param ord_emb: ordinal embedding
    :param true_emb: ground-truth embedding
    :param labels: labels of the data points
    :return classification error on test data on ordinal embedding and ground-truth embedding.
    """
    n_neighbors = int(np.log2(emb.shape[0]))
    x_train, x_test, y_train, y_test = train_test_split(emb, labels, train_size=0.7)
    ordinal_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    ordinal_classifier.fit(x_train, y_train)
    return 1 - ordinal_classifier.score(x_test, y_test)


if __name__ == '__main__':
    compute_evaluations(parse_args())