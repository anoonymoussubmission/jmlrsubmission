from preprocessing_utils import data_select_utils

print('Dataset Name -  Samples - dataset_id')
# birch1
data, labels = data_select_utils.select_dataset(dataset_name='birch1')
print('birch1 - ', data.shape[0], ' - ', 'birch1')

# birch2
data, labels = data_select_utils.select_dataset(dataset_name='birch2')
print('birch2 - ', data.shape[0], ' - ', 'birch2')

# birch3
data, labels = data_select_utils.select_dataset(dataset_name='birch3')
print('birch3 - ', data.shape[0], ' - ', 'birch3')

# worms
data, labels = data_select_utils.select_dataset(dataset_name='worms')
print('worms - ', data.shape[0], ' - ', 'worms')

# d31
data, labels = data_select_utils.select_dataset(dataset_name='d31')
print('d31 - ', data.shape[0], ' - ', 'd31')

# t48k
data, labels = data_select_utils.select_dataset(dataset_name='t48k')
print('t48k - ', data.shape[0], ' - ', 't48k')

# blobs
data, labels = data_select_utils.select_dataset(dataset_name='blobs')
print('blobs - ', data.shape[0], ' - ', 'blobs')

# mnist
data, labels = data_select_utils.select_dataset(dataset_name='mnist')
print('MNIST - ', data.shape[0], ' - ', 'mnist')

# fmnist
data, labels = data_select_utils.select_dataset(dataset_name='fmnist')
print('FashionMNIST - ', data.shape[0], ' - ', 'fmnist')

# gmm
data, labels = data_select_utils.select_dataset(dataset_name='gmm', n_samples=128)
print('Gaussian Mixture Models - ', data.shape[0], ' - ', 'gmm')

# emnist
data, labels = data_select_utils.select_dataset(dataset_name='emnist')
print('EMNIST - ', data.shape[0], ' - ', 'emnist')

# kmnist
data, labels = data_select_utils.select_dataset(dataset_name='kmnist')
print('KMNIST - ', data.shape[0], ' - ', 'kmnist')

# usps
data, labels = data_select_utils.select_dataset(dataset_name='usps')
print('USPS - ', data.shape[0], ' - ', 'usps')

# cover_type
data, labels = data_select_utils.select_dataset(dataset_name='cover_type')
print('CoverType - ', data.shape[0], ' - ', 'cover_type')

# charfonts
data, labels = data_select_utils.select_dataset(dataset_name='charfonts')
print('CharFonts - ', data.shape[0], ' - ', 'charfonts')

# coil20
data, labels = data_select_utils.select_dataset(dataset_name='coil20')
print('COIL-20  - ', data.shape[0], ' - ', 'coil20')

# news_groups
#data, labels = data_select_utils.select_dataset(dataset_name='news')
#print('News20: ', data.shape)

# char
data, labels = data_select_utils.select_dataset(dataset_name='char')
print('Char - ', data.shape[0], ' - ', 'char')

# kdd
data, labels = data_select_utils.select_dataset(dataset_name='kdd_cup')
print('KDD Cup - ', data.shape[0], ' - ', 'kdd_cup')
