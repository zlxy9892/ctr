#coding=utf-8

import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import sparse
import pickle


def pickle_save_data(data, f_save):
    print('Starting pickle to save file...')
    with open(f_save, 'wb') as f:
        pickle.dump(data, f)
    print('Pickle save finished, file: <{}>.'.format(f_save))


def pickle_load_data(f_save):
    print('Starting pickle to load file...')
    with open(f_save, 'rb') as f:
        data = pickle.load(f)
    print('Pickle load finished')
    return data


def read_data():
    df_train = pd.read_csv('./data/train.csv')
    df_ad = pd.read_csv('./data/adFeature.csv')
    df_train = pd.merge(df_train, df_ad, on='aid', how='left')
    return df_train


def transform_categorical_features(df_data, category_feature_names=None):
    df_data_category = None
    if category_feature_names is None or type(category_feature_names) is not list:
        return None
    for cate_feat_name in category_feature_names:
        print('transforming category feature: {}'.format(cate_feat_name))
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
        data_one_col = np.array(df_data[cate_feat_name]).reshape((-1, 1))
        one_hot_encoder.fit(data_one_col)
        data_one_col_ = one_hot_encoder.transform(data_one_col)
        if df_data_category is None:
            df_data_category = data_one_col_
        else:
            df_data_category = np.hstack([df_data_category, data_one_col_])
    return df_data_category


def transform_features(df_data, category_feature_names=None):
    df_data_transformed = df_data.copy()
    df_data_category = transform_categorical_features(df_data, category_feature_names)
    if df_data_category is None:
        return df_data
    df_data_transformed = df_data_transformed.drop(category_feature_names, axis=1)
    df_data_transformed = np.hstack([df_data_transformed, df_data_category])
    return df_data_transformed


def label2y(labels):
    y = []
    for label in labels:
        if label <= 0:
            y.append(0)
        else:
            y.append(1)
    return np.array(y)


def under_sampling(x, y):
    y_unique = np.unique(y)
    assert len(y_unique) == 2, '只支持y值为两类，二分类的情况.'
    count_0 = 0
    count_1 = 0
    data_size = len(y)
    for i in range(data_size):
        if y[i] == y_unique[0]:
            count_0 += 1
        else:
            count_1 += 1
    shuffle_indices = np.random.permutation(np.arange(data_size))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    x_sampled = []
    y_sampled = []
    print(count_0, count_1)
    if count_0 < count_1:
        for i in range(data_size):
            if y[i] == y_unique[0]:
                x_sampled.append(x[i])
                y_sampled.append(y[i])
            else:
                if len(y_sampled) <= 2 * count_0:
                    x_sampled.append(x[i])
                    y_sampled.append(y[i])
                else:
                    break
    elif count_0 > count_1:
        for i in range(data_size):
            if y[i] == y_unique[1]:
                x_sampled.append(x[i])
                y_sampled.append(y[i])
            else:
                if len(y_sampled) <= 2 * count_1:
                    x_sampled.append(x[i])
                    y_sampled.append(y[i])
                else:
                    break
    else:
        x_sampled = x
        y_sampled = y
    x_sampled = np.array(x_sampled)
    y_sampled = np.array(y_sampled)
    return x_sampled, y_sampled


def csr2input(csr_mat):
    if not isinstance(csr_mat, list):
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
        values = csr_mat.data
        shape = csr_mat.shape
        return indices, values, shape
    else:
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr2input(csr_i))
        return inputs


def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """Generates a batch iterator for a dataset."""
    data_size = len(y)
    num_batches_per_epoch = int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_x = x[shuffle_indices]
            shuffled_y = y[shuffle_indices]
        else:
            shuffled_x = x
            shuffled_y = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*batch_size
            end_index = min((batch_num+1)*batch_size, data_size)
            yield shuffled_x[start_index:end_index], shuffled_y[start_index:end_index]


if __name__ == '__main__':
    pass
