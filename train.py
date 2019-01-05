#coding=utf-8

import sys
import numpy as np
import tensorflow as tf
from scipy import sparse
import utils
from model_fm import Model_FM


# hyper parameters
num_epochs = 10
batch_size = 32
evaluate_every = 100

# read data
# df_data = utils.read_data()
# feature_names = ['aid', 'uid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']
# categorical_feature_names = ['aid']
# df_x = df_data[feature_names]
# df_x = utils.transform_features(df_x, categorical_feature_names)    # one hot encoding on categorical features
# x = np.array(df_x)
# y = np.array(df_data['label'])
# y = utils.label2y(y)
# x, y = utils.under_sampling(x, y)
# x_csr = sparse.csr_matrix(x)
# print(x_csr.shape)
# print(y.shape)
# utils.pickle_save_data(x_csr, './data/x_csr.pkl')
# utils.pickle_save_data(y, './data/y.pkl')

x_csr = utils.pickle_load_data('./data/x_csr.pkl')
y = utils.pickle_load_data('./data/y.pkl')
print(x_csr.shape)
print(y.shape)

x_indices, x_values, x_shape = utils.csr2input(x_csr)

# start train
model = Model_FM(
    n_features=x_csr.shape[1],
    input_type='sparse',
    seed=314)
model.build_graph()

session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(config=session_conf, graph=model.graph)
with sess.as_default():
    sess.run(model.init_all_vars)

    batches = utils.batch_iter(
        x=x_csr, y=y, batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
    for batch in batches:
        # x_batch, y_batch = batch
        x_batch = x_csr[:batch_size]
        y_batch = y[:batch_size]
        x_batch_indices, x_batch_values, x_batch_shape = utils.csr2input(x_batch)

        feed_dict = {
            model.raw_indices: x_batch_indices,
            model.raw_values: x_batch_values,
            model.raw_shape: x_batch_shape,
            model.input_y: y_batch
        }

        _, step, summary, logits, y_pred_proba, loss = sess.run(
            [model.train_op, model.global_step, model.train_summary_op, model.logits, model.y_predproba, model.loss],
            feed_dict)

        # print('logits:\n{}'.format(logits))
        # print('y_true:\n{}'.format(y_batch))
        # print('y_predproba:\n{}'.format(y_pred_proba))

        if step % evaluate_every == 0:
            print('step: {} | loss: {}'.format(step, loss))
