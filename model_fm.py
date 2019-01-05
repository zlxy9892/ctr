#coding=utf-8

import numpy as np
import tensorflow as tf


class Model_FM(object):
    def __init__(self, n_features=None, order=2, rank=5, input_type='sparse',
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.01), reg=0.0,
                 init_std=0.1, device_name='/cpu:0', seed=None):
        self.n_features = n_features
        self.order = order
        self.rank = rank
        self.input_type = input_type
        self.optimizer = optimizer
        self.reg = reg
        self.init_std = init_std
        self.device_name = device_name
        self.graph = None
        self.seed = seed

    def init_learnable_params(self):
        # self.order = 2
        # self.w = [None] * self.order
        # for i in range(1, self.order+1):
        #     r = self.rank
        #     if i == 1:
        #         r = 1
        #     rand_weights = tf.random_uniform([self.n_features, r], -1, 1)
        #     self.w[i-1] = tf.verify_tensor_all_finite(
        #         tf.Variable(rand_weights, trainable=True, name='embedding_{}'.format(str(i))),
        #         msg='NaN or Inf in w[{}].'.format(i-1))
        rand_weights = tf.truncated_normal([self.n_features, 1], mean=0.0, stddev=self.init_std, dtype=tf.float32)
        self.w1 = tf.Variable(rand_weights, trainable=True, name='w1', dtype=tf.float32)
        rand_v = tf.truncated_normal([self.n_features, self.rank], mean=0.0, stddev=self.init_std, dtype=tf.float32)
        self.v = tf.Variable(rand_v, trainable=True, name='v', dtype=tf.float32)
        self.b = tf.Variable(self.init_std, trainable=True, name='bias', dtype=tf.float32)
        tf.summary.scalar('bias', self.b)

    def init_placeholders(self):
        if self.input_type == 'dense':
            self.input_x = tf.placeholder(tf.float32, shape=[None, self.n_features], name='input_x')
        else:   # input type is sparse
            self.raw_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
            self.raw_values = tf.placeholder(tf.float32, shape=[None], name='raw_values')
            self.raw_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')
            self.input_x = tf.SparseTensor(self.raw_indices, self.raw_values, self.raw_shape)
        self.input_y = tf.placeholder(tf.float32, shape=[None], name='input_y')

    def init_main_block(self):
        x_square = tf.SparseTensor(self.raw_indices, tf.square(self.raw_values), self.raw_shape)
        self.xv = tf.square(tf.sparse_tensor_dense_matmul(self.input_x, self.v))
        self.xw2 = 0.5 * tf.reshape(
            tf.reduce_sum(self.xv - tf.sparse_tensor_dense_matmul(x_square, tf.square(self.v)), 1),
            [-1, 1])
        self.xw1 = tf.sparse_tensor_dense_matmul(self.input_x, self.w1)
        self.logits = tf.reshape(self.b + self.xw1 + self.xw2, [-1])
        self.y_predproba = tf.sigmoid(self.logits)
        self.y_pred = tf.less

    def init_loss(self):
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))

    def build_graph(self):
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        with self.graph.as_default():
            with tf.device(self.device_name):
                with tf.name_scope('learnable_params'):
                    self.init_learnable_params()
                with tf.name_scope('input_block'):
                    self.init_placeholders()
                with tf.name_scope('main_block'):
                    self.init_main_block()
                with tf.name_scope('optimization_criterion'):
                    self.init_loss()
                # self.trainer = self.optimizer.minimize(self.loss)
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
                self.init_all_vars = tf.global_variables_initializer()
                for var in tf.trainable_variables():
                    tf.summary.histogram(name=var.name, values=var)
                self.train_summary_op = tf.summary.merge_all()
                self.saver = tf.train.Saver()
