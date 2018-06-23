#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.layers import xavier_initializer
from datetime import datetime
import math
import time


class model(object):
    def __init__(self, batch_size, learni_rate, keep_prob, epoch, input_size):
        self.batch_size         = batch_size
        self.learn_rate         = learni_rate
        self.keep_prob          = keep_prob
        self.epoch              = epoch
        self.input_size         = input_size

    def conv_op(self, input_op, name, ksize, n_out, stride, p):
        n_in = input_op.get_shape().value

        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(name       =scope+'w',
                                     shape      =[ksize[0], ksize[1], ksize[2], n_in, n_out],
                                     dtype      =tf.float32,
                                     initializer=xavier_initializer()
                                     )
            conv = tf.nn.conv3d(input           =input_op,
                                filter          =kernel,
                                strides         =[1, stride[0], stride[1], stride[2], 1],
                                padding         ='SAME'
                                )
            bias_init_val = tf.constant(value   =0.0,
                                        shape   =[n_out],
                                        dtype   =tf.float32
                                        )
            biases = tf.Variable(initial_value  =bias_init_val,
                                 trainable      =True,
                                 name           ='b'
                                 )
            z = tf.nn.bias_add(conv, biases)
            activation = tf.nn.relu(z, name=scope)
            p += [kernel, biases]
            return activation

    def mpool_op(self, input_op, name, ksize, stride):
        return tf.nn.max_pool3d(input           =input_op,
                                ksize           =[1, ksize[0], ksize[1], ksize[2], 1],
                                strides         =[1, stride[0], stride[1], stride[2], 1],
                                padding         ='SAME',
                                name            =name
                                )

    def fc_op(self, input_op, name, n_out, p):
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(name       =scope+'w',
                                     shape      =[n_in, n_out],
                                     dtype      =tf.float32,
                                     initializer=xavier_initializer()
                                     )
            biases = tf.Variable(initial_value  =tf.constant(value    =0.1,
                                                            shape     =[n_out],
                                                            dtype     =tf.float32,
                                                            name      ='b')
                                )
            activation = tf.nn.relu_layer(x     =input_op,
                                          weights=kernel,
                                          biases=biases,
                                          name  =scope
                                          )
            p += [kernel, biases]
            return activation

    def inference_op(self, input_op, keep_prob):
        p = []
        conv1_1 = self.conv_op(input_op=input_op, name='conv1_1', ksize=[3, 3, 3], n_out=16, stride=[1, 1, 1], p=p)
        conv1_2 = self.conv_op(input_op=conv1_1, name='conv1_2', ksize=[3, 3, 3], n_out=24, stride=[1, 1, 1], p=p)
        pool1 = self.mpool_op(input_op=conv1_2, name='pool1', ksize=[2, 2, 2], stride=[2, 2, 2])

        conv2_1 = self.conv_op(input_op=pool1, name='conv2_1', ksize=[3, 3, 3], n_out=32, stride=[1, 1, 1], p=p)
        conv2_2 = self.conv_op(input_op=conv2_1, name='conv2_2', ksize=[3, 3, 3], n_out=32, stride=[1, 1, 1], p=p)
        pool2 = self.mpool_op(input_op=conv2_2, name='pool2', ksize=[2, 2, 2], stride=[2, 2, 2])

        conv3_1 = self.conv_op(input_op=pool2, name='conv3_1', ksize=[3, 3, 3], n_out=64, stride=[1, 1, 1], p=p)
        conv3_2 = self.conv_op(input_op=conv3_1, name='conv3_2', ksize=[3, 3, 3], n_out=64, stride=[1, 1, 1], p=p)
        pool3 = self.mpool_op(input_op=conv3_2, name='pool3', ksize=[2, 2, 2], stride=[2, 2, 2])

        conv4_1 = self.conv_op(input_op=pool3, name='conv4_1', ksize=[3, 3, 3], n_out=128, stride=[1, 1, 1], p=p)
        conv4_2 = self.conv_op(input_op=conv4_1, name='conv4_2', ksize=[3, 3, 3], n_out=128, stride=[1, 1, 1], p=p)
        pool4 = self.mpool_op(input_op=conv4_2, name='pool4', ksize=[2, 2, 2], stride=[2, 2, 2])

        shp = pool4.get_shape()
        flattened_shape = shp[1].value*shp[2].value*shp[3].value*shp[4].value
        resh1 = tf.reshape(tensor=pool4, shape=[-1, flattened_shape], name='resh1')

        fc5 = self.fc_op(input_op=resh1, name='fc5', n_out=32, p=p)
        fc5_drop = tf.nn.dropout(x=fc5, keep_prob=keep_prob, name='fc5_drop')

        fc6 = self.fc_op(input_op=fc5_drop, name='fc6', n_out=2, p=p)
        logits = fc6
        softmax = tf.nn.softmax(logits=fc6)
        prediction = tf.argmax(softmax, axis=1, name='prediction')
        return prediction, softmax, logits, p

    def loss_op(self, input_op, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(features=input_op, labels=labels))
        return loss

    def optimizer_op(self, input_op, learn_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss=input_op)
        return optimizer

    def confusionMatrix_op(self, logits, labels):
        predict = tf.argmax(input=logits, axis=1)
        predict_shp = tf.shape(logits, out_type=tf.int32)
        labels_reshp = tf.reshape(tensor=labels,
                                shape=tf.slice(input_=predict_shp,
                                               begin=tf.constant(0, shape=(1,), dtype=tf.int32),
                                               size=tf.shape(predict_shp) - 1))
        labels_reshp = tf.cast(x=tf.greater(x=labels_reshp, y=0), dtype=tf.int32)
        predict_reshp = tf.reshape(tensor=predict, shape=[-1])
        labels_reshp_sec = tf.reshape(tensor=labels_reshp, shape=[-1])
        confusionmatrix = tf.contrib.confusion_matrix(predict_reshp,
                                                      labels_reshp_sec,
                                                      stype=tf.int32,
                                                      num_classes=2,
                                                      name='ConfusionMatrix')
        return confusionmatrix

    def trian_op(self, ):
        print('training...........')


    def time_tf_run(self, session, target, feed, info_string):
        num_batches = 100
        num_steps_brun_in = 10
        total_duration = 0.0
        total_duration_squared = 0.0
        for i in range(num_batches + num_steps_brun_in):
            start_time = time.time()
            _ = session.run(target, feed_dict=feed)
            duration = time.time() - start_time
            if i >= num_steps_brun_in:
                if not i % 10:
                    print('%s: step %d, duration: %3f' % (datetime.now(), i - num_steps_brun_in, duration))
                total_duration += duration
                total_duration_squared += duration*duration
        mn = total_duration / num_batches
        vr = total_duration_squared / num_batches - mn*mn
        sd = math.sqrt(vr)
        print('%s: %s across %d steps, %.3f+/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))

    def run_benchmark(self):
        with tf.Graph().as_default():
            image_size = 224
            images = tf.Variable(tf.random_normal(shape=[self.batch_size, image_size, image_size, 3],
                                                  dtype=tf.float32,
                                                  stddev=1e-1
                                                  )
                                 )
            keep_prob = tf.placeholder(dtype=tf.float32)
            predictions, softmax, logits, p = self.inference_op(input_op=images, keep_prob=keep_prob)
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            self.time_tf_run(session=sess,
                             target=predictions,
                             feed={keep_prob:1.0},
                             info_string='FORWARD')
            objective = tf.nn.l2_loss(t=logits)
            grad = tf.gradients(ys=objective, xs=p)
            self.time_tf_run(session=sess,
                             target=grad,
                             feed={keep_prob:0.5},
                             info_string='FORWARD-benchmark')

















