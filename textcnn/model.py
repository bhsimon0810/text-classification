import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_class, vocab_size,
            embedding_size, filter_sizes, num_filter, pretrained_embeddings, l2_reg_lambda=0.0):

        # placeholders
        self.inputs = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="inputs")
        self.labels = tf.compat.v1.placeholder(tf.int32, [None], name="labels")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding layer
        with tf.device('/cpu:0'), tf.compat.v1.variable_scope("embedding"):
            self.embedding_matrix = tf.compat.v1.get_variable(
                name="embedding_matrix",
                shape=[vocab_size, embedding_size],
                initializer=tf.constant_initializer(pretrained_embeddings),
                dtype=tf.float32)

        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #     self.embedding_matrix = tf.Variable(
        #         tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        #         name="W")
            self.embedded_inputs = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_matrix, self.inputs), -1)

        # create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.compat.v1.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filter]
                W = tf.compat.v1.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.compat.v1.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool2d(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # combine all the pooled features
        num_filters_total = num_filter * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout
        with tf.compat.v1.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # final (unnormalized) scores and predictions
        with tf.compat.v1.name_scope("output"):
            W = tf.compat.v1.get_variable(
                "W",
                shape=[num_filters_total, num_class],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_class]), name="b")
            self.logits = tf.compat.v1.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            labels = tf.one_hot(self.labels, depth=num_class)
            losses = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=self.logits)
            l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()]) * l2_reg_lambda
            self.loss = tf.reduce_mean(losses) + l2_reg

        # accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
