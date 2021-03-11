import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn


class TextRNN(object):
    """
    A RNN for text classification.
    Uses an embedding layer, followed by a rnn layer.
    """

    def __init__(
            self, sequence_length, num_class, vocab_size,
            embedding_size, cell_type, hidden_size, pretrained_embeddings, l2_reg_lambda=0.0):

        # placeholders
        self.inputs = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="inputs")
        self.masks = tf.compat.v1.placeholder(tf.int32, [None], name="masks")
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
            self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)

        # create rnn layer
        with tf.compat.v1.name_scope("rnn"):
            cell = tf.nn.rnn_cell.MultiRNNCell([self._get_cell(hidden_size, cell_type) for _ in range(1)])
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            all_outputs, _ = dynamic_rnn(cell=cell,
                                         inputs=self.embedded_inputs,
                                         sequence_length=self.masks,
                                         dtype=tf.float32)
            self.outputs = self.last_relevant(all_outputs, self.masks)

        # add dropout
        with tf.compat.v1.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.outputs, self.dropout_keep_prob)

        # final (unnormalized) scores and predictions
        with tf.compat.v1.name_scope("output"):
            W = tf.compat.v1.get_variable(
                "W",
                shape=[hidden_size, num_class],
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

    @staticmethod
    def _get_cell(hidden_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

    # Extract the output of last cell of each sequence
    # Ex) The movie is good -> length = 4
    #     output = [ [1.314, -3.32, ..., 0.98]
    #                [0.287, -0.50, ..., 1.55]
    #                [2.194, -2.12, ..., 0.63]
    #                [1.938, -1.88, ..., 1.31]
    #                [  0.0,   0.0, ...,  0.0]
    #                ...
    #                [  0.0,   0.0, ...,  0.0] ]
    #     The output we need is 4th output of cell, so extract it.
    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        hidden_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, hidden_size])
        return tf.compat.v1.gather(flat, index)
