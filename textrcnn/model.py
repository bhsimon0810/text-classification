import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn


class TextRCNN(object):
    """
    A RCNN for text classification.
    Uses an embedding layer, followed by a rcnn layer.
    """

    def __init__(
            self, sequence_length, num_class, vocab_size,
            embedding_size, cell_type, hidden_size, pretrained_embeddings, l2_reg_lambda=0.0):

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
            self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)

        # create rnn layer
        with tf.compat.v1.name_scope("rcnn"):
            shape = [tf.shape(self.embedded_inputs)[0], 1, tf.shape(self.embedded_inputs)[2]]
            embedded_fw = tf.concat([tf.zeros(shape), self.embedded_inputs[:, : -1, :]], axis=1, name="left_inputs")
            embedded_bw = tf.reverse(tf.concat([self.embedded_inputs[:, 1:, :], tf.zeros(shape)], axis=1), axis=[1],
                                     name="right_inputs")

            # forward step --> left contexts
            with tf.compat.v1.variable_scope("left_contexts"):
                cell_fw = tf.nn.rnn_cell.MultiRNNCell([self._get_cell(hidden_size, cell_type) for _ in range(1)])
                cell_fw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
                outputs_fw, _ = dynamic_rnn(cell=cell_fw, inputs=embedded_fw, dtype=tf.float32)

            # backward step --> right contexts
            with tf.compat.v1.variable_scope("right_contexts"):
                cell_bw = tf.nn.rnn_cell.MultiRNNCell([self._get_cell(hidden_size, cell_type) for _ in range(1)])
                cell_bw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)
                outputs_bw, _ = dynamic_rnn(cell=cell_bw, inputs=embedded_bw, dtype=tf.float32)

            # word representations
            outputs_bw = tf.reverse(outputs_bw, axis=[1])
            word_representations = tf.concat([outputs_fw, self.embedded_inputs, outputs_bw], axis=2, name="word_representations")

            # sentence representations
            with tf.compat.v1.name_scope("sentence_representations"):
                W = tf.Variable(tf.random.normal([embedding_size + hidden_size * 2, hidden_size]), name="weights")
                b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="biases")
                sent_representations = tf.tanh(tf.einsum("aij,jk->aik", word_representations, W) + b)

            # max pooling layer
            with tf.compat.v1.name_scope("pooling"):
                self.pooled = tf.reduce_max(sent_representations, axis=1)

        # add dropout
        with tf.compat.v1.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.pooled, self.dropout_keep_prob)

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
        with tf.compat.v1.name_scope("loss"):
            labels = tf.one_hot(self.labels, depth=num_class)
            losses = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=self.logits)
            l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()]) * l2_reg_lambda
            self.loss = tf.reduce_mean(losses) + l2_reg

        # accuracy
        with tf.compat.v1.name_scope("accuracy"):
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
