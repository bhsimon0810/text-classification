import tensorflow as tf
from attention import attention
from utils import get_shape


class HieAttNet(object):
    def __init__(self, cell_type, hid_size, att_size, vocab_size, emb_size, num_classes, pretrained_embs,
                 l2_reg_lambda=0.0):
        self.hid_size = hid_size
        self.att_size = att_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.num_classes = num_classes
        self.pretrained_embs = pretrained_embs

        # placeholders
        self.docs = tf.placeholder(tf.int32, shape=[None, None, None], name="docs")
        self.sent_length = tf.placeholder(tf.int32, shape=[None], name="sent_length")
        self.word_length = tf.placeholder(tf.int32, shape=[None, None], name="word_length")
        self.max_sent_length = tf.placeholder(tf.int32, name="max_sent_length")
        self.max_word_length = tf.placeholder(tf.int32, name="max_word_length")
        self.labels = tf.placeholder(tf.int32, shape=[None], name="labels")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding layer
        with tf.device('/gpu:0'), tf.variable_scope("embedding"):
            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.emb_size],
                initializer=tf.constant_initializer(self.pretrained_embs),
                dtype=tf.float32
            )
            self.embedded_docs = tf.nn.embedding_lookup(self.embedding_matrix, self.docs)

        # word-level encoder
        with tf.variable_scope("word-encoder"):
            word_level_inputs = tf.reshape(self.embedded_docs, [-1, self.max_word_length, self.emb_size])
            word_length = tf.reshape(self.word_length, [-1])
            # rnn cell
            cell_fw = self._get_cell(self.hid_size, self.cell_type)
            cell_bw = self._get_cell(self.hid_size, self.cell_type)
            initial_state_fw = tf.tile(
                tf.get_variable("initial_state_fw", shape=[1, self.hid_size], initializer=tf.constant_initializer(0)),
                multiples=[get_shape(word_level_inputs)[0], 1]
            )
            initial_state_bw = tf.tile(
                tf.get_variable("initial_state_bw", shape=[1, self.hid_size], initializer=tf.constant_initializer(0)),
                multiples=[get_shape(word_level_inputs)[0], 1]
            )
            word_enc_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=word_level_inputs,
                sequence_length=word_length,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
            )
            self.word_enc_outputs = tf.concat(word_enc_outputs, axis=2)

        # word-level attention
        with tf.variable_scope("word-attention"):
            word_att_outputs, _ = attention(self.word_enc_outputs, self.att_size)
            self.word_level_outputs = tf.contrib.layers.dropout(word_att_outputs, self.dropout_keep_prob,
                                                                is_training=self.is_training)

        # sent-level encoder
        with tf.variable_scope("sent-encoder"):
            sent_level_inputs = tf.reshape(self.word_level_outputs, [-1, self.max_sent_length, 2 * self.hid_size])
            # rnn cell
            cell_fw = self._get_cell(self.hid_size, self.cell_type)
            cell_bw = self._get_cell(self.hid_size, self.cell_type)
            initial_state_fw = tf.tile(
                tf.get_variable("initial_state_fw", shape=[1, self.hid_size], initializer=tf.constant_initializer(0)),
                multiples=[get_shape(sent_level_inputs)[0], 1]
            )
            initial_state_bw = tf.tile(
                tf.get_variable("initial_state_bw", shape=[1, self.hid_size], initializer=tf.constant_initializer(0)),
                multiples=[get_shape(sent_level_inputs)[0], 1]
            )
            sent_enc_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=sent_level_inputs,
                sequence_length=self.sent_length,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
            )
            self.sent_enc_outputs = tf.concat(sent_enc_outputs, axis=2)

        # sent-level attention
        with tf.variable_scope("sent-attention"):
            sent_att_outputs, _ = attention(self.sent_enc_outputs, self.att_size)
            self.sent_level_outputs = tf.contrib.layers.dropout(sent_att_outputs, self.dropout_keep_prob,
                                                                is_training=self.is_training)

        # final scores and predictions
        with tf.variable_scope("output"):
            W = tf.get_variable("W", shape=[2 * self.hid_size, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.sent_level_outputs, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # cross entropy loss and l2 regularization
        with tf.variable_scope("loss"):
            labels = tf.one_hot(self.labels, depth=self.num_classes)
            losses = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=self.logits)
            l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2_reg_lambda
            self.loss = tf.reduce_mean(losses) + l2_reg

        # accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    @staticmethod
    def _get_cell(hid_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hid_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hid_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hid_size)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None
