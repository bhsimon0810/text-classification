import tensorflow as tf


class FastText(object):
    def __init__(self, vocab_size, embedding_size, num_class, pretrained_embeddings,
                 l2_reg_lambda=0.0):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_class = num_class
        self.pretrained_embeddings = pretrained_embeddings

        # placeholders
        self.inputs = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="inputs")
        # self.inputs_mask = tf.compat.v1.placeholder(tf.float32, shape=[None, None], name="inputs_mask")
        self.labels = tf.compat.v1.placeholder(tf.int32, shape=[None], name="labels")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding layer
        with tf.device('/cpu:0'), tf.compat.v1.variable_scope("embedding"):
            self.embedding_matrix = tf.compat.v1.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=tf.constant_initializer(self.pretrained_embeddings),
                dtype=tf.float32
            )
            self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)

        # final scores and predictions
        with tf.compat.v1.variable_scope("output"):
            # mask = tf.tile(tf.expand_dims(self.inputs_mask, -1), [1, 1, self.embedding_size])
            # masked_inputs = tf.reduce_mean(self.embedded_inputs * mask, 1)
            inputs = tf.reduce_mean(self.embedded_inputs, 1)
            outputs = tf.contrib.layers.dropout(inputs, self.dropout_keep_prob, is_training=self.is_training)
            W = tf.compat.v1.get_variable("Weights", shape=[self.embedding_size, self.num_class],
                                          initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_class]), name="biases")
            self.logits = tf.compat.v1.nn.xw_plus_b(outputs, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, -1, name="predictions")

        # cross entropy loss and l2 regularization
        with tf.compat.v1.variable_scope("loss"):
            labels = tf.one_hot(self.labels, depth=self.num_class)
            losses = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=self.logits)
            l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()]) * l2_reg_lambda
            self.loss = tf.reduce_mean(losses) + l2_reg

        # accuracy
        with tf.compat.v1.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
