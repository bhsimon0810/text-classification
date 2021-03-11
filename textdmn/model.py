import numpy as np
import tensorflow as tf
from attention_gru_cell import AttentionGRUCell


def _position_encoding(max_num_words, emb_size):
    encoding = np.ones((emb_size, max_num_words), dtype=np.float32)
    ls = max_num_words + 1
    le = emb_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / emb_size / max_num_words
    return np.transpose(encoding)


class DyMemNet(object):
    def __init__(self, hid_size, vocab_size, emb_size, num_classes, num_hops, pretrained_embs, l2_reg_lambda=0.0):
        self.hid_size = hid_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.num_hops = num_hops
        self.max_num_sents = 20
        self.max_num_words = 50
        self.l2_reg_lambda = l2_reg_lambda

        self.add_placeholders()
        self.embeddings = self.get_embeddings(pretrained_embs)
        self.position_encoding = _position_encoding(self.max_num_words, self.emb_size)
        self.logits = self.inference()
        self.predictions, self.accuracy = self.get_predictions_and_accuracy(self.logits)
        self.loss = self.get_loss(self.logits)

    def add_placeholders(self):
        self.context_placeholder = tf.placeholder(tf.int32, shape=[None, self.max_num_sents, self.max_num_words],
                                                  name="context")  # [batch_size, max_num_sents, max_num_words]
        self.query_placeholder = tf.placeholder(tf.int32, shape=[None, 5], name="query")  # [batch_size, 5]
        self.num_sents = tf.placeholder(tf.int32, shape=[None], name="num_sents")  # actual num_sents in per context
        self.labels = tf.placeholder(tf.int32, shape=[None], name="labels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def get_embeddings(self, pretrained_embs):
        embeddings = tf.get_variable(
            name="embeddings",
            shape=[self.vocab_size, self.emb_size],
            initializer=tf.constant_initializer(pretrained_embs),
            dtype=tf.float32
        )
        return embeddings

    def get_context_representation(self):
        context = tf.nn.embedding_lookup(self.embeddings,
                                         self.context_placeholder)  # tensor [batch_size, max_num_sents, max_num_words, emb_size]
        context = tf.reduce_sum(context * self.position_encoding, 2)  # tensor [batch_size, max_num_sents, emb_size]
        # rnn cell
        cell_fw = tf.nn.rnn_cell.GRUCell(self.hid_size)
        cell_bw = tf.nn.rnn_cell.GRUCell(self.hid_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=context,
            sequence_length=self.num_sents,
            dtype=tf.float32
        )

        outputs = tf.reduce_sum(tf.stack(outputs), axis=0)  # forward + backward

        facts = tf.nn.dropout(outputs, self.dropout_keep_prob)  # tensor [batch_size, max_num_sents, hidden_size]

        return facts

    def get_query_representation(self):
        query = tf.nn.embedding_lookup(self.embeddings, self.query_placeholder)
        cell_fw = tf.nn.rnn_cell.GRUCell(self.hid_size)
        cell_bw = tf.nn.rnn_cell.GRUCell(self.hid_size)
        _, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=query,
            dtype=tf.float32
        )
        output_states = tf.reduce_sum(tf.stack(output_states), axis=0)
        q = tf.nn.dropout(output_states, self.dropout_keep_prob)
        return q

    def get_attention(self, q, prev_memory, fact, reuse):
        with tf.variable_scope("attention", reuse=reuse):
            features = [
                fact * q,
                fact * prev_memory,
                tf.abs(fact - q),
                tf.abs(fact - prev_memory)
            ]
            features = tf.concat(features, axis=1)

            attention = tf.contrib.layers.fully_connected(
                features,
                self.emb_size,
                activation_fn=tf.nn.tanh,
                reuse=reuse,
                scope="fc1"
            )

            attention = tf.contrib.layers.fully_connected(
                attention,
                1,
                activation_fn=None,
                reuse=reuse,
                scope="fc2"
            )

        return attention

    def generate_episode(self, memory, q, facts, mask, hop_index):
        attentions = [tf.squeeze(
            self.get_attention(q, memory, fact, bool(hop_index) or bool(i)), axis=1) for i, fact in
            enumerate(tf.unstack(facts, axis=1))
        ]  # list [num_max_sents] * [batch_size,]
        attentions = tf.transpose(tf.stack(attentions)) + (mask * -1e9)  # tensor [batch_size, num_max_sents]
        attentions = tf.nn.softmax(attentions)
        self.attentions.append(attentions)  # for visualization
        attentions = tf.expand_dims(attentions, axis=-1)  # tensor [batch_size, num_max_sents, 1]

        reuse = True if hop_index > 0 else False

        gru_inputs = tf.concat([facts, attentions], 2)  # tensor [batch_size, num_max_sents, hid_size + 1]

        with tf.variable_scope("episode", reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(
                cell=AttentionGRUCell(self.hid_size),
                inputs=gru_inputs,
                sequence_length=self.num_sents,
                dtype=tf.float32
            )
        return episode

    def add_answer_module(self, memory, q):
        memory = tf.nn.dropout(memory, self.dropout_keep_prob)
        logits = tf.layers.dense(
            tf.concat([memory, q], axis=1),
            self.num_classes,
            activation=None
        )
        return logits

    def inference(self):
        # input module
        with tf.variable_scope("context", initializer=tf.contrib.layers.xavier_initializer()):
            facts = self.get_context_representation()
            mask = tf.cast(tf.sequence_mask(self.num_sents, self.max_num_sents), tf.float32)
            mask = tf.ones_like(mask, dtype=tf.float32) - mask

        with tf.variable_scope("query", initializer=tf.contrib.layers.xavier_initializer()):
            # print("==> get query representation")
            q = self.get_query_representation()

        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            # print("==> build episodic memory")

            prev_memory = q

            for i in range(self.num_hops):
                # print("==> generating episode", i)
                episode = self.generate_episode(prev_memory, q, facts, mask, i)

                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(
                        tf.concat([prev_memory, episode, q], axis=1),
                        self.hid_size,
                        activation=tf.nn.relu
                    )
            output = prev_memory

        # answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            logits = self.add_answer_module(output, q)

        return logits

    def get_loss(self, logits):
        labels = tf.one_hot(self.labels, depth=self.num_classes)
        losses = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2_reg_lambda
        return tf.reduce_mean(losses) + l2_reg

    def get_predictions_and_accuracy(self, logits):
        predictions = tf.argmax(logits, 1, name="predictions", output_type = tf.int32)
        correct_predictions = tf.equal(predictions, self.labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        return predictions, accuracy
