import tensorflow as tf


def attention(inputs, att_size):
    """
    Attention mechanism layer.
    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param att_size: linear size of attention weights
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    # if isinstance(inputs, tuple):
    #     inputs = tf.concat(inputs, axis=2)

    # sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer

    hid_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

    # Attention mechanism
    w_omega = tf.Variable(tf.random_normal([hid_size, att_size], stddev=0.1), name="w_omega")
    b_omega = tf.Variable(tf.random_normal([att_size], stddev=0.1), name="b_omega")
    u_omega = tf.Variable(tf.random_normal([att_size], stddev=0.1), name="u_omega")


    v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega, name="v")
    vu = tf.tensordot(v, u_omega, axes=1, name="vu")
    alphas = tf.nn.softmax(vu, name="alphas")

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output, alphas