import tensorflow as tf
import numpy as np
import pickle


def get_shape(tensor): # the type of param. tensor is tf.tensor
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0] for s in zip(static_shape, dynamic_shape)]
    return dims


def count_parameters(trained_vars):
    total_parameters = 0
    print('=' * 100)
    for variable in trained_vars:
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        print('{:70} {:20} params'.format(variable.name, variable_parameters))
        print('-' * 100)
        total_parameters += variable_parameters
    print('=' * 100)
    print("total trainable parameters: %d" % total_parameters)
    print('=' * 100)


def read_vocab(vocab_file):
    print('loading vocabulary ...')
    with open(vocab_file, 'rb') as f:
        word_to_index = pickle.load(f)
        print('vocab size = %d' % len(word_to_index))
        return word_to_index


def normalize(contexts):
    max_num_sents = 20
    max_num_words = 50
    num_sents = np.array([len(context) for context in contexts], dtype=np.int32)
    padded_contexts = np.zeros(shape=[len(contexts), max_num_sents, max_num_words], dtype=np.int32)  # PADDING 0

    for i, context in enumerate(contexts):
        for j, sent in enumerate(context):
            for k, word in enumerate(sent):
                padded_contexts[i, j, k] = word

    return padded_contexts, num_sents


def load_glove(glove_file, emb_size, vocab):
    print('loading glove pre-trained word embeddings ...')
    embedding_weights = {}
    f = open(glove_file, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_weights[word] = vector
    f.close()
    print('total {} word vectors in {}'.format(len(embedding_weights), glove_file))

    embedding_matrix = np.random.uniform(-0.5, 0.5, (len(vocab), emb_size)) / emb_size
    embedding_matrix = np.concatenate(
        (np.zeros((1, emb_size), dtype='float32'), embedding_matrix[1:, :]), 0
    )

    oov_count = 0
    for word, i in vocab.items():
        embedding_vector = embedding_weights.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            oov_count += 1
    print('number of OOV words = %d' % oov_count)

    return embedding_matrix
