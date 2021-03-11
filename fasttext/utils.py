import tensorflow as tf
import numpy as np
import pickle


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


def load_vocab(vocab_file):
    print('loading vocabulary ...')
    with open(vocab_file, 'rb') as f:
        word_dict = pickle.load(f)
        print('vocab size = %d' % len(word_dict))
        return word_dict


def vectorize(docs):
    sequence_lengths = [len(doc) for doc in docs]
    max_sequence_length = np.max(sequence_lengths)
    padded_docs = np.zeros(shape=[len(docs), max_sequence_length], dtype=np.int32)
    padded_docs_mask = np.zeros(shape=[len(docs), max_sequence_length], dtype=np.float32)
    for i, doc in enumerate(docs):
        padded_docs[i, :sequence_lengths[i]] = doc
        padded_docs_mask[i, :sequence_lengths[i]] = 1.0

    return padded_docs, padded_docs_mask, sequence_lengths, max_sequence_length


def load_glove(glove_file, embedding_size, vocab):
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

    embedding_matrix = np.random.uniform(-0.5, 0.5, (len(vocab), embedding_size)) / embedding_size

    oov_count = 0
    for word, i in vocab.items():
        embedding_vector = embedding_weights.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            oov_count += 1
    print('number of OOV words = %d' % oov_count)

    return embedding_matrix
