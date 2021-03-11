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


def normalize(docs):
    # sent_length is a list, containing every doc's length(number of sents)
    sent_length = np.array([len(doc) for doc in docs], dtype=np.int32)
    max_sent_length = sent_length.max()

    # word_length is a list, containing every sent's length(number of words)
    # every doc may contain diferent number of sents, which are have diffetent number of words
    # assumed that docs contain 3 docs, and 2, 3, 4 sents for each doc
    # doc1: 2 sents, 20 words and 25 words each
    # doc2: 3 sents, 18 words, 15 words and 26 words each
    # doc4: 4 sents, 10 words, 12 words, 19 words and 21 words each
    # thus words_length below will be [[20, 25], [18, 15, 26],[10, 12, 19, 21]]
    # considering each element in word_length has different shape, we should use map function to get max_word_length
    word_length = [[len(sent) for sent in doc] for doc in docs]
    max_word_length = max(map(max, word_length))

    # considering there may be several words padded in each sent, the way we get word_length is different from sent_length
    padded_docs = np.zeros(shape=[len(docs), max_sent_length, max_word_length], dtype=np.int32)  # PADDING 0
    word_length = np.zeros(shape=[len(docs), max_sent_length], dtype=np.int32)
    for i, doc in enumerate(docs):
        for j, sent in enumerate(doc):
            # word_length[i,j] means the j-th sent in the i-th doc has # words
            word_length[i, j] = len(sent)
            for k, word in enumerate(sent):
                padded_docs[i, j, k] = word

    return padded_docs, sent_length, max_sent_length, word_length, max_word_length


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

    oov_count = 0
    for word, i in vocab.items():
        embedding_vector = embedding_weights.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            oov_count += 1
    print('number of OOV words = %d' % oov_count)

    return embedding_matrix
