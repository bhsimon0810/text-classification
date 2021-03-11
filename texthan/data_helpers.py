import pandas as pd
import nltk
import itertools
import pickle

WORD_CUT_OFF = 5


def build_vocab(docs, dst_path):  # src_path means the path of raw inputs; dst_path means the path for processed data
    print("building vocab ...")
    sents = itertools.chain(*[doc.split('<sssss>') for doc in docs])
    segmented_sents = [sent.split() for sent in sents]

    # count the word freq.
    vocab_dict = nltk.FreqDist(itertools.chain(*segmented_sents))
    print("%d unique words found" % len(vocab_dict.items()))

    # cut-off
    vocab = [w for (w, f) in vocab_dict.items() if f > WORD_CUT_OFF]
    print("%d words contained" % len(vocab))

    # build w2i and i2w
    word_to_index = {'PAD': 0, 'UNK': 1}
    for i, w in enumerate(vocab):
        word_to_index[w] = i + 2  # because the first two indexes allocated to symbol 'PAD' and 'UNK'
    index_to_word = {i: w for (w, i) in word_to_index.items()}
    print("vocab size = %d" % len(word_to_index))

    # save the vocab
    with open('{}-w2i.pkl'.format(dst_path), 'wb') as f:
        pickle.dump(word_to_index, f)
    with open('{}-i2w.pkl'.format(dst_path), 'wb') as f:
        pickle.dump(index_to_word, f)

    return word_to_index


def transform(w2i, data, dst_path):
    """
    transform raw texts into sequences
    """
    transformed_docs = []
    # type(data) = dataframe, col'4' correspods to labels, col'6' corresponds to raw texts
    for label, doc in zip(data[4], data[6]):
        transformed_doc = [[w2i.get(word, 1) for word in sent] for sent in doc]
        transformed_docs.append((label, transformed_doc))
    with open('{}.pkl'.format(dst_path), 'wb') as f:
        pickle.dump(transformed_docs, f)


def load_data(src_path):
    data = pd.read_csv(src_path, sep='\t', header=None, usecols=[4, 6])  # usecols depend on the format of raw data
    print('{}, shape={}'.format(src_path, data.shape))
    return data


if __name__ == '__main__':
    train_data = load_data('data/yelp-2013-train.txt.ss')
    word_to_index = build_vocab(train_data[6], 'data/yelp-2013')
    transform(word_to_index, train_data, 'data/yelp-2013-train.pkl')

    dev_data = load_data('data/yelp-2013-dev.txt.ss')
    transform(word_to_index, dev_data, 'data/yelp-2013-dev.pkl')

    test_data = load_data('data/yelp-2013-test.txt.ss')
    transform(word_to_index, test_data, 'data/yelp-2013-test.pkl')