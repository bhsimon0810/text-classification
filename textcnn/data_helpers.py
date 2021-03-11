import pandas as pd
import nltk
import itertools
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

WORD_CUT_OFF = 100


def build_vocab(docs, save_path):  # save_path means the path of raw inputs; dst_path means the path for processed data
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
    word_dict = {'PAD': 0}
    for i, w in enumerate(vocab):
        word_dict[w] = i + 1  # because the first index allocated to symbol 'PAD'
    print("vocab size = %d" % len(word_dict))

    # save the vocab
    with open('./{}/word_dict.pkl'.format(save_path), 'wb') as f:
        pickle.dump(word_dict, f)
    print('vocab builded!\n')
    return word_dict


def transform(word_dict, dataset, save_path, flag='train'):
    """
    transform raw texts into sequences
    """
    print('transforming data to token sequences ...')
    lengths = []
    vocab = list(word_dict.keys())
    labels = dataset[4]
    transformed_docs = []
    docs = [doc.split('<sssss>') for doc in dataset[6]]
    # type(data) = dataframe, col'4' correspods to labels, col'6' corresponds to raw texts
    for label, doc in tqdm(zip(labels, docs), 'processing'):
        transformed_doc = [word_dict.get(word) for word in itertools.chain(*[sent.split() for sent in doc]) if word in vocab]
        if len(transformed_doc) > 3:
            transformed_docs.append((label, transformed_doc))
            lengths.append(len(transformed_doc))
    with open('./{}/{}.pkl'.format(save_path, flag), 'wb') as f:
        pickle.dump(transformed_docs, f)
    print('data transformed!\n')
    return transformed_docs, lengths


def load_data(file_path):
    print('loading data from ' + file_path)
    dataset = pd.read_csv(file_path, sep='\t', header=None, usecols=[4, 6])  # usecols depend on the format of raw data
    print('{}, shape={}\n'.format(file_path, dataset.shape))
    return dataset


if __name__ == '__main__':
    data = 'imdb' # or 'yelp-2013', 'yelp-2014', 'yelp-2015'
    dir = './data/'
    train_file = dir + data + '-train.txt.ss'
    dev_file = dir + data + '-dev.txt.ss'
    test_file = dir + data + '-test.txt.ss'

    train_dataset = load_data(train_file)
    word_dict = build_vocab(train_dataset[6], data)
    _, lengths = transform(word_dict, train_dataset, data, flag='train')

    dev_dataset = load_data(dev_file)
    _, _ = transform(word_dict, dev_dataset, data, flag='valid')

    test_dataset = load_data(test_file)
    _, _ = transform(word_dict, test_dataset, data, flag='test')

    print("maximum sequence length is %d." % max(lengths))
    print("minimum sequence length is %d." % min(lengths))

    plt.hist(lengths)
    plt.title('distribution of document length')
    plt.show()