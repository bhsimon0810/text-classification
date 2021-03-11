import pickle
from tqdm import tqdm
import random
import numpy as np


class Dataset(object):
    def __init__(self, filepath, max_word_length=50, max_sent_length=30, num_classes=5):
        self.max_word_length = max_word_length
        self.max_sent_length = max_sent_length
        self.num_classes = num_classes
        self.dataset = self._load_data(filepath)

    def _load_data(self, filepath):
        print("loading dataset from %s" % filepath)
        dataset = []
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            # random.shuffle(data)
            for label, doc in data:
                # cut-off in sent-level and word-level
                doc = doc[:self.max_sent_length]
                doc = [sent[:self.max_word_length] for sent in doc]
                dataset.append((label - 1, doc))
        dataset = sorted(dataset, key=lambda x: len(x[1]))
        return dataset

    def _iter(self, dataset, batch_size, desc=None):
        data_size = len(dataset)
        num_batch = int(np.ceil(data_size / batch_size))
        for batch in tqdm(range(num_batch), desc):
            start_index = batch * batch_size
            end_index = min((batch + 1) * batch_size, data_size)
            yield dataset[start_index: end_index]

    def bacth_iter(self, batch_size, desc=None, shuffle=False):
        if shuffle:
            random.shuffle(self.dataset)
        return self._iter(self.dataset, batch_size, desc)


