import pickle
from tqdm import tqdm
import random
import numpy as np


class Dataset(object):
    def __init__(self, filepath, num_class=5):
        self.num_class = num_class
        self.dataset = self._load_data(filepath)

    def _load_data(self, filepath):
        print("loading dataset from %s" % filepath)
        dataset = []
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            # random.shuffle(data)
            for label, doc in data:
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


