import pickle
from tqdm import tqdm
import random
import numpy as np

class Dataset(object):
    def __init__(self, filepath, max_num_words=50, max_num_sents=20, num_classes=5):
        self.max_num_words = max_num_words
        self.max_num_sents = max_num_sents
        self.num_classes = num_classes
        self.dataset = self._load_data(filepath)

    def _load_data(self, filepath):
        print("loading dataset from %s" % filepath)
        dataset = []
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            # random.shuffle(data)
            for label, context, query in data:
                # cut-off in sent-level and word-level
                context = context[:self.max_num_sents]
                context = [sent[:self.max_num_words] for sent in context]
                dataset.append((label - 1, context, query))
        dataset = sorted(dataset, key=lambda x: len(x[1]))
        return dataset

    def _batch_iter(self, dataset, batch_size, desc=None):
        data_size = len(dataset)
        num_batches = int(np.ceil(data_size / batch_size))
        for batch in tqdm(range(num_batches), desc):
        # for batch in tqdm(range(num_batches), desc):
            start_index = batch * batch_size
            end_index = min((batch + 1) * batch_size, data_size)
            yield dataset[start_index: end_index]

    def bacth_iter(self, batch_size, desc=None, shuffle=False):
        if shuffle:
            random.shuffle(self.dataset)
        return self._batch_iter(self.dataset, batch_size, desc)
