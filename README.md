# Text Classification
Tensorflow implementation of some text classification models.

### Dependencies

- Python 3.6
- Tensorflow == 1.14.0

### Usage

#### Datasets

- The datasets (imdb, yelp-13, yelp-14, yelp15) are collected and processed by [Duyu Tang](https://tangduyu.github.io/), and the resources can be downloaded [here](https://drive.google.com/open?id=1rASDy8v4QPq4ZNEZqIJo5dqcxAGINW8K). 
- The glove word embedding.

#### Training & Evaluating

To preprocess the dataset (make sure all the data are in correct directories)

```
python data_helpers.py
```

To train the model (please check all the hyper-parameters)

```
python train.py
```

To evaluate the trained model

```
python eval.py
```

 ### Models

[fastText](https://arxiv.org/abs/1607.01759), [CNN](https://arxiv.org/abs/1408.5882), [RNN w/o attention](https://arxiv.org/abs/1409.0473), [RCNN](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552), [GatedNN](https://www.aclweb.org/anthology/D15-1167), [HAN](https://www.aclweb.org/anthology/N16-1174), [DMN](https://arxiv.org/abs/1506.07285)
