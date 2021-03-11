import tensorflow as tf
import numpy as np
import os
import time
import datetime
from dataset import Dataset
from model import FastText
from utils import load_vocab, load_glove, vectorize

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_data", "../yelp-2013/train.pkl", "Data source for the training data.")
tf.flags.DEFINE_string("valid_data", "../yelp-2013/valid.pkl", "Data source for the validating data.")
tf.flags.DEFINE_string("vocab_data", "../yelp-2013/word_dict.pkl", "Data source for the vocab data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("num_class", 5, "Number of classes (default: 5)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularization lambda (default: 0.2)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epoch", 50, "Number of training epochs (Default: 50)")
tf.flags.DEFINE_float("max_grad_norm", 10.0, "Maximum value of the global norm of the gradients for clipping (default: 10.0)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def train():
    word_dict = load_vocab(FLAGS.vocab_data)
    glove = load_glove("../glove.6B.{}d.txt".format(FLAGS.embedding_size), FLAGS.embedding_size, word_dict)
    train = Dataset(filepath=FLAGS.train_data)
    valid = Dataset(filepath=FLAGS.valid_data)

    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            fasttext = FastText(
                vocab_size=len(word_dict),
                embedding_size=FLAGS.embedding_size,
                num_class=FLAGS.num_class,
                pretrained_embeddings=glove,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # Define training procedure
            global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)
            train_op = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate).minimize(fasttext.loss, global_step=global_step)
            acc, acc_op = tf.compat.v1.metrics.accuracy(labels=fasttext.labels, predictions=fasttext.predictions, name="metrics/acc")
            metrics_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope="metrics")
            metrics_init_op = tf.compat.v1.variables_initializer(var_list=metrics_vars)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.compat.v1.summary.scalar("loss", fasttext.loss)
            acc_summary = tf.compat.v1.summary.scalar("accuracy", fasttext.accuracy)

            # Train summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Valid summaries
            valid_step = 0
            valid_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary])
            valid_summary_dir = os.path.join(out_dir, "summaries", "valid")
            valid_summary_writer = tf.compat.v1.summary.FileWriter(valid_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # initialize all variables
            best_valid_acc = 0.0
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            # training and validating loop
            for epoch in range(FLAGS.num_epoch):
                print('-' * 100)
                print('\n{}> epoch: {}\n'.format(datetime.datetime.now().isoformat(), epoch))
                sess.run(metrics_init_op)
                # Training process
                for batch in train.bacth_iter(FLAGS.batch_size, desc="Training", shuffle=True):
                    labels, docs = zip(*batch)
                    padded_docs, _, _, _ = vectorize(docs)
                    feed_dict = {
                        fasttext.inputs: padded_docs,
                        # fasttext.inputs_mask: padded_docs_mask,
                        fasttext.labels: labels,
                        fasttext.is_training: True,
                        fasttext.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy, _ = sess.run(
                        [train_op, global_step, train_summary_op, fasttext.loss, fasttext.accuracy, acc_op], feed_dict
                    )
                    train_summary_writer.add_summary(summaries, step)

                print("\ntraining accuracy = {:.2f}\n".format(sess.run(acc) * 100))

                sess.run(metrics_init_op)
                # Validating process
                for batch in valid.bacth_iter(FLAGS.batch_size, desc="Validating", shuffle=False):
                    valid_step += 1
                    labels, docs = zip(*batch)
                    padded_docs, _, _, _ = vectorize(docs)
                    feed_dict = {
                        fasttext.inputs: padded_docs,
                        # fasttext.inputs_mask: padded_docs_mask,
                        fasttext.labels: labels,
                        fasttext.is_training: False,
                        fasttext.dropout_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, _ = sess.run(
                        [valid_summary_op, fasttext.loss, fasttext.accuracy, acc_op], feed_dict
                    )
                    valid_summary_writer.add_summary(summaries, global_step=valid_step)

                valid_acc = sess.run(acc) * 100
                print("\nvalidating accuracy = {:.2f}\n".format(valid_acc))

                # model checkpoint
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    print("current best validating accuracy = {:.2f}\n".format(best_valid_acc))
                    path = saver.save(sess, checkpoint_prefix)
                    print("saved model checkpoint to {}\n".format(path))

            print("{} optimization finished!\n".format(datetime.datetime.now()))
            print("best validating accuracy = {:.2f}\n".format(best_valid_acc))


def main(_):
    train()


if __name__ == "__main__":
    tf.compat.v1.app.run()