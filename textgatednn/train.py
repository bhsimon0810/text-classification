import tensorflow as tf
import numpy as np
import os
import time
import datetime
from dataset import Dataset
from model import HieNet
from utils import read_vocab, count_parameters, load_glove, normalize

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_data", "data/yelp-2013-train.pkl", "Data source for the training data.")
tf.flags.DEFINE_string("valid_data", "data/yelp-2013-dev.pkl", "Data source for the validating data.")
tf.flags.DEFINE_string("vocab_data", "data/yelp-2013-w2i.pkl", "Data source for the vocab data.")

# Model Hyperparameters
tf.flags.DEFINE_string("cell_type", "gru", "Type of rnn cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: gru)")
tf.flags.DEFINE_integer("emb_size", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hid_size", 128, "Dimensionality of rnn cell units (Default: 128)")
tf.flags.DEFINE_integer("num_classes", 5, "Number of classes (default: 5)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Maximum value of the global norm of the gradients for clipping (default: 5.0)")
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
    vocab = read_vocab(FLAGS.vocab_data)
    glove = load_glove("glove.6B.{}d.txt".format(FLAGS.emb_size), FLAGS.emb_size, vocab)
    train = Dataset(filepath=FLAGS.train_data)
    valid = Dataset(filepath=FLAGS.valid_data)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            hnn = HieNet(
                cell_type=FLAGS.cell_type,
                hid_size=FLAGS.hid_size,
                vocab_size=len(vocab),
                emb_size=FLAGS.emb_size,
                num_classes=FLAGS.num_classes,
                pretrained_embs=glove,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # Define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(hnn.loss, global_step=global_step)
            acc, acc_op = tf.metrics.accuracy(labels=hnn.labels, predictions=hnn.predictions, name="metrics/acc")
            metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
            metrics_init_op = tf.variables_initializer(var_list=metrics_vars)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", hnn.loss)
            acc_summary = tf.summary.scalar("accuracy", hnn.accuracy)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Valid summaries
            valid_step = 0
            valid_summary_op = tf.summary.merge([loss_summary, acc_summary])
            valid_summary_dir = os.path.join(out_dir, "summaries", "valid")
            valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # initialize all variables
            best_valid_acc = 0.0
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())


            # training and validating loop
            for epoch in range(FLAGS.num_epochs):
                print('-' * 100)
                print('\n{}> epoch: {}\n'.format(datetime.datetime.now().isoformat(), epoch))
                sess.run(metrics_init_op)
                # Training process
                for batch in train.bacth_iter(FLAGS.batch_size, desc="Training", shuffle=True):
                    labels, docs = zip(*batch)
                    padded_docs, sent_length, max_sent_length, word_length, max_word_length = normalize(docs)
                    feed_dict = {
                        hnn.docs: padded_docs,
                        hnn.labels: labels,
                        hnn.sent_length: sent_length,
                        hnn.word_length: word_length,
                        hnn.max_sent_length: max_sent_length,
                        hnn.max_word_length: max_word_length,
                        hnn.is_training: True,
                        hnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy, _ = sess.run(
                        [train_op, global_step, train_summary_op, hnn.loss, hnn.accuracy, acc_op], feed_dict
                    )
                    train_summary_writer.add_summary(summaries, step)

                print("\ntraining accuracy = {:.2f}\n".format(sess.run(acc) * 100))

                sess.run(metrics_init_op)
                # Validating process
                for batch in valid.bacth_iter(FLAGS.batch_size, desc="Validating", shuffle=False):
                    valid_step += 1
                    labels, docs = zip(*batch)
                    padded_docs, sent_length, max_sent_length, word_length, max_word_length = normalize(docs)
                    feed_dict = {
                        hnn.docs: padded_docs,
                        hnn.labels: labels,
                        hnn.sent_length: sent_length,
                        hnn.max_sent_length: max_sent_length,
                        hnn.word_length: word_length,
                        hnn.max_word_length: max_word_length,
                        hnn.is_training: False,
                        hnn.dropout_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, _ = sess.run(
                        [valid_summary_op, hnn.loss, hnn.accuracy, acc_op], feed_dict
                    )
                    valid_summary_writer.add_summary(summaries, global_step=valid_step)

                valid_acc = sess.run(acc) * 100
                print("\nvalidating accuracy = {:.2f}\n".format(valid_acc))
                print("previous best validating accuracy = {:.2f}\n".format(best_valid_acc))

                # model checkpoint
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    path = saver.save(sess, checkpoint_prefix)
                    print("saved model checkpoint to {}\n".format(path))

            print("{} optimization finished!\n".format(datetime.datetime.now()))
            print("best validating accuracy = {:.2f}\n".format(best_valid_acc))



def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()