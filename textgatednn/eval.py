import tensorflow as tf
import numpy as np
import os
import csv
import time
import datetime
from dataset import Dataset
from utils import normalize

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("test_data", "data/yelp-2013-test.pkl", "Data source for the testing data.")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1553054379/checkpoints", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

print("Evaluating...\n")
# Load test data
test = Dataset(filepath=FLAGS.test_data)
# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        docs = graph.get_operation_by_name("docs").outputs[0]
        sent_length = graph.get_operation_by_name("sent_length").outputs[0]
        max_sent_length = graph.get_operation_by_name("max_sent_length").outputs[0]
        word_length = graph.get_operation_by_name("word_length").outputs[0]
        max_word_length = graph.get_operation_by_name("max_word_length").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        is_training = graph.get_operation_by_name("is_training").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        all_labels = []
        all_predictions = []
        for batch in test.bacth_iter(FLAGS.batch_size, desc="Evaluating", shuffle=False):
            batch_labels, batch_docs = zip(*batch)
            batch_padded_docs, batch_sent_length, batch_max_sent_length, batch_word_length, batch_max_word_length = normalize(batch_docs)
            feed_dict = {
                docs: batch_padded_docs,
                sent_length: batch_sent_length,
                max_sent_length: batch_max_sent_length,
                word_length: batch_word_length,
                max_word_length: batch_max_word_length,
                is_training: False,
                dropout_keep_prob: 1.0
            }
            batch_predictions = sess.run(predictions, feed_dict)
            all_labels = np.concatenate([all_labels, batch_labels])
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy
if all_labels is not None:
    correct_predictions = float(sum(all_predictions == all_labels))
    print("Total number of test examples: {}".format(len(all_labels)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(all_labels))))


# Save the evaluation to a csv
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "predictions.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(map(lambda x: [x], all_predictions.astype(np.int32)))