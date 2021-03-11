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
tf.flags.DEFINE_string("checkpoint_dir", "runs/1563096406/checkpoints", "Checkpoint directory from training run")

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
        context_placeholder = graph.get_operation_by_name("context").outputs[0]
        query_placeholder = graph.get_operation_by_name("query").outputs[0]
        num_sents_placeholder = graph.get_operation_by_name("num_sents").outputs[0]
        labels_placeholder = graph.get_operation_by_name("labels").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]


        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("predictions").outputs[0]

        # Generate batches for one epoch
        all_labels = []
        all_predictions = []
        for batch in test.bacth_iter(FLAGS.batch_size, desc="Evaluating", shuffle=False):
            labels, contexts, queries = zip(*batch)
            contexts, num_sents = normalize(contexts)
            feed_dict = {
                context_placeholder: contexts,
                query_placeholder: queries,
                num_sents_placeholder: num_sents,
                labels_placeholder: labels,
                dropout_keep_prob: 1.0
            }
            batch_predictions = sess.run(predictions, feed_dict)
            all_labels = np.concatenate([all_labels, labels])
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy
if all_labels is not None:
    correct_predictions = float(sum(all_predictions == all_labels))
    print("Total number of test examples: {}".format(len(all_labels)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(all_labels)) * 100))


# Save the evaluation to a csv
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "predictions.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(map(lambda x: [x], all_predictions.astype(np.int32)))