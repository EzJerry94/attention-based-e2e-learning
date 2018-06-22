import copy
import tensorflow as tf
import numpy as np

from pathlib import Path
from sklearn.metrics import recall_score

class EvalOnce():
    def __init__(self):
        pass

    @staticmethod
    def get_eval_tensors(sess, predictions, data_provider, evalute_path):
        dp_eval = copy.copy(data_provider)
        paths = [str(x) for x in Path(evalute_path).glob('*.tfrecords')]
        filename_queue = tf.train.string_input_producer(paths, shuffle=False)
        dp_eval.num_examples = len(paths)
        _, dp_eval.serialized_example = tf.TFRecordReader().read(filename_queue)
        frames, labels, sids = dp_eval.get_batch()
        get_pred = predictions(frames)
        seq_length = 1 if data_provider.seq_length == 0 \
            else data_provider.seq_length
        num_batches = int(np.ceil(dp_eval.num_examples / (dp_eval.batch_size * seq_length)))
        return get_pred, labels, sids, num_batches

    @staticmethod
    def eval_once(sess, get_pred, labels, sids, num_batches, num_outpus, metric_name):
        # metric = EvalOnce.get_metric(metric_name)

        print('\n Start Evaluation \n')
        evaluated_predictions = []
        evaluated_labels = []
        for batch in range(num_batches):
            print('Example {}/{}'.format(batch+1, num_batches))
            preds, labs, s = sess.run([get_pred, labels, sids])
            evaluated_predictions.append(preds)
            evaluated_labels.append(labs)

        predictions = np.reshape(evaluated_predictions, (-1, num_outpus))
        labels = np.reshape(evaluated_labels, (-1, num_outpus))
        # np_uar
        labels = np.argmax(labels, axis=1)
        predictions = np.argmax(predictions, axis=1)
        mean_eval = recall_score(labels, predictions, average="macro")
        return mean_eval
