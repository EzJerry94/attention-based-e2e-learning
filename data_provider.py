import tensorflow as tf
from pathlib import Path

slim = tf.contrib.slim

class DataProvider():
    def __init__(self):
        self.tfrecords_folder = Path('./data_folder/train_tfrecords')
        paths = [str(x) for x in self.tfrecords_folder.glob('*.tfrecords')]
        self.num_examples = len(paths)
        self.num_classes = 3
        #self.frame_shape = [29, 640]
        #self.label_shape = [1]
        self.seq_length = 0
        self.batch_size = 2
        self.is_training = True
        self.noise = None

        filename_queue = tf.train.string_input_producer(paths, shuffle=False)
        reader = tf.TFRecordReader()
        _, self.serialized_example = reader.read(filename_queue)

    def parse_and_decode_example(self):
        features = tf.parse_single_example(
            self.serialized_example,
            features={
                'sample_id': tf.FixedLenFeature([], tf.int64),
                'subject_id': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'frame': tf.FixedLenFeature([], tf.string),
            }
        )

        frame = features['frame']
        label = features['label']
        subject_id = features['subject_id']

        if self.is_training:
            frame, label, subject_id = tf.train.batch(
                [frame, label, subject_id], batch_size=1, num_threads=1, capacity=1000)
            frame = frame[0]
            label = label[0]

        frame = tf.decode_raw(frame, tf.float32)
        label = tf.decode_raw(label, tf.int32)

        return frame, label, subject_id

    def get_single_example_batch(self, batch_size, *args):
        args = tf.train.batch(args, batch_size, capacity=1000, dynamic_pad=True, num_threads=1)
        return args

    def get_batch(self):
        frame, label, subject_id = self.parse_and_decode_example()
        frames, labels, subjects_id = \
            self.get_single_example_batch(self.batch_size, frame, label, subject_id)
        labels = slim.one_hot_encoding(labels, self.num_classes)
        labels = tf.reshape(labels, (self.batch_size, self.num_classes))
        frames = tf.reshape(frames, (self.batch_size, -1, 640))
        return frames, labels, subjects_id