import tensorflow as tf


class DataProvider():

    def __init__(self, tfrecords_path, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.tfrecords_path = tfrecords_path

    def get_batch(self):
        dataset = tf.data.TFRecordDataset(self.tfrecords_path)
        dataset = dataset.map(self.parse_example)
        padded_shapes = ([], [1], [1], [1], [None])
        self.dataset = dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'file': tf.FixedLenFeature([], tf.string),
                'arousal': tf.FixedLenFeature([], tf.string),
                'valence': tf.FixedLenFeature([], tf.string),
                'dominance': tf.FixedLenFeature([], tf.string),
                'frame': tf.FixedLenFeature([], tf.string),
            }
        )

        file = features['file']
        arousal = features['arousal']
        valence = features['valence']
        dominance = features['dominance']
        frame = features['frame']

        arousal = tf.decode_raw(arousal, tf.int32)
        valence = tf.decode_raw(valence, tf.int32)
        dominance = tf.decode_raw(dominance, tf.int32)
        frame = tf.decode_raw(frame, tf.float32)

        return file, arousal, valence, dominance, frame