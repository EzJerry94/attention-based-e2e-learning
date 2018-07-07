import tensorflow as tf
slim = tf.contrib.slim


class DataProvider():

    def __init__(self, tfrecords_folder, batch_size, epochs):
        self.tfrecords_folder = tfrecords_folder
        self.batch_size = batch_size
        self.epochs = epochs
        self.paths = [str(x) for x in self.tfrecords_folder.glob('*.tfrecords')]
        self.sample_num = len(self.paths)

    def get_batch(self):
        dataset = tf.data.TFRecordDataset(self.paths)
        dataset = dataset.map(self.parse_example)
        padded_shapes = ([None], [1], [])
        self.dataset = dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
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

        frame = tf.decode_raw(frame, tf.float32)
        label = tf.decode_raw(label, tf.int32)

        return frame, label, subject_id