import tensorflow as tf

class TrainEval():

    def __init__(self, data_provider, predictions, batch_size, epochs, num_classes, sample_num):
        self.data_provider = data_provider
        self.predictions = predictions
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.sample_num = sample_num

    def start_training(self):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3) # set random seed for initialization

            self.data_provider.get_batch()
            iter_train = self.data_provider.dataset.make_initializable_iterator()
            frames, labels, subject_ids = iter_train.get_next()
            labels = tf.one_hot(labels, depth=3, axis=-1)
            labels = tf.reshape(labels, (self.batch_size, self.num_classes))
            frames = tf.reshape(frames, (self.batch_size, -1, 640))

            prediction = self.predictions(frames)

        with tf.Session(graph=g) as sess:
            sess.run(iter_train.initializer)
            for _ in range(self.epochs):
                for _ in range(self.sample_num // self.batch_size):
                    out_frame, out_label, out_subject_id = sess.run([frames, labels, subject_ids])
                    print(out_label, out_subject_id)
                    print(out_frame.shape, out_label.shape)
                print("**********************")