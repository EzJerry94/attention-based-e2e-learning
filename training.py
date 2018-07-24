import tensorflow as tf


class Train():

    def __init__(self, train_data_provider, validate_data_provider, batch_size, epochs, num_classes,
                 learning_rate):
        self.train_data_provider = train_data_provider
        self.validate_data_provider = validate_data_provider
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_sample_num = 4650
        self.validate_sample_num = 895

    def start_training(self):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3)  # set random seed for initialization

            self.train_data_provider.get_batch()
            self.validate_data_provider.get_batch()

            iterator = tf.data.Iterator.from_structure(output_shapes=self.train_data_provider.dataset.output_shapes,
                                                       output_types=self.train_data_provider.dataset.output_types)
            frames, labels= iterator.get_next()

            labels = tf.one_hot(labels, depth=3, axis=-1)
            labels = tf.reshape(labels, (self.batch_size, self.num_classes))
            frames = tf.reshape(frames, (self.batch_size, -1, 640))

            iter_train = iterator.make_initializer(self.train_data_provider.dataset)
            iter_eval = iterator.make_initializer(self.validate_data_provider.dataset)

        with tf.Session(graph=g) as sess:
            train_num_batches = int(self.train_sample_num / self.batch_size)
            validate_num_batches = int(self.validate_sample_num / self.batch_size)
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.epochs):
                print('\n Start Training for epoch {}\n'.format(epoch + 1))
                sess.run(iter_train)
                for batch in range(train_num_batches):
                    frames_out, labels_out = sess.run([frames, labels])
                    print(frames_out)
                    print(labels_out)
                    print("*************************************8")
