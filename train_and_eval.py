import tensorflow as tf
import numpy as np

class TrainEval():

    def __init__(self, data_provider, predictions, batch_size, epochs, num_classes, sample_num, learning_rate):
        self.data_provider = data_provider
        self.predictions = predictions
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.sample_num = sample_num
        self.learning_rate = learning_rate

    def start_training(self):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3) # set random seed for initialization

            self.data_provider.get_batch()
            #iter_train = self.data_provider.dataset.make_initializable_iterator()
            iterator = tf.data.Iterator.from_structure(output_shapes=self.data_provider.dataset.output_shapes,
                                                       output_types=self.data_provider.dataset.output_types)
            frames, labels, subject_ids = iterator.get_next()

            labels = tf.one_hot(labels, depth=3, axis=-1)
            labels = tf.reshape(labels, (self.batch_size, self.num_classes))
            frames = tf.reshape(frames, (self.batch_size, -1, 640))

            iter_train = iterator.make_initializer(self.data_provider.dataset)

            prediction = self.predictions(frames)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
            cross_entropy_mean = tf.reduce_mean(loss, name='cross_entropy')
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy_mean)

        with tf.Session(graph=g) as sess:
            num_batches = int(np.ceil(self.sample_num / (self.batch_size)))
            sess.run(tf.global_variables_initializer())

            for _ in range(self.epochs):
                sess.run(iter_train)
                for _ in range(num_batches):
                    _, loss_value = sess.run([optimizer, cross_entropy_mean])
                    print("loss: ", loss_value)