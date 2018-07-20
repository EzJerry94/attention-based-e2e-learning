import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import recall_score

class TrainEval():

    def __init__(self, train_data_provider, epochs, batch_size, num_classes, validation_data_provider):
        self.train_data_provider = train_data_provider
        self.validation_data_provider = validation_data_provider
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_sample_num = 6409
        self.validation_sample_num = 1811
        self.num_classes = num_classes

    def start_training(self):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3)

            self.train_data_provider.get_batch()
            self.validation_data_provider.get_batch()

            iterator = tf.data.Iterator.from_structure(output_shapes=self.train_data_provider.dataset.output_shapes,
                                                       output_types=self.train_data_provider.dataset.output_types)

            files, arousals, valences, dominances, frames = iterator.get_next()

            arousals = tf.one_hot(arousals, depth=3, axis=-1)
            valences = tf.one_hot(valences, depth=3, axis=-1)
            dominances = tf.one_hot(dominances, depth=3, axis=-1)
            arousals = tf.reshape(arousals, (self.batch_size, self.num_classes))
            valences = tf.reshape(valences, (self.batch_size, self.num_classes))
            dominances = tf.reshape(dominances, (self.batch_size, self.num_classes))
            frames = tf.reshape(frames, (self.batch_size, -1, 640))

            iter_train = iterator.make_initializer(self.train_data_provider.dataset)

        with tf.Session(graph=g) as sess:
            train_num_batches = int(self.train_sample_num / (self.batch_size))
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.epochs):
                print('\n Start Training for epoch {}\n'.format(epoch + 1))
                sess.run(iter_train)

                for batch in range(train_num_batches):
                    f, a, v, d, frame= sess.run([files, arousals, valences, dominances, frames])
                    print(f)
                    print(a)
                    print(v)
                    print(d)
                    print(frame)
                    print('**********************************************************')