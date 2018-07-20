import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import recall_score

class TrainEval():

    def __init__(self, train_data_provider, epochs, batch_size, num_classes, validation_data_provider,
                 predictions, learning_rate):
        self.train_data_provider = train_data_provider
        self.validation_data_provider = validation_data_provider
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_sample_num = 6409
        self.validation_sample_num = 1811
        self.num_classes = num_classes
        self.predictions = predictions
        self.learning_rate = learning_rate

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
            iter_validation = iterator.make_initializer(self.validation_data_provider.dataset)

            train_prediction = self.predictions(frames)
            validation_prediction = self.predictions(frames)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=train_prediction, labels=arousals)
            cross_entropy_mean = tf.reduce_mean(loss, name='cross_entropy')
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy_mean)

        with tf.Session(graph=g) as sess:
            train_num_batches = int(self.train_sample_num / self.batch_size)
            validation_num_batches = int(self.validation_sample_num / self.batch_size)
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.epochs):
                print('\n Start Training for epoch {}\n'.format(epoch + 1))
                sess.run(iter_train)
                for batch in range(train_num_batches):
                    start_time = time.time()
                    _, loss_value = sess.run([optimizer, cross_entropy_mean])
                    time_step = time.time() - start_time
                    print("Epoch {}/{}: Batch {}/{}: loss = {:.4f} ({:.2f} sec/step)".format(
                        epoch + 1, self.epochs, batch + 1, train_num_batches, loss_value, time_step))

                print('\n Start Validation for epoch {}\n'.format(epoch + 1))

                sess.run(iter_validation)
                eval_predictions_list = []
                eval_labels_list = []
                for batch in range(validation_num_batches):
                    print('Example {}/{}'.format(batch + 1, validation_num_batches))
                    preds, labs= sess.run([validation_prediction, arousals])
                    eval_predictions_list.append(preds)
                    eval_labels_list.append(labs)

                eval_predictions_list = np.reshape(eval_predictions_list, (-1, self.num_classes))
                eval_labels_list = np.reshape(eval_labels_list, (-1, self.num_classes))
                eval_predictions_list = np.argmax(eval_predictions_list, axis=1)
                eval_labels_list = np.argmax(eval_labels_list, axis=1)

                mean_eval = recall_score(eval_labels_list, eval_predictions_list, average="macro")
                print("uar: ", mean_eval)