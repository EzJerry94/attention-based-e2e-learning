import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import recall_score

slim = tf.contrib.slim

class TrainEval():

    def __init__(self, data_provider, predictions, batch_size, epochs, num_classes, sample_num, learning_rate,
                 eval_provider, eval_sample_num):
        self.data_provider = data_provider
        self.predictions = predictions
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.sample_num = sample_num
        self.learning_rate = learning_rate
        self.eval_provider = eval_provider
        self.eval_sample_num = eval_sample_num

    def start_training(self):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3) # set random seed for initialization

            self.data_provider.get_batch()
            self.eval_provider.get_batch()
            #iter_train = self.data_provider.dataset.make_initializable_iterator()
            iterator = tf.data.Iterator.from_structure(output_shapes=self.data_provider.dataset.output_shapes,
                                                       output_types=self.data_provider.dataset.output_types)
            frames, labels, subject_ids = iterator.get_next()

            labels = tf.one_hot(labels, depth=3, axis=-1)
            labels = tf.reshape(labels, (self.batch_size, self.num_classes))
            frames = tf.reshape(frames, (self.batch_size, -1, 640))

            iter_train = iterator.make_initializer(self.data_provider.dataset)
            iter_eval = iterator.make_initializer(self.eval_provider.dataset)

            prediction = self.predictions(frames)
            eval_prediction = self.predictions(frames)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
            #loss = tf.nn.weighted_cross_entropy_with_logits(labels, prediction,pos_weight=1)
            cross_entropy_mean = tf.reduce_mean(loss, name='cross_entropy')
            #cross_entropy_mean = slim.losses.compute_weighted_loss(loss)
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy_mean)

        with tf.Session(graph=g) as sess:
            train_num_batches = int(np.ceil(self.sample_num / (self.batch_size)))
            eval_num_batches = int(np.ceil(self.eval_sample_num / (self.batch_size)))
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
                sess.run(iter_eval)
                eval_predictions_list = []
                eval_labels_list = []
                for batch in range(eval_num_batches):
                    print('Example {}/{}'.format(batch + 1, eval_num_batches))
                    preds, labs, s = sess.run([eval_prediction, labels, subject_ids])
                    eval_predictions_list.append(preds)
                    eval_labels_list.append(labs)

                eval_predictions_list = np.reshape(eval_predictions_list, (-1, self.num_classes))
                eval_labels_list = np.reshape(eval_labels_list, (-1, self.num_classes))
                eval_predictions_list = np.argmax(eval_predictions_list, axis=1)
                eval_labels_list = np.argmax(eval_labels_list, axis=1)

                '''
                correct = 0
                for i in range(len(eval_predictions_list)):
                    if eval_predictions_list[i] == eval_labels_list[i]:
                        correct += 1
                accuracy = float(correct) / float(self.eval_sample_num)
                print("accuracy = ",accuracy)
                '''
                mean_eval = recall_score(eval_labels_list, eval_predictions_list, average="macro")
                print("uar: ", mean_eval)
