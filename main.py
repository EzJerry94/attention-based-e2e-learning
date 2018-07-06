import tensorflow as tf
from pathlib import Path
from data_provider import DataProvider
from train_and_eval import TrainEval


class AttentionNet:

    def __init__(self):
        self.tfrecords_folder = Path('./data_folder/sample_tfrecords')
        self.batch_size = 2
        self.epochs = 2
        self.num_classes = 3

    def get_data_provider(self):
        self.data_provider = DataProvider(self.tfrecords_folder, self.batch_size, self.epochs)
        self.sample_num = self.data_provider.sample_num

    def read_data(self):
        self.data_provider.get_batch()
        iter_train = self.data_provider.dataset.make_initializable_iterator()
        frames, labels, subject_ids = iter_train.get_next()
        with tf.Session() as sess:
            sess.run(iter_train.initializer)
            for _ in range(self.epochs):
                for _ in range(self.sample_num // self.batch_size):
                    out_frame, out_label, out_subject_id = sess.run([frames, labels, subject_ids])
                    print(out_label, out_subject_id)
                    print(out_frame.shape, out_label.shape)
                print("**********************")

    def get_model(self):
        pass

    def start_process(self):
        prediction = self.get_model()
        self.get_data_provider()
        train_class = TrainEval(self.data_provider, prediction, self.batch_size, self.epochs,
                                self.num_classes, self.sample_num)
        train_class.start_training()


def main():
    attention_net = AttentionNet()
    attention_net.start_process()
    #attention_net.get_data_provider()
    #attention_net.read_data()

if __name__ == '__main__':
    main()
