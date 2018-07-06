import tensorflow as tf
from pathlib import Path
from data_provider import DataProvider


class AttentionNet:

    def __init__(self):
        self.tfrecords_folder = Path('./data_folder/sample_tfrecords')
        self.batch_size = 2
        self.epochs = 2

    def get_data_provider(self):
        self.data_provider = DataProvider(self.tfrecords_folder, self.batch_size, self.epochs)
        self.sample_num = self.data_provider.sample_num

    def read_data(self):
        self.data_provider.get_batch()
        iter_train = self.data_provider.dataset.make_initializable_iterator()
        frame, label, subject_id = iter_train.get_next()
        with tf.Session() as sess:
            sess.run(iter_train.initializer)
            for _ in range(self.epochs):
                for _ in range(self.sample_num // self.batch_size):
                    out_frame, out_label, out_subject_id = sess.run([frame, label, subject_id])
                    print(out_label, out_subject_id)
                    print(out_frame.shape, out_label.shape)
                print("**********************")


def main():
    attention_net = AttentionNet()
    attention_net.get_data_provider()
    attention_net.read_data()

if __name__ == '__main__':
    main()
