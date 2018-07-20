import utils
import tensorflow as tf
from tfrecord_generator import Generator
from pathlib import Path
from data_provider import DataProvider
from train_and_eval import TrainEval
from cnn_model import CNNModel
from rnn_model import RNNModel
from attention_model import AttentionModel
from fc_model import FCModel

class MultiTaskNet():

    def __init__(self):
        self.validation_csv = './data/train_set.csv'
        self.train_tfrecords = './data/train_set.tfrecords'
        self.validation_tfrecords = './data/validation_set.tfrecords'
        self.batch_size = 2
        self.num_classes = 3
        self.learning_rate = 1e-4
        self.epochs = 1

    def _reshape_to_conv(self, frames):
        frame_shape = frames.get_shape().as_list()
        num_featurs = frame_shape[-1]
        batch = -1
        frames = tf.reshape(frames, (batch, num_featurs))
        return frames

    def _reshape_to_rnn(self, frames):
        batch_size, num_features = frames.get_shape().as_list()
        seq_length = -1
        frames = tf.reshape(frames, [2, seq_length, num_features])
        return frames

    def read_stats(self):
        utils.preprocess_stats('./IEMOCAP_full_releaseA/test_set.txt','test_set.csv')
        utils.preprocess_stats('./IEMOCAP_full_releaseA/validation_set.txt', 'validation_set.csv')
        utils.preprocess_stats('./IEMOCAP_full_releaseA/train_set.txt', 'train_set.csv')

    def show_wav(self):
        utils.show_wav('./IEMOCAP_full_releaseA/Session3/sentences/wav/Ses03M_script02_2/Ses03M_script02_2_F000.wav')

    def generate_tfrecords(self):
        generator = Generator(self.validation_csv)
        generator.write_tfrecords()

    def get_data_provider(self):
        self.train_data_provider = DataProvider(self.train_tfrecords, self.batch_size, self.epochs)
        self.validation_data_provider = DataProvider(self.validation_tfrecords, self.batch_size, self.epochs)

    def get_model(self, frames):
        frames = self._reshape_to_conv(frames)
        cnn = CNNModel()
        cnn_output = cnn.create_model_2(frames, 40, True)
        cnn_output = self._reshape_to_rnn(cnn_output)
        rnn = RNNModel()
        rnn_output = rnn.create_model(cnn_output)
        rnn_output = rnn_output[:, -1, :]
        #attention = AttentionModel(self.batch_size)
        #attention_output = attention.create_model(rnn_output)
        fc = FCModel(self.num_classes)
        outputs = fc.create_model(rnn_output)
        return outputs

    def start_process(self):
        predictions = self.get_model
        self.get_data_provider()
        train_class = TrainEval(self.train_data_provider, self.epochs, self.batch_size, self.num_classes,
                                self.validation_data_provider, predictions, self.learning_rate)
        train_class.start_training()


def main():
    multi_task_net = MultiTaskNet()
    multi_task_net.start_process()

if __name__ == '__main__':
    main()