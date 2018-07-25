import tensorflow as tf
from tfrecord_generator import Generator
from training import Train
from data_provider import DataProvider
from models.cnn import CNN
from models.rnn import RNN
from models.fc import FC
from models.attention import Attention
from evaluation import Evaluation

class AttentionNet:

    def __init__(self):
        self.operation = 'evaluation'
        self.train_tfrecords_folder = './data/train_set.tfrecords'
        self.validate_tfrecords_folder = './data/devel_set.tfrecords'
        self.batch_size = 2
        self.epochs = 2
        self.num_classes = 3
        self.learning_rate = 1e-4
        self.is_attention = True

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

    def tfrecords_generate(self):
        generator = Generator()
        generator.write_tfrecords()

    def get_data_provider(self):
        self.train_data_provider = DataProvider(self.train_tfrecords_folder, self.batch_size, True)
        self.validate_data_provider = DataProvider(self.validate_tfrecords_folder, self.batch_size, False)

    def training(self):
        predictions = self.get_predictions
        train = Train(self.train_data_provider, self.batch_size, self.epochs,
                      self.num_classes, self.learning_rate, predictions)
        train.start_training()

    def get_predictions(self, frames):
        frames = self._reshape_to_conv(frames)
        cnn = CNN()
        cnn_output = cnn.create_model(frames, cnn.conv_filters)
        cnn_output = self._reshape_to_rnn(cnn_output)
        rnn = RNN()
        rnn_output = rnn.create_model(cnn_output)
        if self.is_attention:
            attention = Attention(self.batch_size)
            attention_output = attention.create_model(rnn_output)
            fc = FC(self.num_classes)
            outputs = fc.create_model(attention_output)
        else:
            rnn_output = rnn_output[:, -1, :]
            fc = FC(self.num_classes)
            outputs = fc.create_model(rnn_output)
        return outputs

    def evaluation(self):
        predictions = self.get_predictions
        eval = Evaluation(self.validate_data_provider, self.batch_size, self.epochs,
                          self.num_classes, self.learning_rate, predictions)
        eval.start_evaluation()


def main():
    net = AttentionNet()
    if net.operation == 'generate':
        net.tfrecords_generate()
    elif net.operation == 'training':
        net.get_data_provider()
        net.training()
    elif net.operation == 'evaluation':
        net.get_data_provider()
        net.evaluation()

if __name__ == '__main__':
    main()