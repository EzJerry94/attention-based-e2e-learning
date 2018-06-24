import tensorflow as tf

from data_provider import DataProvider
from train_and_eval import TrainEval
from audio_model import AudioModel
from rnn_model import RNNModel
from fc_model import fully_connected
from attention_model import AttentionModel


class AttentionNet:
    def __init__(self):
        pass

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

    def get_data_provider(self):
        self.data_provider = DataProvider()

    def get_model(self, frames):
        frames = self._reshape_to_conv(frames)
        audio = AudioModel(is_training=True).create_model(frames)
        output_model = self._reshape_to_rnn(audio)
        rnn = RNNModel().create_model(output_model)
        #rnn = rnn[:, -1, :]
        attention = AttentionModel().create_model(rnn)
        num_outputs = self.data_provider.num_classes
        outputs = fully_connected(attention, num_outputs)
        return outputs

    def start_process(self):
        predictions = self.get_model
        self.get_data_provider()
        train_class = TrainEval(self.data_provider, predictions)
        train_class.start_training()

def main():
    attention_net = AttentionNet()
    attention_net.start_process()

if __name__ == '__main__':
    main()