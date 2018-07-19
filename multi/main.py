import utils
import tensorflow as tf
from tfrecord_generator import Generator
from pathlib import Path
from data_provider import DataProvider
from train_and_eval import TrainEval

class MultiTaskNet():

    def __init__(self):
        self.validation_csv = './data/train_set.csv'
        self.train_tfrecords = './data/train_set.tfrecords'
        self.batch_size = 2
        self.num_classes = 3
        self.learning_rate = 1e-4
        self.epochs = 1

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

    def start_process(self):
        self.get_data_provider()
        train_class = TrainEval(self.train_data_provider, self.epochs, self.batch_size)
        train_class.start_training()


def main():
    multi_task_net = MultiTaskNet()
    multi_task_net.start_process()
    #multi_task_net.generate_tfrecords()

if __name__ == '__main__':
    main()