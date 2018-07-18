#import tensorflow as tf
import utils
from tfrecord_generator import Generator

class MultiTaskNet():

    def __init__(self):
        pass

    def read_stats(self):
        utils.preprocess_stats('./IEMOCAP_full_releaseA/test_set.txt','test_set.csv')
        utils.preprocess_stats('./IEMOCAP_full_releaseA/validation_set.txt', 'validation_set.csv')
        utils.preprocess_stats('./IEMOCAP_full_releaseA/train_set.txt', 'train_set.csv')

    def show_wav(self):
        utils.show_wav('./IEMOCAP_full_releaseA/Session3/sentences/wav/Ses03M_script02_2/Ses03M_script02_2_F000.wav')

    def generate_tfrecords(self):
        generator = Generator(self.validation_csv)
        generator.write_tfrecords()

def main():
    multi_task_net = MultiTaskNet()

if __name__ == '__main__':
    main()