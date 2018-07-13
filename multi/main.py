#import tensorflow as tf
import utils

class MultiTaskNet():

    def __init__(self):
        pass

    def read_stats(self):
        utils.preprocess_stats('./IEMOCAP_full_releaseA/test_set.txt','test_set.csv')
        utils.preprocess_stats('./IEMOCAP_full_releaseA/validation_set.txt', 'validation_set.csv')
        utils.preprocess_stats('./IEMOCAP_full_releaseA/train_set.txt', 'train_set.csv')

def main():
    multi_task_net = MultiTaskNet()
    multi_task_net.read_stats()

if __name__ == '__main__':
    main()