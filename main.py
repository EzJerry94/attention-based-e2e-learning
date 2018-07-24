from tfrecord_generator import Generator
from training import Train
from data_provider import DataProvider

class AttentionNet:

    def __init__(self):
        self.operation = 'training'
        self.train_tfrecords_folder = './data/train_set.tfrecords'
        self.validate_tfrecords_folder = './data/devel_set.tfrecords'
        self.batch_size = 2
        self.epochs = 2
        self.num_classes = 3
        self.learning_rate = 1e-4

    def tfrecords_generate(self):
        generator = Generator()
        generator.write_tfrecords()

    def get_data_provider(self):
        self.train_data_provider = DataProvider(self.train_tfrecords_folder, self.batch_size)
        self.validate_data_provider = DataProvider(self.validate_tfrecords_folder, self.batch_size)

    def training(self):
        train = Train(self.train_data_provider, self.validate_data_provider, self.batch_size, self.epochs,
                      self.num_classes, self.learning_rate)
        train.start_training()

def main():
    net = AttentionNet()
    if net.operation == 'generate':
        net.tfrecords_generate()
    elif net.operation == 'training':
        net.get_data_provider()
        net.training()

if __name__ == '__main__':
    main()