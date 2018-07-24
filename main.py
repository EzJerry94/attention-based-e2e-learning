from tfrecord_generator import Generator

class AttentionNet:

    def __init__(self):
        self.operation = 'generate'

    def tfrecords_generate(self):
        generator = Generator()
        generator.write_tfrecords()

def main():
    net = AttentionNet()
    if net.operation == 'generate':
        net.tfrecords_generate()

if __name__ == '__main__':
    main()