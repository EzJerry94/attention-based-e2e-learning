import numpy as np
class Generator():

    def __init__(self, csv):
        self.attributes_name = ['file', 'arousal', 'valence', 'dominance']
        self.attributes_typt = ['str', 'int', 'int']
        self.csv = csv

    def read_csv(self, file):
        lines = np.loadtxt(file, dtype='str')
        print(lines)
        pass

    def write_tfrecords(self):
        self.read_csv(self.csv)