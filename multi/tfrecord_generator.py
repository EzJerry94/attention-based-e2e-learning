import numpy as np
from moviepy.editor import AudioFileClip


class Generator():

    def __init__(self, csv):
        self.attributes_name = ['file', 'arousal', 'valence', 'dominance']
        self.attributes_type = ['str', 'int', 'int', 'int']
        self.csv = csv

    def read_csv(self, file):
        self.data = np.loadtxt(file, dtype='str')
        print(self.data.shape)

    def read_single_data(self, file):
        clip = AudioFileClip
        end_time = clip(file).duration
        pass

    def write_tfrecords(self):
        self.read_csv(self.csv)
        self.dict_files = dict()
        for row in self.data:
            print(row)
            time = self.read_single_data(row[0])
        pass