import tensorflow as tf
import numpy as np
from moviepy.editor import AudioFileClip



class Generator():

    def __init__(self, csv):
        self.attributes_name = ['file', 'arousal', 'valence', 'dominance']
        self.attributes_type = ['str', 'int', 'int', 'int']
        self.csv = csv
        self.tfrecords_file = './data/multi_set.tfrecords'


    def read_csv(self, file):
        self.data = np.loadtxt(file, dtype='str')

    def read_single_data_time(self, file):
        clip = AudioFileClip
        end_time = clip(file).duration
        return end_time

    def get_samples(self, data_file):
        time = self.dict_files[data_file]['time']
        audio_clip = AudioFileClip(data_file)
        clip = audio_clip.set_fps(16000)
        num_samples = int(clip.fps * time)
        frames = []
        data_frame = np.array(list(clip.subclip(0).iter_frames()))
        data_frame = data_frame.mean(1)[:num_samples]
        return frames

    def serialize_sample(self, writer, data_file):
        frames = self.get_samples(data_file)

    def write_tfrecords(self):
        self.read_csv(self.csv)
        self.dict_files = dict()
        for row in self.data:
            time = self.read_single_data_time(row[0])
            self.dict_files[row[0]] = {'file': row[0],
                                       'time': np.float32(time),
                                       'arousal': np.int32(row[1]),
                                       'valence': np.int32(row[2]),
                                       'dominance': np.int32(row[3])}

        print('\n Start generating tfrecords \n')

        writer = tf.python_io.TFRecordWriter(self.tfrecords_file)

        for data_file in self.dict_files.keys():
            print('Writing file : {}'.format(data_file))
            self.serialize_sample(writer, data_file)

        writer.close()