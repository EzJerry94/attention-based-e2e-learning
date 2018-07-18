import tensorflow as tf
import numpy as np
from moviepy.editor import AudioFileClip



class Generator():

    def __init__(self, csv):
        self.csv = csv
        self.tfrecords_file = './data/multi_set.tfrecords'

    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_csv(self, file):
        self.data = np.loadtxt(file, dtype='str')

    def read_single_data_time(self, file):
        clip = AudioFileClip
        end_time = clip(file).duration
        return end_time

    def get_samples(self, data_file):
        #time = self.dict_files[data_file]['time']
        audio_clip = AudioFileClip(data_file)
        clip = audio_clip.set_fps(16000)
        #num_samples = int(clip.fps * time)
        data_frame = np.array(list(clip.subclip(0).iter_frames()))
        data_frame = data_frame.mean(1)
        chunk_size = 640 # split audio file to chuncks of 40ms
        audio = np.pad(data_frame, (0, chunk_size - data_frame.shape[0] % chunk_size), 'constant')
        audio = np.reshape(audio, (-1, chunk_size)).astype(np.float32)
        return audio

    def write_tfrecords(self):
        self.read_csv(self.csv)
        self.dict_files = dict()
        for row in self.data:
            #print('Get duration of file : {}'.format(row[0]))
            #time = self.read_single_data_time(row[0])
            self.dict_files[row[0]] = {'file': row[0],
                                       'arousal': np.int32(row[1]),
                                       'valence': np.int32(row[2]),
                                       'dominance': np.int32(row[3])}

        print('\n Start generating tfrecords \n')

        writer = tf.python_io.TFRecordWriter(self.tfrecords_file)

        for data_file in self.dict_files.keys():
            print('Writing file : {}'.format(data_file))
            frame = self.get_samples(data_file)
            frame = np.array(frame)

            example = tf.train.Example(features=tf.train.Features(feature={
                'file': self._bytes_feature(self.dict_files[data_file]['file'].encode()),
                'arousal': self._bytes_feature(self.dict_files[data_file]['arousal'].tobytes()),
                'valence': self._bytes_feature(self.dict_files[data_file]['valence'].tobytes()),
                'dominance': self._bytes_feature(self.dict_files[data_file]['dominance'].tobytes()),
                'frame': self._bytes_feature(frame.tobytes())
            }))
            writer.write(example.SerializeToString())

        writer.close()