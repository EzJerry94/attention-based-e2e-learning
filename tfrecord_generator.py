import tensorflow as tf
import numpy as np
import copy
from moviepy.editor import AudioFileClip

class Generator:

    def __init__(self):
        self.csv = 'data/train_set.csv'
        self.upsample = True
        self.classes = 3

    def read_csv(self, file):
        self.data = np.loadtxt(file, dtype='str')

    def upsample_process(self, sample_data):
        classes = [int(x['label']) for x in sample_data.values()]
        class_ids = set(classes)
        num_samples_per_class = {class_name: sum(x == class_name for x in classes) for class_name in class_ids}

        max_samples = np.max(list(num_samples_per_class.values()))
        augmented_data = copy.copy(sample_data)
        for class_name, n_samples in num_samples_per_class.items():
            n_samples_to_add = max_samples - n_samples

            while n_samples_to_add > 0:
                for key, value in sample_data.items():
                    label = int(value['label'])
                    sample = key
                    if n_samples_to_add <= 0:
                        break

                    if label == class_name:
                        augmented_data[sample + '_' + str(n_samples_to_add)] = {'file': value['file'],
                                                                                'label': np.int32(label)}
                        n_samples_to_add -= 1

        return augmented_data

    def write_tfrecords(self):
        self.read_csv(self.csv)
        print(self.data)
        self.dict_files = dict()
        for row in self.data:
            # print('Get duration of file : {}'.format(row[0]))
            # time = self.read_single_data_time(row[0])
            self.dict_files[row[0]] = {'file': row[0],
                                       'label': np.int32(row[2]),
                                        }
        if self.upsample:
            self.dict_files = self.upsample_process(self.dict_files)

        print('\n Start generating tfrecords \n')