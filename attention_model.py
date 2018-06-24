import tensorflow as tf

class AttentionModel:
    def __init__(self):
        pass

    def create_model(self, inputs):
        mean = tf.reduce_mean(inputs, axis=1)
        return mean