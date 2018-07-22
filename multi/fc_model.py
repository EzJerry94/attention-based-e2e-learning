import tensorflow as tf

slim = tf.contrib.slim

class FCModel():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def create_model(self, inputs):
        with tf.variable_scope("fc", reuse=tf.AUTO_REUSE):
            fc = slim.layers.linear(inputs, 3)
            #fc = tf.contrib.layers.fully_connected(inputs, self.num_classes)
            #weight = tf.get_variable("fc_weight", shape=[128, 3], dtype=tf.float32,
            #                          initializer=tf.contrib.layers.xavier_initializer())
            #bias = tf.Variable(tf.zeros([3], dtype=tf.float32), name='fc_bias')
            #fc = tf.nn.bias_add(tf.matmul(inputs, weight), bias)
        return fc