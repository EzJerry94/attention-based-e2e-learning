import tensorflow as tf

slim = tf.contrib.slim

class CNNModel():

    def __init__(self):
        self.is_training = True
        self.conv_filters = 40

    def create_model(self, frames):
        with tf.variable_scope("cnn", reuse=tf.AUTO_REUSE):
            batch_size, num_features = frames.get_shape().as_list()
            shape = ([-1, 1, num_features, 1])
            cnn_input = tf.reshape(frames, shape)

            weight1 = tf.get_variable("weight1", shape=[1, 20, 1, self.conv_filters], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
            kernel1 = tf.nn.conv2d(cnn_input,weight1, strides=[1, 1, 1, 1], padding='SAME')
            bias1 = tf.Variable(tf.zeros([40], dtype=tf.float32), name='bias1')
            conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))

            # Subsampling of the signal to 8Khz
            max_pool1 = tf.nn.max_pool(
                conv1,
                ksize=[1, 1, 2, 1],
                strides=[1, 1, 2, 1],
                padding='SAME',
                name='pool1'
            )

            weight2 = tf.get_variable("weight2", shape=[1, 40, 40, self.conv_filters], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
            kernel2 = tf.nn.conv2d(max_pool1, weight2, strides=[1, 1, 1, 1], padding='SAME')
            bias2 = tf.Variable(tf.zeros([40], dtype=tf.float32), name='bias2')
            conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))

            reshape1 = tf.reshape(conv2, (-1, num_features // 2, self.conv_filters, 1))

            # Pooling over the feature maps.
            max_pool2 = tf.nn.max_pool(
                reshape1,
                ksize=[1, 1, 10, 1],
                strides=[1, 1, 10, 1],
                padding='SAME',
                name='pool2'
            )

            net = tf.reshape(max_pool2, (-1, num_features // 2 * 4))

        return net

    def create_model_2(self, frames:tf.Tensor, conv_filters:int = 40, is_training: bool = True):
        with tf.variable_scope("audio_model", reuse=tf.AUTO_REUSE):
            batch_size, num_features = frames.get_shape().as_list()
            shape = ([-1, 1, num_features, 1])
            audio_input = tf.reshape(frames, shape)

            with slim.arg_scope([slim.layers.conv2d], padding='SAME'):
                #net = slim.dropout(audio_input, is_training=self.is_training)
                net = slim.layers.conv2d(audio_input, conv_filters, (1, 20))

                # Subsampling of the signal to 8Khz
                net = tf.nn.max_pool(
                    net,
                    ksize=[1, 1, 2, 1],
                    strides=[1, 1, 2, 1],
                    padding='SAME',
                    name='pool1'
                )

                # Original model had 400 output filters for the second conv layer
                # but this trains much faster and achieves comparable accuracy.
                net = slim.layers.conv2d(net, conv_filters, (1, 40))
                net = tf.reshape(net, (-1, num_features // 2, conv_filters, 1))

                # Pooling over the feature maps.
                net = tf.nn.max_pool(
                    net,
                    ksize=[1, 1, 10, 1],
                    strides=[1, 1, 10, 1],
                    padding='SAME',
                    name='pool2'
                )

            net = tf.reshape(net, (-1, num_features // 2 * 4))

        return net