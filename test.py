import tensorflow as tf
import numpy as np

tf.set_random_seed(3)

#data = tf.get_variable('data',initializer=tf.ones(shape=(2, 3, 4)),dtype=tf.float32)
data = tf.get_variable('data',initializer=tf.truncated_normal(shape=(2, 3, 4)),dtype=tf.float32)
u = tf.get_variable('weight', shape=(4, 1),initializer=tf.truncated_normal_initializer(),dtype=tf.float32)
data0 = data[0] # 3*4
u_y = tf.matmul(data0, u, name='u_y')
softmax = tf.nn.softmax(u_y, dim=0)
output = tf.matmul(data0, softmax, transpose_a=True)
output_reshape = tf.reshape(output, (1, -1))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(data0.eval())
    print(u.eval())
    print(u_y.eval())
    print(softmax.eval())
    print(output.eval())
    print(output_reshape.eval())


'''
tensor_u = tf.convert_to_tensor(u)

data0 = data[0] # (3,4)
u_y = []
for i in range(data0.get_shape()[0]):
    u_y.append(tf.tensordot(data0[i], u, axes=1))
u_y_tensor = tf.convert_to_tensor(u_y, dtype=tf.float32)
u_y_tensor = tf.reshape(u_y_tensor,(1, 3))
u_y_tensor = u_y_tensor[0]
softmax = tf.nn.softmax(u_y_tensor)

result = []
for i in range(data0.get_shape()[0]):
    result.append(tf.multiply(data0[i], softmax[i]))
#result_tensor = tf.convert_to_tensor(result)
result_tensor = tf.reduce_sum(result, axis=0)
'''
