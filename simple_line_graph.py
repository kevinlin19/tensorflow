# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import time
def add_layer(data, input_size, output_size, activation = None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal(shape=[input_size, output_size]), name = 'W')
        with tf.name_scope('bias'):
            biases = tf.zeros(shape=[1, output_size], name = 'b') + 0.1
        with tf.name_scope('Wx_plus_biases'):
            Wx_plus_biases = tf.matmul(data, Weights) + biases
        
        if activation == None:
            output = Wx_plus_biases
        else:
            output = activation(Wx_plus_biases)
            
        return output
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) + 0.5 + noise
# print(x_data.shape)
# print(y_data.shape)
#1 input_layer 1 hidden_layer with 10 neurun 1 output_layer
xs = tf.placeholder(dtype=tf.float32, shape=[None, 1], name = 'x_input')
ys = tf.placeholder(dtype=tf.float32, shape=[None, 1], name = 'y_input')

l1 = add_layer(xs, 1, 10, activation=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
with tf.name_scope('train'):
    train = optimizer.minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
writer = tf.summary.FileWriter('~/logs',sess.graph)
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.ioff()
# plt.show()

plt.ion()

for i in range(1000):
    sess.run(train, feed_dict = {xs : x_data, ys : y_data})
    preds = sess.run(prediction, feed_dict = {xs : x_data})
    if i % 50 == 0:
        # try:
        #     lines = ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        plt.cla()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, preds, 'r-', lw=5)
        plt.pause(0.5)

plt.ioff()
plt.show()

        # lines = ax.plot(x_data, preds, 'r-', lw = 5)

#             plt.ion()
#             plt.show()
#             p1lt.pause(0.1)
#             ax.plot(x_data, preds, 'r-', lw = 5)
    
#             print(sess.run(loss, feed_dict = {xs : x_data, ys : y_data}))
