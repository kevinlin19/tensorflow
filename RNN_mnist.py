# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]
n_step = 28
n_input = 28
batch_size = 64
n_class = 10
x = tf.placeholder(dtype=tf.float32, shape=[None, n_step * n_input]) #input an image 
images = tf.reshape(x, [-1, n_step, n_input])
y = tf.placeholder(dtype=tf.int32, shape=[None, 10])
# RNN
rnn_cell = tf.contrib.rnn.LSTMCell(num_units=64)
# defining initial state
# initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=images,
                                   initial_state=None,
                                   dtype=tf.float32)
preds = tf.layers.dense(inputs=outputs[:, -1, :], units=n_class, activation=tf.nn.softmax)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=preds)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(y, axis=1), predictions=tf.argmax(preds, axis=1),)[1]
sess = tf.Session()
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess.run(init_g)
sess.run(init_l)
for step in range(1200):
    x_input, y_label = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict = {x : x_input, y : y_label})
    if step % 50 == 0:
        print(sess.run(accuracy, feed_dict = {x : test_x, y : test_y}))
