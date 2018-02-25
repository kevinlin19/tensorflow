import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
x_train = mnist.train.

def accuracy():
	global preds


def weights_variable(shape):
	initial = tf.truncated_normal(shape=shape,stddev=1.0)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pooling():
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding= 'SAME')

