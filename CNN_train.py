# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

def accuracy(v_xs, v_ys):
    global preds
    y_preds = sess.run(preds, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_preds,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weights_variable(shape):
	initial = tf.truncated_normal(shape=shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pooling(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding= 'SAME')
#placeholder
with tf.name_scope('input'):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 784], name = 'x')/255.
ys = tf.placeholder(dtype=tf.float32, shape=[None, 10], name = 'y')
keep_prob = tf.placeholder(dtype=tf.float32)
images = tf.reshape(xs, [-1, 28, 28, 1])


## conv2d_layer1 ##

weight_layer1 = weights_variable([5, 5, 1, 32])
bias_layer1 = bias_variable([32])
conv_layer1 = tf.nn.relu(conv2d(images, weight_layer1) + bias_layer1) #n_samplesx28x28x32
pool1 = max_pooling(conv_layer1) #n_samplesx14x14x32

## conv2d_layer2 ##
weight_layer2 = weights_variable([5, 5, 32, 64])
bias_layer2 = bias_variable([64])
conv_layer2 = tf.nn.relu(conv2d(pool1, weight_layer2) + bias_layer2) #n_samplesx14x14x64
pool2 = max_pooling(conv_layer2)                                     #n_samplesx7x7x64

## flatten ##
conv_flat = tf.reshape(pool2, [-1, 7*7*64]) #n_samplesx7*7*64

## full connect ##
weight_full1 = weights_variable([7*7*64, 1024])
bias_full = bias_variable([1024])
full1 = tf.nn.relu(tf.matmul(conv_flat, weight_full1) + bias_full)  #n_samplesx1024
full_drop = tf.nn.dropout(full1, keep_prob)

## preds ##
weight_preds = weights_variable([1024, 10])
bias_preds = bias_variable([10])
preds = tf.nn.softmax(tf.matmul(full_drop, weight_preds) + bias_preds)

## train ##
# loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=preds)
with tf.name_scope('loss'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=preds)
    tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()

merged = tf.summary.merge_all() # tensorflow >= 0.12

# writer = tf.train.SummaryWriter('logs/', sess.graph)    # tensorflow < 0.12
writer = tf.summary.FileWriter("logs/", sess.graph) # tensorflow >=0.12

saver = tf.train.Saver()

sess.run(init)
for i in range(1000):
    # x_train = mnist.train.images[:2000] #[n_samples, 28x28x1] 1:rgb
    # y_train = mnist.train.labels[:2000]
    x_test = mnist.test.images[:500]
    y_test = mnist.test.labels[:500]
    b_x, b_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={xs: b_x, ys: b_y, keep_prob: 0.5})
    # sess.run(train, feed_dict = {xs : x_train, ys : y_train, keep_prob : 0.5})
    if i % 50 == 0:
        res = sess.run(merged, feed_dict={xs: b_x, ys: b_y, keep_prob: 0.5})
        writer.add_summary(res, i)
        print(accuracy(x_test, y_test))

save_path = saver.save(sess, "/my_variable_saver/save_CNN_variable.ckpt")
print("Save to path: ", save_path)

sess.close()

