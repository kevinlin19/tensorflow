import tensorflow as tf
import numpy as np

#create fake data
x_data = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.05, size=[100, 1])
y_data = np.power(x_data, 2) - 0.5 + noise

#constructe tensorflow
with tf.variable_scope('Input'):
    xs = tf.placeholder(dtype=tf.float32, shape = [100, 1], name = 'x_input')
    ys = tf.placeholder(dtype=tf.float32, shape = [100, 1], name = 'y_input')
with tf.variable_scope('layer'):
    layer1 = tf.layers.dense(xs, 10, tf.nn.relu, name = 'hidden_layer')
    tf.summary.histogram('layer1', layer1)
    preds = tf.layers.dense(layer1, 1, name = 'output_layer')
    tf.summary.histogram('preds', preds)

loss = tf.losses.mean_squared_error(y_data, preds, scope='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)
tf.summary.scalar('loss', loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWriter('./logs', graph=sess.graph)
merge = tf.summary.merge_all()


for i in range(1000):
	sess.run(train, feed_dict = {xs : x_data, ys : y_data})
	if i % 50 == 0:
		print(sess.run(loss, feed_dict = {xs : x_data, ys : y_data}))
		result = sess.run(merge, feed_dict = {xs : x_data, ys : y_data})
		writer.add_summary(result, i)