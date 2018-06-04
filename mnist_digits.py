from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Define hyperparameters
learning_rate = 0.5
batch_size = 64
num_steps = 100

# Load MNIST data set
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

# Set up loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits = y_pred, labels = y))

# Use GradientDescentOptimizer as optimization algorithm
gd = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Accuracy of the model
correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(num_steps):
        x_train, y_train = mnist.train.next_batch(batch_size)
        sess.run(gd, feed_dict={x: x_train, y: y_train})

        if step % 10 == 0:
            res = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print('step {}, accuracy: {}'.format(step, res))

print('final accuracy: {}'.format(np.mean(res)))
