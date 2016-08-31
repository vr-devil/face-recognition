import os

import tensorflow as tf

import inputs

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', '/tmp/kai_face_data',
                           """Path to the Kai-Face data directory.""")


def train_set():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    train_data_dir = os.path.join(FLAGS.data_dir, 'train')
    images, labels = inputs.distorted_inputs(train_data_dir)
    return images, labels


def test_set():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    test_data_dir = os.path.join(FLAGS.data_dir, 'test')
    images, labels = inputs.normal_inputs(test_data_dir)
    return images, labels


def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.random_normal([5, 5, 3, 20]), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([20]), name='biases')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    with tf.variable_scope('local1') as scope:
        reshape = tf.reshape(pool1, [images.get_shape()[0].value, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.random_normal([dim, 100], dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.zeros([100]), name='biases')
        local1 = tf.nn.sigmoid(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('local2') as scope:
        weights = tf.Variable(tf.random_normal([100, 1]), name='weights')
        biases = tf.Variable(tf.zeros([1]), name='biases')
        logits = tf.add(tf.matmul(local1, weights), biases, name=scope.name)

    return logits


def loss(logits, labels):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels, name='cross_entropy')
    return tf.reduce_mean(cross_entropy, name='cross_entropy_mean')


def train(loss_op, learning_rate):
    tf.scalar_summary(loss_op.op.name, loss_op)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer.minimize(loss_op)


def evaluation():
    pass
