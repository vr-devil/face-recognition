import os

import tensorflow as tf

import inputs

IMAGE_SIZE = inputs.IMAGE_SIZE
IMAGE_DEPTH = inputs.IMAGE_DEPTH
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', '/tmp/kai_face_data',
                           """Path to the Kai-Face data directory.""")

def train_set():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    train_data_dir = os.path.join(FLAGS.data_dir, 'train')
    images, labels = inputs.distorted_inputs(train_data_dir)
    return tf.reshape(images, [tf.shape(images)[0], IMAGE_PIXELS]), labels


def test_set():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    test_data_dir = os.path.join(FLAGS.data_dir, 'test')
    images, labels = inputs.normal_inputs(test_data_dir)
    return tf.reshape(images, [tf.shape(images)[0], IMAGE_PIXELS]), labels


def inference(images):

    hidden_units = 100

    with tf.variable_scope('hidden'):
        w = tf.Variable(tf.random_normal([IMAGE_PIXELS, hidden_units]), name='weights')
        b = tf.Variable(tf.zeros([hidden_units]), name='biases')
        hidden = tf.sigmoid(tf.matmul(images, w) + b)

    with tf.variable_scope('output'):
        w = tf.Variable(tf.random_normal([hidden_units, 1]), name='weights')
        b = tf.Variable(tf.zeros([1]), name='biases')
        logits = tf.matmul(hidden, w) + b

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
