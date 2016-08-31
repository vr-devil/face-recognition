# -*- coding: utf-8 -*-

import tensorflow as tf
import os

IMAGE_SIZE = 32
IMAGE_DEPTH = 3

def read_data(filename_queue):
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH])
    label = tf.constant(1, tf.float32)
    return image, label

def _generete_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 4
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    tf.image_summary('images', images)

    return images, tf.reshape(labels, [batch_size, 1])


def distorted_inputs(data_dir):
    filenames = [os.path.join(data_dir, filename)
                 for filename in os.listdir(data_dir)]
    num_examples = len(filenames)

    filename_queue = tf.train.string_input_producer(filenames)
    image, label = read_data(filename_queue)

    reshaped_image = tf.cast(image, tf.float32)
    distorted_image = tf.image.random_flip_left_right(reshaped_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_whitening(distorted_image)

    return _generete_image_and_label_batch(float_image, label,
                                           num_examples, num_examples,
                                           shuffle=True)

def normal_inputs(data_dir):
    filenames = [os.path.join(data_dir, filename)
                 for filename in os.listdir(data_dir)]
    num_examples = len(filenames)

    filename_queue = tf.train.string_input_producer(filenames)
    image, label = read_data(filename_queue)

    reshaped_image = tf.cast(image, tf.float32)
    float_image = tf.image.per_image_whitening(reshaped_image)

    return _generete_image_and_label_batch(float_image, label,
                                           num_examples, num_examples,
                                           shuffle=False)