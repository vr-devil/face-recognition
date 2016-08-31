import os

import tensorflow as tf

import cnn as network
import inputs

IMAGE_SIZE = inputs.IMAGE_SIZE
IMAGE_DEPTH = inputs.IMAGE_DEPTH
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('train_dir', '/tmp/kai_face_cnn_train',
                           """Directory where to write event logs and checkpoint.""")


def main(args=None):

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)


    with tf.Graph().as_default():
        images, labels = network.train_set()
        logits = network.inference(images)
        loss = network.loss(logits, labels)
        train = network.train(loss, 0.01)

        summary = tf.merge_all_summaries()
        init = tf.initialize_all_variables()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                for step in range(300):
                    if not coord.should_stop():
                        _, loss_value = sess.run([train, loss])
                        print 'Step %d: loss = %.2f' % (step, loss_value)

                        summary_str = sess.run(summary)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                        saver.save(sess, checkpoint_file, global_step=step)

            except tf.errors.OutOfRangeError:
                print 'Done training -- epoch limit reached'
            finally:
                coord.request_stop()

            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()