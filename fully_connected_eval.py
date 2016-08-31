import tensorflow as tf

import fully_connected as network
import inputs

IMAGE_SIZE = inputs.IMAGE_SIZE
IMAGE_DEPTH = inputs.IMAGE_DEPTH
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/kai_face_train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/kai_face_eval',
                           """Directory where to write event logs.""")


def main(args=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    with tf.Graph().as_default():
        images, labels = network.test_set()
        logits = network.inference(images)

        correct_prediction = tf.greater_equal(tf.reshape(tf.sigmoid(logits), [tf.shape(logits)[0]]), 0.9)

        saver = tf.train.Saver()
        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return


            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                if not coord.should_stop():
                    result = sess.run(correct_prediction)

                    true_count = 0.0
                    for i in result:
                        if i == True:
                            true_count += 1

                    print 'accuracy: %.2f' % (true_count / len(result))


            except tf.errors.OutOfRangeError:
                print 'Done training -- epoch limit reached'
            finally:
                coord.request_stop()

            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
