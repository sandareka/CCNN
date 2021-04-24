import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from vgg import vgg_arg_scope
import params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def list_data(directory):
    """
    List image names and class labels
    :param directory: Images directory
    :return: set of image paths, labels
    """

    classes = {}
    class_no = 0
    with open(params.CLASS_NAME_FILE, 'r') as f:
        for line in f.readlines():
            class_name = line.rstrip("\n\r").split(".")[1]
            classes[class_name] = class_no
            class_no += 1

    labels = os.listdir(directory)

    files_and_labels = []

    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    file_names, labels = zip(*files_and_labels)
    file_names = list(file_names)
    labels = list(labels)
    labels = [classes[l] for l in labels]

    return file_names, labels


def main():
    ###############
    # Get Dataset #
    ###############
    file_names, labels = list_data(params.TEST_IMAGES_DIR)

    print('Successfully loaded the dataset!')

    graph = tf.Graph()
    with graph.as_default():

        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = 512.0
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)

            scale = tf.cond(tf.greater(height, width),
                            lambda: smallest_side / width,
                            lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])

            return resized_image, label

        # -------- Pre-processing for evaluation --------
        def preprocess(image, label, indicator_vector):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 448, 448)

            means = tf.reshape(tf.constant(params.VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means

            return centered_image, label, indicator_vector

        #######################
        # Setup Test Dataset #
        #######################

        dataset = tf.contrib.data.Dataset.from_tensor_slices((tf.constant(file_names), tf.constant(labels)))
        dataset = dataset.map(_parse_function,
                              num_threads=params.NUM_WORKERS, output_buffer_size=params.TEST_BATCH_SIZE)
        dataset = dataset.map(preprocess,
                              num_threads=params.NUM_WORKERS, output_buffer_size=params.TEST_BATCH_SIZE)
        batched_dataset = dataset.batch(params.TEST_BATCH_SIZE)

        iterator = tf.contrib.data.Iterator.from_structure(batched_dataset.output_types,
                                                           batched_dataset.output_shapes)

        data_init_op = iterator.make_initializer(batched_dataset)

        # -------- Indicates whether we are in training or in test mode --------
        is_training = tf.placeholder(tf.bool)

        # -------- Placeholders --------
        initializer = tf.contrib.layers.xavier_initializer()

        images, labels = iterator.get_next()

        ##############
        # CCNN Model #
        ##############

        with slim.arg_scope(vgg_arg_scope(weight_decay=params.WEIGHT_DECAY)):
            with tf.variable_scope('vgg_16', 'vgg_16', [images]) as sc:
                end_points_collection = sc.original_name_scope + '_end_points'

                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')

                    net = slim.conv2d(net, params.NO_CONCEPTS, [1, 1], scope='ccnn_concepts')
                    net = slim.dropout(net, params.DROP_OUT_KEEP_PROB, is_training=is_training, scope='ccnn_dropout')

                    net = tf.reduce_mean(net, [1, 2], name='global_pool')

        with tf.variable_scope('ccnn_fc'):
            fc_w = tf.Variable(initializer([params.NO_CONCEPTS, params.NO_CLASSES]), name='w')

        logits = tf.matmul(net, fc_w, name="logits")

        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)

        saver = tf.train.Saver()

        tf.get_default_graph().finalize()

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, os.path.join(params.MODEL_SAVE_FOLDER, 'model'))
        print('Successfully loaded trained CCNN!')

        sess.run(data_init_op)
        num_correct, num_samples = 0, 0
        while True:
            try:
                correct_pred = sess.run(correct_prediction,
                                        {is_training: False})
                num_correct += correct_pred.sum()
                num_samples += correct_pred.shape[0]
            except tf.errors.OutOfRangeError:
                break

        # -------- The fraction of data points that were correctly classified --------
        acc = float(num_correct) / num_samples
        print('Accuracy %f' % acc)


if __name__ == '__main__':
    main()
