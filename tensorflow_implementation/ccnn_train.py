"""
Tensorflow implementation of CCNN.
Uses Tensorflow slim and tf.contrib.data.
Based on the Tensorflow slim implementation of VGG in
https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/
"""
import os
import numpy as np
from os import makedirs
import tensorflow as tf
import tensorflow.contrib.slim as slim
from vgg import vgg_arg_scope
import math
import cub_params as params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def list_data(directory, is_training=True):
    """
    List image names, class labels and indicator vectors
    :param directory: Images directory
    :param is_training: if data set being prepared is for training or validation
    :return: set of image paths, labels, indicator vectors
    """

    classes = {}
    class_no = 0
    with open(params.CLASS_NAME_FILE, 'r') as f:
        for line in f.readlines():
            class_name = line.rstrip("\n\r").split(".")[1]
            classes[class_name] = class_no
            class_no += 1

    indicator_vectors_dict = np.load(params.INDICATOR_VECTORS, allow_pickle=True).item()

    labels = os.listdir(directory)

    files_and_labels = []
    indicator_vectors = []

    for label in labels:
        no_images = 0
        for f in os.listdir(os.path.join(directory, label)):
            image_name = f
            if is_training:
                if no_images % 10 != 0:
                    indicator_vectors.append(indicator_vectors_dict[image_name])
                    files_and_labels.append((os.path.join(directory, label, f), label))
                no_images += 1
            else:
                if no_images % 10 == 0:
                    files_and_labels.append((os.path.join(directory, label, f), label))
                    indicator_vectors.append(np.ones(shape=params.NO_CONCEPTS))  # Let's put some dummy data as validation indicator vectors
                no_images += 1

    file_names, labels = zip(*files_and_labels)
    file_names = list(file_names)
    labels = list(labels)
    labels = [classes[l] for l in labels]
    indicator_vectors = np.asanyarray(indicator_vectors)

    return file_names, labels, indicator_vectors.astype(np.float32)


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op, word_phrases_vector):
    """
    Check the accuracy of the model on either train or validation (depending on dataset_init_op).
    :param sess:
    :param correct_prediction: tensorflow tensor
    :param is_training: tensorflow placeholder
    :param dataset_init_op: dataset on which accuracy should be calculated
    :param word_phrases_vector: tensorflow placeholder for concept word vectors
    :return: classification accuracy
    """
    concept_text_features = np.load(params.CONCEPT_WORD_VECTORS).astype(np.float32)
    # -------- Initialize the dataset --------
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction,
                                    {is_training: False,
                                     word_phrases_vector: concept_text_features})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # -------- Return the fraction of datapoints that were correctly classified --------
    acc = float(num_correct) / num_samples
    return acc


def main():
    ################
    # Get Datasets #
    ################
    train_file_names, train_labels, train_indicator_vectors = list_data(params.TRAIN_IMAGES_DIR, True)
    val_file_names, val_labels, val_indicator_vectors = list_data(params.TRAIN_IMAGES_DIR, False)
    class_discriminative_features_array = np.load(params.CLASS_DISCRIMINATIVE_FEATURE)

    print('Successfully loaded datasets!')

    assert set(train_labels) == set(val_labels), \
        "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels), set(val_labels))

    graph = tf.Graph()
    with graph.as_default():

        # -------- Functions to do online data augmentation --------
        def flip(image):
            flip_image = tf.image.random_flip_left_right(image)
            return flip_image

        def rotate(image):
            random_value = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32, seed=None, name=None)[0] * 100
            rot_image = tf.contrib.image.rotate(image, (random_value * math.pi) / 180, interpolation='BILINEAR')
            return rot_image

        def gaussian_noise(image):
            noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1.0,
                                     dtype=tf.float32)
            noise_image = tf.add(image, noise)
            return noise_image

        def _corrupt_contrast(image):
            image = tf.image.random_contrast(image, 0, 5)
            return image

        def _corrupt_saturation(image):
            image = tf.image.random_saturation(image, 0, 5)
            return image

        def _corrupt_brightness(image, ):
            image = tf.image.random_brightness(image, 5)
            return image

        def _parse_function(filename, label, indicator_vector):
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

            return resized_image, label, indicator_vector

        # -------- Pre-processing for training --------
        def training_preprocess(image, label, indicator_vector):
            crop_image = tf.random_crop(image, [448, 448, 3])
            pre_process_type = tf.random_uniform([1], minval=0, maxval=7, dtype=tf.int32, seed=None, name=None)[
                0]

            pre_processed_image = tf.case(pred_fn_pairs=[
                (tf.equal(pre_process_type, 1), lambda: flip(crop_image)),
                (tf.equal(pre_process_type, 2), lambda: _corrupt_contrast(crop_image)),
                (tf.equal(pre_process_type, 3), lambda: _corrupt_brightness(crop_image)),
                (tf.equal(pre_process_type, 4), lambda: gaussian_noise(crop_image)),
                (tf.equal(pre_process_type, 5), lambda: _corrupt_saturation(crop_image)),
                (tf.equal(pre_process_type, 6), lambda: rotate(crop_image))],
                default=lambda: crop_image, exclusive=True)

            means = tf.reshape(tf.constant(params.VGG_MEAN), [1, 1, 3])
            centered_image = pre_processed_image - means

            return centered_image, label, indicator_vector

        # -------- Pre-processing for validation --------
        def val_preprocess(image, label, indicator_vector):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 448, 448)

            means = tf.reshape(tf.constant(params.VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means

            return centered_image, label, indicator_vector

        ############################
        # Setup Train Val Datasets #
        ############################


        train_dataset = tf.contrib.data.Dataset.from_tensor_slices(
            (tf.constant(train_file_names), tf.constant(train_labels), train_indicator_vectors))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_file_names))
        train_dataset = train_dataset.map(_parse_function,
                                          num_threads=params.NUM_WORKERS, output_buffer_size=params.TRAIN_BATCH_SIZE)
        train_dataset = train_dataset.map(training_preprocess,
                                          num_threads=params.NUM_WORKERS, output_buffer_size=params.TRAIN_BATCH_SIZE)
        batched_train_dataset = train_dataset.batch(params.TRAIN_BATCH_SIZE)

        # -------- Validation dataset ---------
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((tf.constant(val_file_names), tf.constant(val_labels), val_indicator_vectors))
        val_dataset = val_dataset.map(_parse_function,
                                      num_threads=params.NUM_WORKERS, output_buffer_size=params.VAL_BATCH_SIZE)
        val_dataset = val_dataset.map(val_preprocess,
                                      num_threads=params.NUM_WORKERS, output_buffer_size=params.VAL_BATCH_SIZE)
        batched_val_dataset = val_dataset.batch(params.VAL_BATCH_SIZE)

        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types, batched_train_dataset.output_shapes)

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        # --------- Indicates whether we are in training or in evaluation mode ---------
        is_training = tf.placeholder(tf.bool)

        class_discriminative_features = tf.convert_to_tensor(class_discriminative_features_array, dtype=tf.float32, name="class_discriminative_features")

        # -------- Placeholders --------
        word_phrases_vector = tf.placeholder(tf.float32, shape=[params.NO_CONCEPTS, params.TEXT_SPACE_DIM], name="word_phrases_vector")

        images, labels, indicator_vectors = iterator.get_next()

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

                    concept_layer_feats = net

                    net = tf.reduce_mean(net, [1, 2], name='global_pool')

        with tf.variable_scope('ccnn_fc'):
            init = np.multiply(0.1 * np.ones((params.NO_CONCEPTS, params.NO_CLASSES)), class_discriminative_features_array)
            fc_w = tf.Variable(initial_value=init, name='w', dtype=tf.float32)

        logits = tf.matmul(net, fc_w, name="logits")

        ####################
        # Loss Calculation #
        ####################

        with tf.variable_scope('ccnn_embedding'):
            v_e_converter = tf.Variable(tf.random_normal([196, params.EMBED_SPACE_DIM], stddev=0.1), name='v_e_converter')
            w_e_converter = tf.Variable(tf.random_normal([params.TEXT_SPACE_DIM, params.EMBED_SPACE_DIM], stddev=0.1), name='w_e_converter')

        visual_feats = tf.reshape(
            tf.transpose(tf.reshape(concept_layer_feats, [-1, 196, params.NO_CONCEPTS], name="flat_visual_feats"), perm=[0, 2, 1]),
            [-1, 196], name="visual_feats")
        embedded_visual_feats = tf.matmul(visual_feats, v_e_converter, name="embedded_visual_features")

        embedded_text_feats = tf.matmul(word_phrases_vector, w_e_converter, name="embedded_text_features")

        normalized_emd_visual_feats = tf.nn.l2_normalize(embedded_visual_feats, dim=1)
        normalized_text_feats = tf.nn.l2_normalize(embedded_text_feats, dim=1)

        # -------- Semantic loss calculation --------
        cosine_similarity = tf.matmul(normalized_emd_visual_feats, tf.transpose(normalized_text_feats))
        cos_sim_reshaped = tf.reshape(cosine_similarity, [-1, params.NO_CONCEPTS, params.NO_CONCEPTS])
        positive = tf.reduce_sum(tf.multiply(cos_sim_reshaped, tf.eye(params.NO_CONCEPTS)), axis=2)
        metric_p = tf.tile(tf.expand_dims(positive, axis=2), [1, 1, params.NO_CONCEPTS])
        delta = tf.subtract(cos_sim_reshaped, metric_p)
        semantic_loss = tf.multiply(
            tf.reduce_sum(tf.nn.relu(tf.add((params.ALPHA + delta), -2 * tf.eye(params.NO_CONCEPTS))), axis=2),
            indicator_vectors)

        # -------- Counter loss calculation --------

        indices = tf.range(start=0, limit=tf.shape(positive)[0], dtype=tf.int32)
        shuffled_indices = tf.random_shuffle(indices)

        shuffled_positive = tf.gather(positive, shuffled_indices)
        shuffled_indicator_vectors = tf.gather(indicator_vectors, shuffled_indices)

        img_only_feats = tf.nn.relu(tf.subtract(indicator_vectors, shuffled_indicator_vectors))

        count_loss = tf.multiply(tf.nn.relu(tf.subtract(tf.multiply(shuffled_positive, img_only_feats), tf.multiply(positive, img_only_feats)) + params.BETA),
                                 img_only_feats)

        # -------- Uniqueness loss calculation --------
        sigmoid_inputs = tf.subtract(net, tf.tile(tf.expand_dims(tf.reduce_mean(net, axis=1), axis=1), [1, params.NO_CONCEPTS]))

        uniqueness_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=indicator_vectors,
                                                                  logits=(sigmoid_inputs + tf.constant(1e-5)))  # -5

        interpretation_loss = uniqueness_loss + semantic_loss + count_loss
        interpretation_loss = tf.reduce_mean(interpretation_loss, axis=1, keep_dims=True)

        # -------- Classification loss calculation --------

        classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                             logits=(logits + tf.constant(1e-5)))

        total_loss = tf.reduce_mean(tf.add(classification_loss, params.LAMBDA_VALUE * interpretation_loss))

        ######################
        # Evaluation Metrics #
        ######################

        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)

        #######################
        # Optimization set up #
        #######################

        # -------- Specify where the model checkpoint is (pre-trained weights)--------
        model_path = params.PRE_TRAINED_MODEL_WEIGHTS
        assert (os.path.isfile(model_path))

        global_step = tf.Variable(0, name='global_step', trainable=False)

        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/ccnn_concepts', 'global_step', 'ccnn_embedding/', 'ccnn_fc/'])
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # ------- First only train newly added variables --------
        model_variables = tf.contrib.framework.get_variables('vgg_16/ccnn_concepts') + tf.contrib.framework.get_variables('ccnn_embedding/')

        learning_rate1 = tf.train.exponential_decay(params.LEARNING_RATE_1, global_step, 100000, 0.96, staircase=True)

        new_var_optimizer = tf.train.AdamOptimizer(learning_rate1)
        new_var_train_op = new_var_optimizer.minimize(total_loss, var_list=model_variables, global_step=global_step)

        optimizer_slots = [
            new_var_optimizer.get_slot(var, name)
            for name in new_var_optimizer.get_slot_names()
            for var in model_variables if var is not None
        ]
        if isinstance(new_var_optimizer, tf.train.AdamOptimizer):
            optimizer_slots.extend([
                new_var_optimizer._beta1_power, new_var_optimizer._beta2_power
            ])

        new_variables = [
            *model_variables,
            *optimizer_slots,
            global_step,
            fc_w,
        ]

        new_variables = filter(lambda x: x is not None, new_variables)
        new_variables_init = tf.variables_initializer(new_variables)

        # -------- Next fine-tune all the variables in the network --------
        model_variables = tf.contrib.framework.get_variables('vgg_16/') + tf.contrib.framework.get_variables('ccnn_embedding/')

        learning_rate2 = tf.train.exponential_decay(params.LEARNING_RATE_2, global_step, 100000, 0.96, staircase=True)

        full_var_optimizer = tf.train.AdamOptimizer(learning_rate2)
        full_var_train_op = full_var_optimizer.minimize(total_loss, global_step=global_step, var_list=model_variables)
        optimizer_slots = [
            full_var_optimizer.get_slot(var, name)
            for name in full_var_optimizer.get_slot_names()
            for var in model_variables if var is not None
        ]
        if isinstance(full_var_optimizer, tf.train.AdamOptimizer):
            optimizer_slots.extend([
                full_var_optimizer._beta1_power, full_var_optimizer._beta2_power
            ])

        full_variables = [
            *optimizer_slots
        ]

        full_variables = filter(lambda x: x is not None, full_variables)
        full_variables_init = tf.variables_initializer(full_variables)

        # -------- Finally fine-tune all the fully connected layer --------
        model_variables = tf.contrib.framework.get_variables('ccnn_fc/')

        learning_rate3 = tf.train.exponential_decay(params.LEARNING_RATE_3, global_step, 100000, 0.96, staircase=True)

        fc_total_loss = tf.reduce_mean(classification_loss) + tf.reduce_mean(tf.abs(tf.multiply((1 - class_discriminative_features), fc_w)))

        fc_var_optimizer = tf.train.AdamOptimizer(learning_rate3)
        fc_var_train_op = fc_var_optimizer.minimize(fc_total_loss, global_step=global_step, var_list=model_variables)
        optimizer_slots = [
            full_var_optimizer.get_slot(var, name)
            for name in full_var_optimizer.get_slot_names()
            for var in model_variables if var is not None
        ]
        if isinstance(full_var_optimizer, tf.train.AdamOptimizer):
            optimizer_slots.extend([
                full_var_optimizer._beta1_power, full_var_optimizer._beta2_power
            ])

        fc_variables = [
            *optimizer_slots
        ]

        fc_variables = filter(lambda x: x is not None, fc_variables)
        fc_variables_init = tf.variables_initializer(fc_variables)

        saver = tf.train.Saver()

        tf.get_default_graph().finalize()

    #################
    # Training CCNN #
    #################

    concept_text_features = np.load(params.CONCEPT_WORD_VECTORS)
    if not os.path.exists(params.MODEL_SAVE_FOLDER):
        makedirs(os.path.join(params.MODEL_SAVE_FOLDER))

    save_path = os.path.join(params.MODEL_SAVE_FOLDER, 'model')

    with tf.Session(graph=graph) as sess:
        init_fn(sess)  # load the pre-trained weights
        print('Successfully loaded pre-trained weights!')
        sess.run(new_variables_init)
        sess.run(full_variables_init)
        sess.run(fc_variables_init)
        best_val_acc = 0.0

        # -------- Update only the newly added variables for a few epochs.
        for epoch in range(params.MAX_NUM_EPOCHS_1):
            loss_for_epoch = 0.0
            c_loss_for_epoch = 0.0
            u_loss_for_epoch = 0.0
            s_loss_for_epoch = 0.0
            count_loss_for_epoch = 0.0
            iteration = 0

            # -------- Run an epoch over the training data.
            print('Starting epoch %d / %d' % (epoch + 1, params.NUM_EPOCHS_1))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.
            sess.run(train_init_op)
            while True:
                try:
                    _, u_loss, s_loss, co_loss, c_loss, t_loss = sess.run(
                        [new_var_train_op, uniqueness_loss, semantic_loss, count_loss, classification_loss,
                         total_loss],
                        {is_training: True,
                         word_phrases_vector: concept_text_features})

                    loss_for_epoch = loss_for_epoch + t_loss
                    c_loss_for_epoch = c_loss_for_epoch + np.mean(c_loss)
                    u_loss_for_epoch = u_loss_for_epoch + np.mean(u_loss)
                    s_loss_for_epoch = s_loss_for_epoch + np.mean(s_loss)
                    count_loss_for_epoch = count_loss_for_epoch + np.mean(co_loss)
                    iteration = iteration + 1

                except tf.errors.OutOfRangeError:
                    break

            # -------- Check accuracy on the train and val sets every epoch.
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op, word_phrases_vector)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op, word_phrases_vector)
            mean_total_loss = loss_for_epoch / iteration
            mean_cls_loss = c_loss_for_epoch / iteration
            mean_uni_loss = u_loss_for_epoch / iteration
            mean_sim_loss = s_loss_for_epoch / iteration
            mean_count_loss = count_loss_for_epoch / iteration

            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)
            print('Uniqueness loss %f, Semantic loss %f, Count loss %f, Classification loss %f, Total loss: %f' % (
                mean_uni_loss, mean_sim_loss, mean_count_loss, mean_cls_loss, mean_total_loss))

            if val_acc > best_val_acc:
                print("Saving model")
                saver.save(sess, save_path)
                best_val_acc = val_acc

        # -------- Train the entire model for a few more epochs, continuing with the *same* weights.
        for epoch in range(params.MAX_NUM_EPOCHS_2):
            loss_for_epoch = 0.0
            c_loss_for_epoch = 0.0
            u_loss_for_epoch = 0.0
            s_loss_for_epoch = 0.0
            count_loss_for_epoch = 0.0
            iteration = 0

            print('Starting epoch %d / %d' % (epoch + 1, params.NUM_EPOCHS_2))
            sess.run(train_init_op)
            while True:
                try:
                    _, u_loss, s_loss, co_loss, c_loss, t_loss = sess.run(
                        [full_var_train_op, uniqueness_loss, semantic_loss, count_loss,
                         classification_loss,
                         total_loss],
                        {is_training: True,
                         word_phrases_vector: concept_text_features})

                    loss_for_epoch = loss_for_epoch + t_loss
                    c_loss_for_epoch = c_loss_for_epoch + np.mean(c_loss)
                    u_loss_for_epoch = u_loss_for_epoch + np.mean(u_loss)
                    s_loss_for_epoch = s_loss_for_epoch + np.mean(s_loss)
                    count_loss_for_epoch = count_loss_for_epoch + np.mean(co_loss)
                    iteration = iteration + 1

                except tf.errors.OutOfRangeError:
                    break

            # -------- Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op, word_phrases_vector)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op,  word_phrases_vector)

            mean_total_loss = loss_for_epoch / iteration
            mean_cls_loss = c_loss_for_epoch / iteration
            mean_uni_loss = u_loss_for_epoch / iteration
            mean_sim_loss = s_loss_for_epoch / iteration
            mean_count_loss = count_loss_for_epoch / iteration

            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)
            print('Uniqueness loss %f, Semantic loss %f, Count loss %f, Classification loss %f, Total loss: %f' % (
                mean_uni_loss, mean_sim_loss, mean_count_loss, mean_cls_loss, mean_total_loss))

            if val_acc > best_val_acc:
                print("Saving model")
                saver.save(sess, save_path)
                best_val_acc = val_acc

        # -------- Fine-tune the FC layer while keeping other weights fixed.
        for epoch in range(params.MAX_NUM_EPOCHS_3):
            loss_for_epoch = 0.0
            c_loss_for_epoch = 0.0
            iteration = 0

            print('Starting epoch %d / %d' % (epoch + 1, params.NUM_EPOCHS_3))
            sess.run(train_init_op)
            while True:
                try:
                    _, c_loss, t_loss = sess.run(
                        [fc_var_train_op, classification_loss,
                         total_loss],
                        {is_training: True,
                         word_phrases_vector: concept_text_features})

                    loss_for_epoch = loss_for_epoch + t_loss
                    c_loss_for_epoch = c_loss_for_epoch + np.mean(c_loss)
                    iteration = iteration + 1

                except tf.errors.OutOfRangeError:
                    break

            # -------- Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op, word_phrases_vector)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op, word_phrases_vector)

            mean_total_loss = loss_for_epoch / iteration
            mean_cls_loss = c_loss_for_epoch / iteration

            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)
            print('Classification loss %f, Total loss: %f' % (
                mean_cls_loss, mean_total_loss))

            if val_acc > best_val_acc:
                print("Saving model")
                saver.save(sess, save_path)
                best_val_acc = val_acc


if __name__ == '__main__':
    main()
