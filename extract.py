# ==============================================================================
"""Generic script that extracts cnn features or scores from a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import scipy.io
import h5py
from nets import nets_factory
from preprocessing import preprocessing_factory
from PIL import Image
from classifier.app_config import FLAGS
from classifier import loss_ops

slim = tf.contrib.slim

def _get_variables_to_restore(tf_global_step):
    """
        TODO - Check with moving averages
    :param tf_global_step:
    :return:
    """
    if FLAGS.moving_average_decay:
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, tf_global_step)
        variables_to_restore_1 = variable_averages.variables_to_restore(
            slim.get_model_variables())
        variables_to_restore_1[tf_global_step.op.name] = tf_global_step
    else:
        variables_to_restore_1 = slim.get_variables_to_restore()
    variables_to_restore = []
    exclusions = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
    print('Variables to restore:')
    for var in variables_to_restore_1:
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            print(var.op.name)
            variables_to_restore.append(var)
    return variables_to_restore


def _get_image_list(file_image_list,st_img_idx,max_num_images):
    with open(file_image_list) as fid_image_list:
        img_lines = fid_image_list.read().splitlines()
        img_list = []
        for img in img_lines:
            img_list.append(img.split(' ')[0])
    img_list = np.array(img_list)
    if max_num_images<0:
        max_num_images=np.size(img_list)
    ed_img_idx = st_img_idx + max_num_images - 1
    return img_list[st_img_idx:ed_img_idx+1],max_num_images


def _get_images_feed(sess,
                     preprocess_input,
                     preprocess_output,
                     shape,
                     image_dir,
                     images_list):
    images_batch = np.zeros(shape)
    for idx, image_file in enumerate(images_list):
        image_path = os.path.join(image_dir, image_file)
        image_obj = Image.open(image_path)
        image = np.array(image_obj.convert('RGB'))  # Probably for gray scale images
        images_batch[idx, :, :, :] = sess.run(preprocess_output, feed_dict={preprocess_input: image})
    return images_batch


def _add_graph_preprocess(eval_image_size):
    preprocess_input = tf.placeholder(tf.uint8, shape=(None, None, 3), name='PreProcess_Placeholder')
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    preprocess_output = image_preprocessing_fn(preprocess_input, eval_image_size, eval_image_size)
    return preprocess_input, preprocess_output


def _add_graph_inputs(eval_image_size):
    images_input = tf.placeholder(tf.float32, shape=(None, eval_image_size, eval_image_size, 3),
                                  name='ImageInput_Placeholder')
    labels_input = tf.placeholder(tf.float32, [None, FLAGS.num_classes], name='LabelInput_Placeholder')
    return images_input, labels_input


def _add_graph_features(max_num_images,end_points):
    bottleneck_shape_values = str(max_num_images)+',' + FLAGS.bottleneck_shape
    features_dim = [int(val) for val in bottleneck_shape_values.split(',')]
    print('Feature Dim:', features_dim)
    if ':0' in FLAGS.bottleneck_scope:
        features_tensor = tf.get_default_graph().get_tensor_by_name(FLAGS.bottleneck_scope)
    else:
        features_tensor = end_points[FLAGS.bottleneck_scope]
    return features_tensor,features_dim


def _validate_args():
    """
     Checks arguments to run
  """
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')


def _init_graph(sess, tf_global_step, checkpoint_step):
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        if checkpoint_step is not None:
            checkpoint_file = os.path.basename(checkpoint_path).split('-')[0] + '-' + str(checkpoint_step)
            checkpoint_path = os.path.join(FLAGS.checkpoint_path, checkpoint_file)
        else:
            checkpoint_step = int(os.path.basename(checkpoint_path).split('-')[-1])
    else:
        checkpoint_path = FLAGS.checkpoint_path
    tf.logging.info('Evaluating %s' % checkpoint_path)
    saver = tf.train.Saver(_get_variables_to_restore(tf_global_step))
    saver.restore(sess, checkpoint_path)
    return checkpoint_step


def extract():
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False)
        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
        images_input, _ = _add_graph_inputs(eval_image_size)
        preprocess_input, preprocess_output = _add_graph_preprocess(eval_image_size)
        tf_global_step = slim.get_or_create_global_step()
        _, end_points = network_fn(images_input)
        end_points['Predictions'] = loss_ops.add_prediction(FLAGS.loss, end_points['Logits'])
        images_list,max_num_images=_get_image_list(FLAGS.eval_file_image_list, FLAGS.st_img_idx,FLAGS.max_num_images)
        features_tensor, features_dim = _add_graph_features(max_num_images,end_points)
        features = np.zeros(features_dim)
        #h = h5py.File(FLAGS.eval_file_image_features, 'w')
        #dset_f = h.create_dataset('ftr', features_dim, 'f')

        # Execute the graph in a session
        with tf.Session() as sess:
            _init_graph(sess, tf_global_step, None)
            tf.logging.info('global_step: %s' % tf.train.global_step(sess, tf_global_step))
            for i in range(0, max_num_images, FLAGS.batch_size):
                    tf.logging.info('Extracting image: %s' % i)
                    batch_size = min(FLAGS.batch_size,max_num_images-i)
                    images_batch = _get_images_feed(sess, preprocess_input, preprocess_output,
                                                    (batch_size, eval_image_size, eval_image_size, 3),
                                                    FLAGS.dataset_dir,
                                                    images_list[i:i + batch_size])
                    feature_values = sess.run(features_tensor,feed_dict={images_input: images_batch})
                    #dset_f[i:i + batch_size, :] = feature_values
                    features[i:i + batch_size, :] = feature_values
            scipy.io.savemat(FLAGS.eval_file_image_features, mdict={'ftr': features})
            #h.close()

def main(_):
    _validate_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    extract()


if __name__ == '__main__':
    tf.app.run()
