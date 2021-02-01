from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from preprocessing import preprocessing_factory
from classifier.app_config import FLAGS

slim = tf.contrib.slim

MAX_PAIRS = 1000
MAX_NEG_PAIRS = 250
DATA_KEYS = ['image', 'path', 'label_map', 'label_pair', 'label_prob']
LABEL_KEYS = ['label_map', 'label_pair', 'label_prob']

def add_image_queue(dataset, default_image_size):
    """
     Defines inputs to the classifier
  """
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name, is_training=FLAGS.data_augmentation)
    train_image_size = FLAGS.train_image_size or default_image_size
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=4 * FLAGS.batch_size,  # from 20
        common_queue_min=2 * FLAGS.batch_size)  # from 10
    [image, label, path] = provider.get(['image', 'label', 'path'])
    path = tf.Print(path,[path],message='Path',first_n=3)
    image = image_preprocessing_fn(image, train_image_size, train_image_size)
    sample_items = process_data(image, path, label)
    data_batch = tf.train.batch(
        sample_items,
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=4 * FLAGS.batch_size)  # from 5
    batch_queue = slim.prefetch_queue.prefetch_queue(
        data_batch, capacity=2)
    batch_dequeue = batch_queue.dequeue()
    data = {}
    for key, value in zip(DATA_KEYS, batch_dequeue):
        data[key] = value
    return data


def process_data(image, path, label, image_process=False):
    label_map = tf.equal(label, 1)
    label_pairs = _lm2lp(label_map)
    labels = tf.where(label_map)
    random_label = tf.random_shuffle(labels)[0]
    label_map = tf.cast(label_map, tf.float32)
    label_count = tf.reduce_sum(label_map)
    label_prob = tf.truediv(label_map, label_count)
    if image_process:
        image=preprocess_image(image)
    data_values = [image, path, label_map, label_pairs, label_prob, random_label]
    return data_values


def _lm2lp(label_map):
    pos = tf.reshape(tf.where(label_map), [-1])
    neg = tf.reshape(tf.where(tf.logical_not(label_map)), [-1])
    neg_pos = tf.meshgrid(neg, pos, indexing='ij')
    neg_pos_mat = tf.reshape(tf.transpose(tf.stack(neg_pos)), [-1, 2])
    neg_pos_rand = tf.random_shuffle(neg_pos_mat)
    neg_pos_pad = tf.pad(neg_pos_rand, [[0, MAX_PAIRS], [0, 0]]) # In case pairs < max_pairs
    neg_pos_res = tf.slice(neg_pos_pad, [0, 0], [MAX_PAIRS, -1])
    # MAX_PAIRS x 2
    return neg_pos_res


def preprocess_label(label):
    label_map = tf.equal(label, 1)
    label_pairs = _lm2lp(label_map)
    #labels = tf.where(label_map)
    #random_label = tf.random_shuffle(labels)[0]
    label_map = tf.cast(label_map, tf.float32)
    label_count = tf.reduce_sum(label_map)
    label_prob = tf.truediv(label_map, label_count)
    label_data = [label_map, label_pairs, label_prob]
    return label_data


def preprocess_image(image_input,eval_image_size):
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    preprocess_output = image_preprocessing_fn(image_input, eval_image_size, eval_image_size)
    return preprocess_output


def create_images_placeholder(eval_image_size,data):
    images_input = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, eval_image_size, eval_image_size, 3),
                                  name='images/placeholder')
    if data != None:
        data['image'] = images_input
    return images_input


def create_placeholder(inputs_list,**kwargs):
    data = {}
    if 'images' in inputs_list:
        create_images_placeholder(kwargs['eval_image_size'],data)
    if 'labels' in inputs_list:
        create_labels_placeholder(kwargs['num_classes'],data)
    return data


def create_labels_placeholder(num_classes,data):
    label_map=tf.placeholder(tf.float32, [FLAGS.batch_size, num_classes])
    label_pair=tf.placeholder(tf.int64, [FLAGS.batch_size, MAX_PAIRS, 2])
    label_prob=tf.placeholder(tf.float32, [FLAGS.batch_size, num_classes])
    random_label=tf.placeholder(tf.float32, [FLAGS.batch_size, 1])
    if data != None:
        data['label_map']=label_map
        data['label_pair']=label_pair
        data['label_prob']=label_prob
        data['random_label']=random_label
    return label_map,label_pair,label_prob,random_label


def create_image_placeholder():
    image_input = tf.placeholder(tf.uint8, shape=(None, None, 3), name='image')
    return image_input


def create_label_placeholder(num_classes):
    label_input = tf.placeholder(tf.float32, [num_classes], name='label')
    return label_input