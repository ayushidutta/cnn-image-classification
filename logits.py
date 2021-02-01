from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py
import scipy.io
import tensorflow as tf
import math
import inputs
import os
import graph_def

from datasets import dataset_factory
from classifier import graph_ops
from classifier import loss_ops
from classifier.annotation import annotateTopK, annotate_by_probability
from eval.multilabel_metrics1 import evaluate_per_row_col_metrics
from classifier.app_config import FLAGS

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('num_fc', 0, 'No of FC layer')
tf.app.flags.DEFINE_boolean('use_weight_decay', True, 'use_weight_decay')
tf.app.flags.DEFINE_boolean('use_batch_norm', False, 'use_batch_norm')
tf.app.flags.DEFINE_boolean('use_pre_batch_norm', True, 'use_pre_batch_norm')
tf.app.flags.DEFINE_integer('topK', 3, 'topK')
tf.app.flags.DEFINE_integer('num_classes', 81, 'num_classe')

def _init_graph(sess, global_step=None):
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    var_to_restore = None
    tf.logging.info('Init Session!')
    # Latest Checkpoint: Restore all variables
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        var_to_restore = tf.global_variables()
        tf.logging.info('Var restored from latest ckpt: %s' % var_to_restore)
    # Pretrained graph. Restore only Model Variables
    elif FLAGS.checkpoint_path != None:
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        tf.logging.info('Checkpoint: %s' % checkpoint_path)
        var_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if var_to_restore != None:
        tf.logging.info('Restoring Variables!')
        saver = tf.train.Saver(var_to_restore)
        saver.restore(sess, checkpoint_path)
    if global_step:
        return sess.run(global_step)


def _eval_predictions(prediction_scores, images_annotations, topK):
    predicted_annot = annotateTopK(prediction_scores, topK)
    resI, resL = evaluate_per_row_col_metrics(images_annotations, prediction_scores, predicted_annot, topK, True)
    perf = {
        'acc_top1': resI['acc_top1'],
        'acc_topK': resI['acc_topK'],
        'f1_image': resI['f1'],
        'f1_label': resL['f1']
    }
    tf.logging.info('Performance : %s' % perf)
    return perf


def _eval_predictions2(prediction_scores, images_annotations, topK):
    predicted_annot = annotateTopK(prediction_scores, topK)
    resI, resL = evaluate_per_row_col_metrics(images_annotations, prediction_scores, predicted_annot, topK, True)
    perf = {
        'acc_top1': resI['acc_top1'],
        'acc_topK': resI['acc_topK'],
        'f1_label': resL['f1'],
        'prec_label': resL['prec'],
        'rec_label': resL['rec']
    }
    #tf.logging.info('Performance : %s' % perf)
    tf.logging.info('Performance: %s,%s,%s,%s,%s' % (resL['prec'],resL['rec'],resL['f1'],resL['map'],resL['nplus']))
    tf.logging.info('Performance: %s,%s,%s,%s' % (resI['prec'],resI['rec'],resI['f1'],resI['map']))
    return perf

def _eval_confidence(sigmoid_predictions, images_annotations, topK):
    predicted_prob = annotate_by_probability(sigmoid_predictions)
    resI, resL = evaluate_per_row_col_metrics(images_annotations, sigmoid_predictions, predicted_prob, topK, True)
    perf = {'acc_top1': resI['acc_top1'], 'acc_topK': resI['acc_topK'],
            'f1_label': resL['f1'], 'prec_label': resL['prec'],
            'rec_label': resL['rec']}
    tf.logging.info('Confidence >0.5 Performance : %s' % perf)


def _cache_bottlenecks(file_image_features, file_image_annotations):
    tf.logging.info('Reading %s' % file_image_features)
    with open(file_image_annotations) as fid_image_annotations:
        img_annot_lines = fid_image_annotations.read().splitlines()
    n_imgs = len(img_annot_lines)
    n_labels = len(img_annot_lines[0].strip().split(' '))
    tf.logging.info('No. of imgs:%d No. of labels:%d' % (n_imgs,n_labels))
    img_annots = np.zeros([n_imgs, n_labels])
    for idx, annot_line in enumerate(img_annot_lines):
        annot_val = [float(val) for val in annot_line.strip().split(' ')]
        img_annots[idx, :] = annot_val
    image_db = h5py.File(file_image_features, 'r')
    return image_db['ftr'], img_annots, n_imgs


def _get_random_cached_bottlenecks(image_db, image_gt, num_images, ftr_dim, batch_size):
    select_images = np.random.randint(0, num_images, batch_size)
    ground_truth = image_gt[select_images, :]
    bottlenecks = np.zeros([batch_size, ftr_dim])
    for i, img_idx in enumerate(select_images):
        while np.sum(image_gt[img_idx, :]) == 0:
            img_idx = np.random.choice(num_images, 1)
        ground_truth[i,:] = image_gt[img_idx, :]
        bottlenecks[i, :] = image_db[:, img_idx].T
    return bottlenecks, ground_truth


def _get_cached_bottlenecks_by_diversity(image_db, img_annots, ftr_dim, how_many_imgs):
    """
        TODO : Check its correctness
        Returns the images to use for evaluation, such that all labels are evaluated for model correctness
    :return:
    """
    n_labels = np.shape(img_annots)[1]
    sel_min_label_freq = int(math.ceil(how_many_imgs / n_labels)) - 1
    sorted_labels = np.argsort(np.sum(img_annots, axis=0))
    sel_images_idx = []
    sel_images_annots = np.zeros([how_many_imgs, n_labels], dtype=np.int32)
    sel_label_freq = np.zeros([n_labels, 1], dtype=np.int32)
    cnt_sel_imgs = 0
    while cnt_sel_imgs < how_many_imgs:
        sel_min_label_freq = sel_min_label_freq + 1
        for i in range(n_labels):
            if how_many_imgs <= cnt_sel_imgs:
                break
            label_imgs_idx = np.argwhere(img_annots[:, sorted_labels[i]])
            label_imgs_idx = np.setdiff1d(label_imgs_idx, sel_images_idx, assume_unique=True)
            n_select = max(0, sel_min_label_freq - sel_label_freq[sorted_labels[i]])
            n_select = min(len(label_imgs_idx), n_select, how_many_imgs - cnt_sel_imgs)
            if n_select > 0:
                sel_label_imgs = np.random.choice(label_imgs_idx, n_select, replace=False)
                for sel_img in sel_label_imgs:
                    sel_images_idx.append(sel_img)
                    sel_images_annots[cnt_sel_imgs, :] = img_annots[sel_img, :]
                    sel_labels = np.argwhere(img_annots[sel_img, :])
                    sel_label_freq[sel_labels] = sel_label_freq[sel_labels] + 1
                    cnt_sel_imgs = cnt_sel_imgs + 1
    bottlenecks = np.zeros([how_many_imgs, ftr_dim])
    for i, img_idx in enumerate(sel_images_idx):
        bottlenecks[i, :] = image_db[:, img_idx].T
    return bottlenecks, sel_images_annots


def _get_labels_feed(sess, label_input, label_process, labels):
    batch_size = np.shape(labels)[0]
    labels_data = {}
    for key in inputs.LABEL_KEYS:
        labels_data[key] = []
    for idx in range(batch_size):
        label_data = sess.run(label_process, feed_dict={
            label_input: labels[idx, :]
        })
        for i, key in enumerate(inputs.LABEL_KEYS):
            labels_data[key].append(label_data[i])
    return labels_data


def _get_data_feed(pre_logits, data, pre_logits_val, data_values):
    data_feed = {
        pre_logits: pre_logits_val
    }
    for key in data_values:
        data_feed[data[key]] = data_values[key]
    return data_feed


def _batch_norm(x, mode=True, name=None):
    return tf.contrib.layers.batch_norm(inputs=x,
                                        decay=0.997,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=mode,
                                        updates_collections=None,
                                        scope=(name + '_batch_norm'))


def _add_logits(num_classes, pre_logits, is_training=False):
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    if FLAGS.use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    if FLAGS.use_weight_decay:
        weights_regularizer = slim.l2_regularizer(FLAGS.weight_decay)
    else:
        weights_regularizer = None
    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params):
        if FLAGS.use_pre_batch_norm:
            net = _batch_norm(pre_logits, mode=is_training, name='prelogit')
        else:
            net = pre_logits
        if FLAGS.num_fc == 1:
            net = slim.fully_connected(net, 2048, activation_fn=tf.nn.relu, scope='fc_1')
        #net = tf.nn.dropout(net, 0.8) if is_training else net
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      normalizer_fn=None, normalizer_params={}, scope='Logits')
    return logits


def _train():
    with tf.Graph().as_default():
        image_train_db, image_train_gt, n_train = _cache_bottlenecks(FLAGS.train_file_image_features,
                                                                     FLAGS.train_file_image_annotations)
        image_test_db, image_test_gt, n_test= _cache_bottlenecks(FLAGS.eval_file_image_features,
                                                                  FLAGS.eval_file_image_annotations)
        dataset = {}
        dataset['num_classes'] = np.size(image_train_gt,1)
        dataset['num_samples'] = n_train
        kwargs = {'num_classes': dataset['num_classes']}
        data = inputs.create_placeholder(['labels'], **kwargs)
        ftr_dim = int(FLAGS.bottleneck_shape)
        pre_logits = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, ftr_dim))
        end_points = {}
        end_points['Logits'] = _add_logits(dataset['num_classes'], pre_logits, True)
        _ = loss_ops.add_losses(dataset['num_classes'], FLAGS.loss, end_points, data, False)
        train_op, total_loss = graph_ops.add_logits_training(dataset, None)
        end_points['Predictions'] = loss_ops.add_prediction(FLAGS.loss, end_points['Logits'])
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        tf.logging.info('Summaries: %s' % summaries)
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        label_input = inputs.create_label_placeholder(dataset['num_classes'])
        label_process = inputs.preprocess_label(label_input)
        number_of_steps = int(FLAGS.max_number_of_epochs * math.ceil(dataset['num_samples'] / FLAGS.batch_size))
        saver = tf.train.Saver()
        model_path = os.path.join(FLAGS.train_dir, 'model')
        num_eval = int(FLAGS.batch_size * math.ceil(500 / FLAGS.batch_size))
        tf.logging.info('Start training!')
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            _init_graph(sess)
            for i in range(number_of_steps):
                train_ftrs, train_labels = _get_random_cached_bottlenecks(image_train_db,
                                                                          image_train_gt,
                                                                          n_train,
                                                                          ftr_dim,
                                                                          FLAGS.batch_size)
                labels_data = _get_labels_feed(sess, label_input, label_process, train_labels)
                feed_data = _get_data_feed(pre_logits, data, train_ftrs, labels_data)
                sess.run(train_op, feed_dict=feed_data)
                if i % FLAGS.log_every_n_steps == 0:
                    curr_loss = sess.run(total_loss, feed_dict=feed_data)
                    tf.logging.info('Step: %d Loss:%.4f' % (i, curr_loss))
                if i % FLAGS.save_every_n_steps == 0:
                    summary, train_scores = sess.run([summary_op, end_points['Predictions']], feed_dict=feed_data)
                    train_writer.add_summary(summary, i)
                    saver.save(sess, model_path, global_step=i)
                    _eval_predictions(train_scores, train_labels, FLAGS.topK)
                    test_ftrs, test_labels = _get_cached_bottlenecks_by_diversity(image_test_db,
                                                                                  image_test_gt,
                                                                                  ftr_dim,
                                                                                  num_eval)
                    test_scores = np.zeros([num_eval, dataset['num_classes']])
                    for j in range(0, num_eval, FLAGS.batch_size):
                        test_scores[j:j + FLAGS.batch_size, :] = sess.run(end_points['Predictions'],
                                                                          feed_dict={pre_logits: test_ftrs[j:j + FLAGS.batch_size, :]})
                    _eval_predictions(test_scores, test_labels, FLAGS.topK)
                    #_eval_confidence(test_scores, test_labels, 3)


def _extract():
    image_test_db, test_labels, n_test = _cache_bottlenecks(FLAGS.eval_file_image_features,
                                                            FLAGS.eval_file_image_annotations)
    dataset = {}
    dataset['num_classes'] = np.size(test_labels, 1)
    ftr_dim = int(FLAGS.bottleneck_shape)
    pre_logits = tf.placeholder(tf.float32, shape=(None, ftr_dim))
    end_points = {}
    end_points['Logits'] = _add_logits(dataset['num_classes'], pre_logits, False)
    end_points['Predictions'] = loss_ops.add_prediction(FLAGS.loss, end_points['Logits'])
    test_scores = np.zeros([n_test, dataset['num_classes']])
    test_prob=None
    with tf.Session() as sess:
        _init_graph(sess)
        for j in range(0, n_test, FLAGS.batch_size):
            batch_remain = min(FLAGS.batch_size, n_test - j)
            tf.logging.info('Extracting %d - %d' % (j, j + batch_remain))
            test_ftrs = image_test_db[:, j:j + batch_remain].T
            test_scores[j:j + batch_remain, :] = sess.run(end_points['Logits'],
                                                          feed_dict={pre_logits: test_ftrs})
        if 'sigmoid' in FLAGS.loss:
            test_prob = sess.run(tf.nn.sigmoid(test_scores))
            _eval_confidence(test_prob, test_labels, FLAGS.topK)
    _eval_predictions2(test_scores, test_labels, FLAGS.topK)
    scipy.io.savemat(FLAGS.eval_file_image_scores, mdict={'testScores': test_scores,
                                                          'testProb':test_prob})


def main(_):
    if FLAGS.run_opt == 'train':
        _train()
    elif FLAGS.run_opt == 'extract':
        _extract()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
