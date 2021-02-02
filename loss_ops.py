from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

margin = 1.0


def _batch_gather(input, indices):
  """
  output[i, ..., j] = input[i, indices[i, ..., j]]
  """
  shape_output = indices.get_shape().as_list()
  shape_input = input.get_shape().as_list()
  assert len(shape_input) == 2
  batch_base = shape_input[1] * np.arange(shape_input[0])
  batch_base_shape = [1] * len(shape_output)
  batch_base_shape[0] = shape_input[0]

  batch_base = batch_base.reshape(batch_base_shape)
  indices = batch_base + indices

  input = tf.reshape(input, [-1])
  return tf.gather(input, indices)


def _get_M(num_classes):
  alpha = [1./(i+1) for i in range(num_classes)]
  alpha = np.cumsum(alpha)
  return alpha.astype(np.float32)


def _pairwise(label_pairs, logits, num_classes):
  mapped = _batch_gather(logits, label_pairs)
  neg, pos = tf.split(mapped, 2, 2) # Note-Split does not reduce rank.
  delta = neg - pos

  neg_idx, pos_idx = tf.split(label_pairs, 2, 2)
  _, indices = tf.nn.top_k(tf.stop_gradient(logits), num_classes)
  _, ranks = tf.nn.top_k(-indices, num_classes)
  pos_ranks = _batch_gather(ranks, pos_idx)

  weights = _get_M(num_classes)
  pos_weights = tf.gather(weights, pos_ranks)

  delta_nnz = tf.cast(tf.not_equal(neg_idx, pos_idx), tf.float32)
  return delta, delta_nnz, pos_weights


def sigmoid_1(logits, labels):
    tf.logging.info('Logistic Loss')
    return tf.losses.sigmoid_cross_entropy(
        logits=logits, multi_class_labels=labels['label_map'],
        label_smoothing=0.0)


def sigmoid(logits, labels):
    tf.logging.info('Logistic Loss')
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=labels['label_map'])
    reduction = tf.reduce_sum(cross_entropy, 1)
    loss = tf.reduce_mean(reduction, name='sigmoid')
    return loss


def softmax_1(logits, labels):
    """
     Defines the loss function. In case of multi label, instead of one hot
     label, we pass label probabilities with smoothing =0.0 as a hack to
     softmax cross entropy. Alternatively, we can use sigmoid cross entropy.
  """
    tf.logging.info('Softmax Loss')
    return tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=labels['label_prob'],
            label_smoothing=0.0)


def softmax(logits, labels):
    tf.logging.info('Softmax Loss')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels['label_prob'])
    loss = tf.reduce_mean(cross_entropy, name='softmax')
    return loss


def ranking(logits, labels, num_classes, weighted_pairs):
    tf.logging.info('Ranking Loss, weighted= %s' % weighted_pairs)
    delta, delta_nnz, pos_weights = _pairwise(labels['label_pair'], logits, num_classes)
    delta = tf.nn.relu(margin + delta)
    delta *= delta_nnz
    #pairs_per_sample = tf.reduce_sum(delta_nnz,1)
    #max_pairs = tf.reduce_max(pairs_per_sample)
    #w_sample = tf.truediv(max_pairs,pairs_per_sample)
    if weighted_pairs:
        delta *= pos_weights
    reduction = tf.reduce_sum(delta, 1)
    #reduction_2 = reduction * w_sample
    #w_sum = tf.reduce_sum(w_sample)
    #loss = tf.truediv(tf.reduce_sum(reduction_2),w_sum)
    loss = tf.reduce_mean(reduction, name='ranking')
    return loss


def lsep(logits, labels, num_classes, weighted_pairs):
    tf.logging.info('LSEP Loss')
    # compute label pairs
    # batch_size x num_pairs x 2
    delta, delta_nnz, pos_weights = _pairwise(labels['label_pair'], logits, num_classes)
    delta_max = tf.reduce_max(delta, 1, keep_dims=True)
    delta_max_nnz = tf.nn.relu(delta_max)
    inner_exp_diff = tf.exp(delta - delta_max_nnz)
    inner_exp_diff *= delta_nnz
    if weighted_pairs:
        inner_exp_diff *= pos_weights
    inner_sum = tf.reduce_sum(inner_exp_diff, 1, keep_dims=True)
    reduction = delta_max_nnz + tf.log(tf.exp(-delta_max_nnz) + inner_sum)
    loss = tf.reduce_mean(reduction, name='lsep')
    return loss


def add_prediction(loss_name, logits, thresh=0.5):
    if loss_name == 'softmax':
        prediction = tf.nn.softmax(logits, name='Predictions')
    elif loss_name == 'sigmoid':
        prediction = tf.nn.sigmoid(logits, name='Predictions')
    else:
        prediction = logits
    return prediction


def add_loss(num_classes, loss_name, logits, labels, thresh=None):
    if loss_name == 'softmax':
        loss = softmax(logits, labels)
    elif loss_name == 'sigmoid':
        loss = sigmoid(logits, labels)
    elif loss_name == 'ranking':
        loss = ranking(logits, labels, num_classes, False)
    elif loss_name == 'warp':
        loss = ranking(logits, labels, num_classes, True)
    elif loss_name == 'lsep':
        loss = lsep(logits, labels, num_classes, False)
    tf.add_to_collection(tf.GraphKeys.LOSSES,loss)
    return loss


def add_losses(num_classes, loss_name, end_points, labels, aux_loss):
    losses = []
    thresh = 0.5
    if 'Thresh' in end_points:
        thresh = end_points['Thresh']
    with tf.name_scope("loss", values=labels.values() + [end_points['Logits']]): # labels.values() returns a list. So works.
        loss = add_loss(num_classes, loss_name, end_points['Logits'], labels, thresh)
    losses.append(loss)
    # TODO: Auxillary loss to be added with a weighting factor and a scope
    # if 'AuxLogits' in end_points and aux_loss == True:
    #     add_loss(loss_name, end_points['AuxLogits'], labels)
    return losses


