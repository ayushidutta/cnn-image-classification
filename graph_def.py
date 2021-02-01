from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inputs
import tensorflow as tf
import os

from classifier.app_config import FLAGS
from classifier import graph_ops
from nets import nets_factory
from tensorflow.python.platform import gfile


def _import_slim(dataset,add_queue=False):
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=dataset.num_classes,
        weight_decay=FLAGS.weight_decay,
        is_training=True)
    if add_queue:
        data = inputs.add_image_queue(dataset, network_fn.default_image_size)
    else:
        kwargs = {'num_classes': dataset.num_classes}
        data = inputs.create_placeholder(['images','labels'], **kwargs)
    logits, end_points = network_fn(data['image'])
    return logits, end_points, data, network_fn


def import_base(dataset,add_queue=False):
    if FLAGS.graph_def_type == 'slim':
        logits,end_points,data,network_fn = _import_slim(dataset,add_queue)
    return logits, end_points, data, network_fn


#TODO: Fiinish this later
def _import_meta(dataset):
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=dataset.num_classes,
        weight_decay=FLAGS.weight_decay,
        is_training=True)
    checkpoint_path,_ = get_checkpoint_path()
    meta_file = checkpoint_path + '.meta'
    _ = tf.train.import_meta_graph(meta_file)
    end_points = {}
    end_points_keys = FLAGS.end_points_keys.split(',')
    end_points_scopes = FLAGS.end_points_scopes.split(',')
    for idx,key in enumerate(end_points_keys):
        end_points[key] = graph_ops.get_tensor(end_points_scopes[idx])
    data = {}
    data_keys = FLAGS.data_keys.split(',')
    data_scopes = FLAGS.data_scopes.split(',')
    for idx, key in enumerate(data_keys):
        data[key] = graph_ops.get_tensor(data_scopes[idx])
    return end_points['Logits'], end_points, data, network_fn


#TODO: Finish this later
def _import_pb(dataset):
    with tf.Session() as sess:
        with gfile.FastGFile(FLAGS.model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor = (
            # In return elements, jpeg_data_tensor here is necessary, not sure why.
                tf.import_graph_def(graph_def, name='', return_elements=[
                    FLAGS.bottleneck_tensor_name, FLAGS.jpeg_tensor_name]))
    end_points = {}
    end_points['PreLogitsFlatten'] = bottleneck_tensor
    end_points['Logits'] = graph_ops.add_logits(dataset.num_classes,bottleneck_tensor,FLAGS.add_dropout)
    data = {}
    data['image'] = jpeg_data_tensor
    return end_points['Logits'], end_points, data


def get_checkpoint_path(checkpoint_step=None):
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        if checkpoint_path:
            if checkpoint_step is not None:
                checkpoint_file = os.path.basename(checkpoint_path).split('-')[0] + '-' + str(checkpoint_step)
                checkpoint_path = os.path.join(FLAGS.checkpoint_path, checkpoint_file)
            else:
                checkpoint_step = int(os.path.basename(checkpoint_path).split('-')[-1])
    else:
        checkpoint_path = FLAGS.checkpoint_path
    tf.logging.info('Evaluating %s' % checkpoint_path)
    return checkpoint_path,checkpoint_step
