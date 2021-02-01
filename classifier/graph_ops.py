from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from classifier.app_config import FLAGS

slim = tf.contrib.slim


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _get_loss_gradients(optimizer,
                        regularization_losses=None,
                        **kwargs):
    all_losses = []
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    tf.logging.info('Losses: %s' % losses)
    if regularization_losses is None:
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
    sum_loss = tf.add_n(losses)
    all_losses.append(sum_loss)
    regularization_loss = tf.add_n(regularization_losses,
                                   name='regularization_loss')
    all_losses.append(regularization_loss)
    if sum_loss is not None:
        tf.summary.scalar('sum_loss', sum_loss)
    if regularization_loss is not None:
        tf.summary.scalar('regularization_loss', regularization_loss)
    if all_losses:
        total_loss = tf.add_n(all_losses)
    if total_loss is not None:
        grads_and_vars = optimizer.compute_gradients(total_loss, **kwargs)
    return total_loss, grads_and_vars


def _get_loss(loss_scope):
    all_losses = []
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    tf.logging.info('Losses: %s' % losses)
    regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES, loss_scope)
    tf.logging.info('Reg Losses: %s' % regularization_losses)
    if losses:
        sum_loss = tf.add_n(losses)
        all_losses.append(sum_loss)
        tf.summary.scalar('sum_loss', sum_loss)
    if regularization_losses:
        regularization_loss = tf.add_n(regularization_losses,
                                       name='regularization_loss')
        all_losses.append(regularization_loss)
        tf.summary.scalar('regularization_loss', regularization_loss)
    if all_losses:
        total_loss = tf.add_n(all_losses)
    return total_loss


def _get_variables_to_train():
    """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def add_net_summaries(end_points):
    """
      Adds summaries to the graph
  """
    for end_point in end_points:
        x = end_points[end_point]
        tf.summary.histogram('activations/' + end_point, x)
        tf.summary.scalar('sparsity/' + end_point,
                          tf.nn.zero_fraction(x))
    for loss in tf.get_collection(tf.GraphKeys.LOSSES):
        tf.summary.scalar('losses/%s' % loss.op.name, loss)
    for variable in slim.get_model_variables():
        tf.summary.histogram(variable.op.name, variable)
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    return summary_op


def add_training(dataset):
    """
     Defines the classifier training with optimizer, summaries etc..
  """
    global_step = slim.create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Configure the moving averages
    if FLAGS.moving_average_decay:
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
    else:
        moving_average_variables, variable_averages = None, None

    # Configure the optimization procedure.
    learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
    optimizer = _configure_optimizer(learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    if FLAGS.moving_average_decay:
        update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()
    tf.logging.info('Variables to train %s' % variables_to_train)
    # Returns a train_tensor and summary_op
    total_loss, gradients = _get_loss_gradients(
        optimizer,
        var_list=variables_to_train)
    tf.summary.scalar('total_loss', total_loss)

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')
    return train_tensor, total_loss


def add_logits_training(dataset,loss_scope):
    global_step = slim.create_global_step()
    learning_rate = _configure_learning_rate(dataset['num_samples'], global_step)
    #learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
    optimizer = _configure_optimizer(learning_rate)
    tf.logging.info('Learning Rate: %s' % learning_rate)
    learning_rate = tf.Print(learning_rate,[learning_rate],message='lr is:')
    tf.summary.scalar('learning_rate', learning_rate)
    total_loss = _get_loss(loss_scope)
    tf.summary.scalar('total_loss', total_loss)
    train_op = optimizer.minimize(total_loss,global_step=global_step)
    return train_op,total_loss


def add_evaluation_step(prediction, labels):
    correct_prediction = tf.equal(
         tf.argmax(prediction, 1), tf.argmax(labels['label_map'], 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    tf.summary.scalar('accuracy/top_1', evaluation_step)
    return evaluation_step


def _batch_norm(x, mode=True, name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=mode,
                                            updates_collections=None,
                                            scope=(name + '_batch_norm'))

def add_logits(num_classes, pre_logits, dropout=0.2, weight_decay=0.00004):
    tf.logging.info('Dropout,%f Weight decay:%f' % (dropout,weight_decay))
    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        features = _batch_norm(pre_logits, mode=dropout > 0, name='prelogit')
        #features = slim.fully_connected(pre_logits, 2048, activation_fn=tf.nn.relu,
        #                           scope='thresh-h1')
        features = tf.nn.dropout(features, 1 - dropout) if dropout > 0 else features
        logits = slim.fully_connected(features, num_classes, activation_fn=None,
                                      scope='Logits')
    return logits


def get_tensor(scope,end_points=None):
    if ':0' in scope:
        features_tensor = tf.get_default_graph().get_tensor_by_name(scope)
    else:
        features_tensor = end_points[scope]
    features_dim = features_tensor.get_shape().as_list()
    return features_tensor,features_dim