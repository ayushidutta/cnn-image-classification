import tensorflow as tf

### Environment Flags ###

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')  # CHANGE - from 32

tf.app.flags.DEFINE_integer(
    'num_readers', 1,  # CHANGE-from 4
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,  # CHANGE-from 4
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

### Solver Flags ###

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_every_n_steps', 500,
    'The frequency with which the model and summary is saved, in steps, if not using secs flag.')

### Optimization Flags ###

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'momentum',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

### Learning Rate Flags ###

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')#0.01

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.90, 'Learning rate decay factor.') #0.94

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 10.0, #From 2.0
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

### Loss Flags ###
tf.app.flags.DEFINE_string(
    'loss', 'softmax',
    'Specifies which loss to use. One of "softmax", "sigmoid" , "ranking", "WARP", or "LSEP"')

### Dataset Flags ###

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_integer('num_classes', 1001, 'Number of classes for the ImageNet dataset.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'train_file_image_features', None, 'Train image features if using cached nottlenecks.')

tf.app.flags.DEFINE_string(
    'train_file_image_annotations', None, 'Train image annotations if using cached nottlenecks.')

tf.app.flags.DEFINE_string(
    'train_file_image_list', None, 'File with the list of images')

tf.app.flags.DEFINE_string('train_file_image_scores', None,
                           'Train image scores if using cached bottlenecks.')

tf.app.flags.DEFINE_string('train_file_image_neighbours', None,
                           'Train image NN if using cached bottlenecks.')

tf.app.flags.DEFINE_string('eval_file_image_features', None,
                           'Test image features if using cached bottlenecks.')

tf.app.flags.DEFINE_string('eval_file_image_annotations', None,
                           'Test image annotations if using cached nottlenecks.')

tf.app.flags.DEFINE_string(
    'eval_file_image_list', None, 'File with the list of images')

tf.app.flags.DEFINE_string('eval_file_image_scores', None,
                           'Test image scores if using cached bottlenecks.')

tf.app.flags.DEFINE_string('eval_file_image_neighbours', None,
                           'Test image NN if using cached bottlenecks.')

### Evaluation Flags ###

tf.app.flags.DEFINE_integer(
    'max_num_batches', 5, # CHANGE - from None
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_integer(
    'checkpoint_step', None,
    'The latest checkpoint step')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'topK', None,
    'No of top labels predicted')

### Fine-Tuning Flags ###

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'graph_def_type', 'slim', 'Graph definitions type, one of slim/meta/pb')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_boolean(
    'data_augmentation', False, 'If data augmentation, then preprocessing involves multiple crops, '
                                'flip, colour processing etc..')

tf.app.flags.DEFINE_boolean(
    'aux_loss', False, 'Will auxillary loss if present in the net, be trained.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_string(
    'bottleneck_scope', None,
    'Optional. Scope to feed bottlenecks i.e. cached feature values into.')

tf.app.flags.DEFINE_string(
    'bottleneck_shape', None,
    'Optional. Comma separated shape of supplied cached bottlenecks.e.g. 8,1536')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_integer('max_number_of_epochs', 10,
                            'The maximum number of training steps.')

### Graph Node Names, when using train_logits(), or extract() with graph defs ###

tf.app.flags.DEFINE_string(
    'pre_logits', None, 'Pre logits scope name')

tf.app.flags.DEFINE_string(
    'end_points_keys', None, 'End Points Keys in the graph.')

tf.app.flags.DEFINE_string(
    'end_points_scopes', None, 'Scopes in the graph in order of the keys.')

tf.app.flags.DEFINE_string(
    'data_keys', None, 'Data Keys in the graph.')

tf.app.flags.DEFINE_string(
    'data_scopes', None, 'Data Scopes in the graph in order of the keys.')

### Run Options ###
tf.app.flags.DEFINE_string(
    'run_opt', 'train', 'Run options. One of train/extract')

### Extraction Flags only if using extract_cnn() ###

tf.app.flags.DEFINE_integer(
    'st_img_idx', 0, 'The image index to start extracting from')

tf.app.flags.DEFINE_integer(
    'max_num_images', -1,
    'Max number of images to extract. if -1 then consider all images')

FLAGS = tf.app.flags.FLAGS
