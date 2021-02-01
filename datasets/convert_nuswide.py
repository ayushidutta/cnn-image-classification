r"""Converts NUSWIDE data to TFRecords of TF-Example protos.

This module reads the files that make up the NUSWIDE data and creates two
TFRecord datasets: one for train and one for test. Each TFRecord dataset
is comprised of a set of TF-Example protocol buffers, each of which contain
a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import tensorflow as tf
import numpy as np

from PIL import Image
from datasets import dataset_utils

_SPLIT_NAMES = ['train','test']
_DATA_FILENAMES = ['nus1_train_list.txt','nus1_test_list.txt']
_LABEL_FILENAMES = ['nus1_train_annot.txt','nus1_test_annot.txt']
_NUM_PER_SHARD = 50000
_IMAGE_DIRECTORY = '/home/ayushi/Git/dataset/nuswide/images/Flickr'
_TFRECORD_SUB_DIRECTORY='tfrecord'
_CLASSNAMES_FILENAME = 'nus1_dict81.txt'

def _extract_imagelist(filename):
  """Extract the imagelist.

  Args:
    filename: The path to an NUSWIDE images file.

  Returns:
    An image list.
  """
  print('Extracting images from: ', filename)
  imgList = []
  with open(filename) as fImgList:
      imgList1 = fImgList.read().splitlines()
  for x in imgList1:
      img = x.split(' ')[0]
      imgList.append(img)
  return imgList


def _extract_labels(filename):
  """Extract the label list.

  Args:
    filename: The path to an NUSWIDE labels file.

  Returns:
    A label list.
  """
  print('Extracting labels from: ', filename)
  labels = []
  with open(filename) as fAnnot:
    annots=fAnnot.read().splitlines()
  for annotLine in annots:
    label_indicator = map(int,annotLine.strip().split(' '))
    labels.append(label_indicator)
  return labels


def _convert_to_tfrecord(dataset_dir,split_name,data_filename,label_filename):
  """Loads data from the image list files and writes images to a TFRecord.

  Args:
    dataset_dir: The root directory.
    split_name: The name of the train/test split.
    data_filename: The filename of the MNIST images.
    label_filename: The filename of the MNIST labels.
  """
  imageList = _extract_imagelist(data_filename)
  labels = _extract_labels(label_filename)
  num_images=len(imageList)
  assert(len(labels)==num_images)
  num_shards =  int(math.floor(float(num_images)/_NUM_PER_SHARD))
  for i in range(num_shards+1):
    st_idx = i*_NUM_PER_SHARD
    ed_idx = min(st_idx+_NUM_PER_SHARD,num_images)
    output_filename = _get_tfrecord_filename(dataset_dir,split_name,i,num_shards)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for j in range(st_idx,ed_idx):
            sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
            sys.stdout.flush()
            image_path = os.path.join(_IMAGE_DIRECTORY,imageList[j])
            image_string = tf.gfile.FastGFile(image_path, 'rb').read()
            image = Image.open(image_path)
            example = dataset_utils.image_to_tfexample(
                image_string, image.format.encode(), image.size[0], image.size[1], labels[j],image_path, True)
            tfrecord_writer.write(example.SerializeToString())

def _get_tfrecord_filename(dataset_dir, split_name, shard_id, num_shards):
  """Creates the output filename.

  Args:
    dataset_dir: The root directory.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  output_filename = 'nus1_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir,_TFRECORD_SUB_DIRECTORY,output_filename)

def run(dataset_dir,create_label_dict):

  """Runs the conversion operation.

  Args:
    dataset_dir: The root directory.
  """

  if not tf.gfile.Exists(dataset_dir):
    raise ValueError('The dataset directory must exist.')

  # Process for each of the data splits:
  for idx,split in enumerate(_SPLIT_NAMES):
    data_filename = os.path.join(dataset_dir, _DATA_FILENAMES[idx])
    label_filename = os.path.join(dataset_dir, _LABEL_FILENAMES[idx])
    _convert_to_tfrecord(dataset_dir,split,data_filename,label_filename)

  if create_label_dict==True:
      class_filename = os.path.join(dataset_dir,_CLASSNAMES_FILENAME)
      with open(class_filename) as fClassNames:
          class_names = fClassNames.read().splitlines()
      labels_to_class_names = dict(zip(range(len(class_names)), class_names))
      dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the NUSWIDE dataset!')
