# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


import deep_cnn
import input
import metrics
import os
import numpy


dataset='mnist'
nb_labels=10
data_dir='tmp/mnist'
train_dir='tmp/train_dir'
max_steps=3000
nb_teachers= 2
teacher_id=0

deeper=False

FLAGS = tf.app.flags.FLAGS


def train_teacher(dataset, nb_teachers, teacher_id):
  """
  This function trains a teacher (teacher id) among an ensemble of nb_teachers
  models for the dataset specified.
  :param dataset: string corresponding to dataset (svhn, cifar10)
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # If working directories do not exist, create them
  assert input.create_dir_if_needed(data_dir)
  assert input.create_dir_if_needed(train_dir)

  # Load the dataset
  if dataset == 'svhn':
    train_data,train_labels,test_data,test_labels = input.ld_svhn(extended=True)
  elif dataset == 'cifar10':
    train_data, train_labels, test_data, test_labels = input.ld_cifar10()
  elif dataset == 'mnist':
    train_data, train_labels, test_data, test_labels = input.ld_mnist()
  else:
    print("Check value of dataset flag")
    return False

  # Retrieve subset of data for this teacher
  data, labels = input.partition_dataset(train_data,
                                         train_labels,
                                         nb_teachers,
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path
  if deeper:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt'
  else:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt'
  ckpt_path = train_dir + '/' + str(dataset) + '_' + filename

  # Perform teacher training
  assert deep_cnn.train(data, labels, ckpt_path)

  # Append final step value to checkpoint for evaluation
  ckpt_path_final = ckpt_path + '-' + str(max_steps - 1)

  # Retrieve teacher probability estimates on the test data
  teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)

  # Compute teacher accuracy
  precision = metrics.accuracy(teacher_preds, test_labels)
  print('Precision of teacher after training: ' + str(precision))
  file_handle = open("result_teachers.txt", 'a+')
  file_handle.write(str(teacher_id) + ',' + str(precision) + '\n')
  file_handle.close()

  return True


def main(argv=None):  # pylint: disable=unused-argument
    if os.path.exists("tmp/t10k-images-idx3-ubyte.gz.npy"):
      os.remove("tmp/mnist/t10k-images-idx3-ubyte.gz.npy")
      os.remove("tmp/mnist/t10k-labels-idx1-ubyte.gz.npy")
      os.remove("tmp/mnist/train-images-idx3-ubyte.gz.npy")
      os.remove("tmp/mnist/train-labels-idx1-ubyte.gz.npy")
    for i in numpy.arange(0, 2, 1):
      teacher_id=i
      # Make a call to train_teachers with values specified in flags
      assert train_teacher(dataset, nb_teachers, teacher_id)
      os.remove("tmp/mnist/t10k-images-idx3-ubyte.gz.npy")
      os.remove("tmp/mnist/t10k-labels-idx1-ubyte.gz.npy")
      os.remove("tmp/mnist/train-images-idx3-ubyte.gz.npy")
      os.remove("tmp/mnist/train-labels-idx1-ubyte.gz.npy")

if __name__ == '__main__':
  tf.app.run()
