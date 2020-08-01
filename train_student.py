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

import numpy as np
from six.moves import xrange
import tensorflow as tf

import aggregation
import deep_cnn
import input
import metrics

import os
import random
from PIL import Image
from array import *
from random import shuffle

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('dataset', 'mnist', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 20, 'Number of output classes')

tf.flags.DEFINE_string('data_dir','tmp','Temporary storage')
tf.flags.DEFINE_string('train_dir','tmp/train_dir','Where model chkpt are saved')
tf.flags.DEFINE_string('teachers_dir','tmp/train_dir',
                       'Directory where teachers checkpoints are stored.')

tf.flags.DEFINE_integer('teachers_max_steps', 3000,
                        'Number of steps teachers were ran.')
tf.flags.DEFINE_integer('max_steps', 3000, 'Number of steps to run student.')
tf.flags.DEFINE_integer('nb_teachers', 20, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('stdnt_share', 1000,
                        'Student share (last index) of the test data')
tf.flags.DEFINE_float('lap_scale', 0.01,
                        'Scale of the Laplacian noise added for privacy')
tf.flags.DEFINE_boolean('save_labels', False,
                        'Dump numpy arrays of labels and clean teacher votes')
tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')


#train和test中真实数据的个数==================================================
tf.flags.DEFINE_integer('nb_train', 1000,
                        'The number of train data')
tf.flags.DEFINE_integer('nb_test', 1000,
                        'The number of test data')
#==========================================================================


#随机取样的函数
def getRandomTrain(indexs, num_train):
  TrainIndex = []
  i = 0
  length = len(indexs)
  while i < num_train:
    tmp = random.sample(indexs, 1)
    if tmp[0] not in TrainIndex:
      TrainIndex.append(tmp[0])
      indexs.remove(tmp[0])
      i = i + 1
    else:
      continue
  return indexs, TrainIndex

#读取gan图片数据
def LoadImages(filename, image_size, pixel_depth):
  # Load from and save to
  name = filename
  if not os.path.exists(name + "_data"):
    data_image = array('B')
    data_label = array('B')

    FileList = []
    for dirname in os.listdir(name):  # [1:] Excludes .DS_Store from Mac OS
      path = os.path.join(name, dirname)
      for filename in os.listdir(path):
        if filename.endswith(".png"):
          FileList.append(os.path.join(name, dirname, filename))

    shuffle(FileList)  # Usefull for further segmenting the validation set

    for filename in FileList:
      label = int(filename.split('\\')[1])
      Im = Image.open(filename)
      Im = Im.convert("L")
      pixel = Im.load()
      width, height = Im.size

      for x in range(0, width):
        for y in range(0, height):
          temp = pixel[y, x]
          data_image.append(temp)
      data_label.append(label)  # labels start (one unsigned byte each)
    data_image = np.frombuffer(data_image, dtype=np.uint8).astype(np.float32)
    data_image = (data_image - (pixel_depth / 2.0)) / pixel_depth
    data_image = data_image.reshape(len(FileList), image_size, image_size, 1)
    data_label = np.frombuffer(data_label, dtype=np.uint8).astype(np.int32)

    dir = name + "_data"
    os.mkdir(dir)
    imagepath = os.path.join(dir, "image.npy")
    np.save(imagepath, data_image)
    labelpath = os.path.join(dir, "label.npy")
    np.save(labelpath, data_label)

  else:
    dir = name + "_data"
    imagepath = os.path.join(dir, "image.npy")
    data_image = np.load(imagepath)
    labelpath = os.path.join(dir, "label.npy")
    data_label = np.load(labelpath)

  return data_image, data_label


def ensemble_preds(dataset, nb_teachers, stdnt_data):
  """
  Given a dataset, a number of teachers, and some input data, this helper
  function queries each teacher for predictions on the data and returns
  all predictions in a single array. (That can then be aggregated into
  one single prediction per input using aggregation.py (cf. function
  prepare_student_data() below)
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param stdnt_data: unlabeled student training data
  :return: 3d array (teacher id, sample id, probability per class)
  """

  # Compute shape of array that will hold probabilities produced by each
  # teacher, for each training point, and each output class
  result_shape = (nb_teachers, len(stdnt_data), FLAGS.nb_labels)

  # Create array that will hold result
  result = np.zeros(result_shape, dtype=np.float32)

  # Get predictions from each teacher
  for teacher_id in xrange(nb_teachers):
    # Compute path of checkpoint file for teacher model with ID teacher_id
    if FLAGS.deeper:
      ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt-' + str(FLAGS.teachers_max_steps - 1) #NOLINT(long-line)
    else:
      ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt-' + str(FLAGS.teachers_max_steps - 1)  # NOLINT(long-line)

    # Get predictions on our training data and store in result array
    result[teacher_id] = deep_cnn.softmax_preds(stdnt_data, ckpt_path)

    # This can take a while when there are a lot of teachers so output status
    print("Computed Teacher " + str(teacher_id) + " softmax predictions")

  return result


def prepare_student_data(dataset, nb_teachers, save=False):
  """
  Takes a dataset name and the size of the teacher ensemble and prepares
  training data for the student model, according to parameters indicated
  in flags above.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param save: if set to True, will dump student training labels predicted by
               the ensemble of teachers (with Laplacian noise) as npy files.
               It also dumps the clean votes for each class (without noise) and
               the labels assigned by teachers
  :return: pairs of (data, labels) to be used for student training and testing
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Load the dataset
  '''if dataset == 'svhn':
    test_data, test_labels = input.ld_svhn(test_only=True)
  elif dataset == 'cifar10':
    test_data, test_labels = input.ld_cifar10(test_only=True)
  elif dataset == 'mnist':
    test_data, test_labels = input.ld_mnist(test_only=True)
  else:
    print("Check value of dataset flag")
    return False'''

  #=============================================================================
  #读入gan数据
  test_data_gan,test_labels_gan=LoadImages('training-images',28,1)
  #读入nb_train个真实数据
  testdata_true,testlabels_true=input.ld_mnist(test_only=True)
  all_index = [i for i in range(len(testdata_true))]
  test_index, train_index = getRandomTrain(all_index, FLAGS.nb_train)
  train_bool = np.full(len(testdata_true), False)
  test_bool = np.full(len(testdata_true), True)
  for i in train_index:
    train_bool[i] = True
    test_bool[i] = False
  test_data_true=testdata_true[train_bool]
  test_labels_true=testlabels_true[train_bool]
  #==================================================================================



  # Make sure there is data leftover to be used as a test set
  assert FLAGS.stdnt_share < len(test_data_gan)

  # Prepare [unlabeled] student training data (subset of test set)
  '''stdnt_data = test_data[:FLAGS.stdnt_share]'''



  # =====================================================
  #gan与真实数据组合
  stdnt_data=np.vstack((test_data_gan,test_data_true))
  #=========================================================================




  # Compute teacher predictions for student training data
  teachers_preds = ensemble_preds(dataset, nb_teachers, stdnt_data)

  # Aggregate teacher predictions to get student training labels
  if not save:
    stdnt_labels = aggregation.noisy_max(teachers_preds, FLAGS.lap_scale)
  else:
    # Request clean votes and clean labels as well
    stdnt_labels, clean_votes, labels_for_dump = aggregation.noisy_max(teachers_preds, FLAGS.lap_scale, return_clean_votes=True) #NOLINT(long-line)

    # Prepare filepath for numpy dump of clean votes
    filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_clean_votes_lap_' + str(FLAGS.lap_scale) + '.npy'  # NOLINT(long-line)
    # Prepare filepath for numpy dump of clean labels
    filepath_labels = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_teachers_labels_lap_' + str(FLAGS.lap_scale) + '.npy'  # NOLINT(long-line)
    # Dump clean_votes array
    with tf.gfile.Open(filepath, mode='w') as file_obj:
      np.save(file_obj, clean_votes)

    # Dump labels_for_dump array
    with tf.gfile.Open(filepath_labels, mode='w') as file_obj:
      np.save(file_obj, labels_for_dump)



  #随机取样=======================================================
  #gan与真实数据组合
  testlabels = np.hstack((test_labels_gan,test_labels_true))
  #教师模型预测和实际比较
  ac_ag_labels = metrics.accuracy(stdnt_labels, testlabels)
  print("Accuracy of the aggregated labels: " + str(ac_ag_labels))
  #从除去nb_train个真实数据里选出nb_test个测试
  test_index2, train_index2 = getRandomTrain(test_index, FLAGS.nb_test)
  train_bool2 = np.full(len(testdata_true), False)
  test_bool2 = np.full(len(testdata_true), True)
  for i in train_index2:
    train_bool2[i] = True
    test_bool2[i] = False
  stdnt_test_data=testdata_true[train_bool2]
  stdnt_test_labels=testlabels_true[train_bool2]
  #==============================================================



  '''stdnt_test_data=np.ndarray(shape=(9000,),dtype=float)
  stdnt_test_labels=np.ndarray(shape=(9000,),dtype=int)
  for i in test_index:
    stdnt_test_data.append(test_data[i])
    stdnt_test_labels.append(test_labels[i])


  # Print accuracy of aggregated labels
  ac_ag_labels = metrics.accuracy(stdnt_labels, test_labels[:FLAGS.stdnt_share])
  print("Accuracy of the aggregated labels: " + str(ac_ag_labels))

  # Store unused part of test set for use as a test set after student training
  stdnt_test_data = test_data[FLAGS.stdnt_share:]
  stdnt_test_labels = test_labels[FLAGS.stdnt_share:]'''

  if save:
    # Prepare filepath for numpy dump of labels produced by noisy aggregation
    filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_labels_lap_' + str(FLAGS.lap_scale) + '.npy' #NOLINT(long-line)
    # Dump student noisy labels array
    with tf.gfile.Open(filepath, mode='w') as file_obj:
      np.save(file_obj, stdnt_labels)

  return stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels


def train_student(dataset, nb_teachers):
  """
  This function trains a student using predictions made by an ensemble of
  teachers. The student and teacher models are trained using the same
  neural network architecture.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :return: True if student training went well
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Call helper function to prepare student data using teacher predictions
  stdnt_dataset = prepare_student_data(dataset, nb_teachers, save=True)

  # Unpack the student dataset
  stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = stdnt_dataset

  # Prepare checkpoint filename and path
  if FLAGS.deeper:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student_deeper.ckpt' #NOLINT(long-line)
  else:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student.ckpt'  # NOLINT(long-line)

  # Start student training
  assert deep_cnn.train(stdnt_data, stdnt_labels, ckpt_path)

  # Compute final checkpoint name for student (with max number of steps)
  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

  # Compute student label predictions on remaining chunk of test set
  student_preds = deep_cnn.softmax_preds(stdnt_test_data, ckpt_path_final)

  # Compute teacher accuracy
  precision = metrics.accuracy(student_preds, stdnt_test_labels)
  print('Precision of student after training: ' + str(precision))
  file_handle = open("result_student.txt", 'a+')
  file_handle.write(str(FLAGS.lap_scale) + ',' + str(precision) + '\n')
  file_handle.close()
  return True

def main(argv=None): # pylint: disable=unused-argument
  # Run student training according to values specified in flags
  assert train_student(FLAGS.dataset, FLAGS.nb_teachers)

if __name__ == '__main__':
  tf.app.run()
