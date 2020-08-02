from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import six
import pickle
import tensorflow as tf


#tf.reset_default_graph()
from imageio import imread

#numpy.array（Image.fromarray（arr）.resize（））
#from scipy.misc import imresize
import scipy.io as sio


import gzip
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated
import aggregation
import random
import deep_cnn

'''
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_path', 'mnist', 'The name of the dataset to use')
tf.flags.DEFINE_string('dataset', 'mnist', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')
tf.flags.DEFINE_integer('nb_teachers', 2, 'Teachers in the ensemble.')
tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')
tf.flags.DEFINE_float('lap_scale', 0.1,
                        'Scale of the Laplacian noise added for privacy')
tf.flags.DEFINE_string('teachers_dir','tmp/train_dir',
                       'Directory where teachers checkpoints are stored.')
tf.flags.DEFINE_integer('teachers_max_steps', 3000,
                        'Number of steps teachers were ran.')
'''

dataset='mnist'
nb_labels=10
nb_teachers=2
deeper=False
lap_scale=0.1
teachers_dir='tmp/train_dir'
teachers_max_steps=3000

def one_hot_encoded(class_numbers, num_classes):
    return np.eye(num_classes, dtype=float)[class_numbers]

#========================移植来的函数=========================
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
  result_shape = (nb_teachers, len(stdnt_data), nb_labels)

  # Create array that will hold result
  result = np.zeros(result_shape, dtype=np.float32)

  # Get predictions from each teacher
  for teacher_id in xrange(nb_teachers):
    # Compute path of checkpoint file for teacher model with ID teacher_id
    if deeper:
      ckpt_path = teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt-' + str(teachers_max_steps - 1) #NOLINT(long-line)
    else:
      ckpt_path = teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt-' + str(teachers_max_steps - 1)  # NOLINT(long-line)

    # Get predictions on our training data and store in result array
    result[teacher_id] = deep_cnn.softmax_preds(stdnt_data, ckpt_path)

    # This can take a while when there are a lot of teachers so output status
    print("Computed Teacher " + str(teacher_id) + " softmax predictions")

  return result
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
@deprecated(None, 'Please use tf.one_hot on tensors.')
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
@deprecated(None, 'Please use tf.data to implement this functionality.')
def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data
def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels
class DataSet(object):
  """Container class for a dataset (deprecated).

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  """

  @deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
              ' from tensorflow/models.')
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
#===========================================================

class AttributeDict(dict):
    print("begin_util")
    print(dict)
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
        
        
def print_images(sampled_images, label, index, directory, save_all_samples=False):
    import matplotlib as mpl
    mpl.use('Agg') # for server side
    import matplotlib.pyplot as plt

    def unnormalize(img, cdim):
        img_out = np.zeros_like(img)
        for i in range(cdim):
            img_out[:, :, i] = 255.* ((img[:, :, i] + 1.) / 2.0)
        img_out = img_out.astype(np.uint8)
        return img_out
        

    if type(sampled_images) == np.ndarray:
        N, h, w, cdim = sampled_images.shape
        idxs = np.random.choice(np.arange(N), size=(5,5), replace=False)
    else:
        sampled_imgs, sampled_probs = sampled_images
        sampled_images = sampled_imgs[sampled_probs.argsort()[::-1]]
        idxs = np.arange(5*5).reshape((5,5))
        N, h, w, cdim = sampled_images.shape

        
    fig, axarr = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            if cdim == 1:
                axarr[i, j].imshow(unnormalize(sampled_images[idxs[i, j]], cdim)[:, :, 0], cmap="gray")
            else:
                axarr[i, j].imshow(unnormalize(sampled_images[idxs[i, j]], cdim))
            axarr[i, j].axis('off')
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_aspect('equal')

    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(os.path.join(directory, "%s_%i.png" % (label, index)), bbox_inches='tight')
    plt.close("all")

    if "raw" not in label.lower() and save_all_samples:
        np.savez_compressed(os.path.join(directory, "samples_%s_%i.npz" % (label, index)),
                            samples=sampled_images)

class FigPrinter():
    
    def __init__(self, subplot_args):
        import matplotlib as mpl
        mpl.use('Agg') # guarantee work on servers
        import matplotlib.pyplot as plt
        self.fig, self.ax_arr = plt.subplots(*subplot_args)
        
    def print_to_file(self, file_name, close_on_exit=True):
        import matplotlib as mpl
        mpl.use('Agg') # guarantee work on servers
        import matplotlib.pyplot as plt
        self.fig.savefig(file_name, bbox_inches='tight')
        if close_on_exit:
            plt.close("all")

class SynthDataset():
    
    def __init__(self, x_dim=100, num_clusters=10, seed=1234):
        
        np.random.seed(seed)
        
        self.x_dim = x_dim
        self.N = 10000
        self.true_z_dim = 2
        # generate synthetic data
        self.Xs = []
        for _ in range(num_clusters):
            cluster_mean = np.random.randn(self.true_z_dim) * 5 # to make them more spread
            A = np.random.randn(self.x_dim, self.true_z_dim) * 5
            X = np.dot(np.random.randn(self.N / num_clusters, self.true_z_dim) + cluster_mean,
                       A.T)
            self.Xs.append(X)
        X_raw = np.concatenate(self.Xs)
        self.X = (X_raw - X_raw.mean(0)) / (X_raw.std(0))
        print (self.X.shape)
        
        
    def next_batch(self, batch_size):

        rand_idx = np.random.choice(range(self.N), size=(batch_size,), replace=False)
        return self.X[rand_idx]

class MnistDataset():
    
    def __init__(self, data_dir, data_share):
        
        from tensorflow.examples.tutorials.mnist import input_data
        #=======================================================
        validation_size = 500
        dtype = dtypes.float32
        reshape = True
        seed = None
        one_hot = False
        data_share=data_share

        source_url='https://storage.googleapis.com/cvdf-datasets/mnist/'
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


        #修改这部分代码,train_image为ndarray(60000,28,28,1),train_labels为ndarray(60000,10)-----------------------------------------------------
        train_images,train_labels=self.ld_data(data_dir,data_share)
        '''读取真实数据的标签
            local_file = base.maybe_download(TRAIN_LABELS, data_dir,
                                             source_url + TRAIN_LABELS)
            with gfile.Open(local_file, 'rb') as f:
                test_data_labels = extract_labels(f, one_hot=one_hot)
        '''
        #----------------------------------------------------------------

        local_file = base.maybe_download(TEST_IMAGES, data_dir,
                                         source_url + TEST_IMAGES)
        with gfile.Open(local_file, 'rb') as f:
            test_images = extract_images(f)

        local_file = base.maybe_download(TEST_LABELS, data_dir,
                                         source_url + TEST_LABELS)
        with gfile.Open(local_file, 'rb') as f:
            test_labels = extract_labels(f, one_hot=one_hot)

        if not 0 <= validation_size <= len(train_images):
            raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                             .format(len(train_images), validation_size))

        validation_images = train_images[:validation_size]
        validation_labels = train_labels[:validation_size]
        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]

        options = dict(dtype=dtype, reshape=reshape, seed=seed)

        train = DataSet(train_images, train_labels, **options)
        validation = DataSet(validation_images, validation_labels, **options)
        test = DataSet(test_images, test_labels, **options)
        #===================================================================================

        #self.mnist = input_data.read_data_sets(data_dir, one_hot=True)
        #下面的一行是上面更改后的
        self.mnist = base.Datasets(train=train, validation=validation, test=test)

        self.x_dim = [28, 28, 1]
        self.num_classes = 10
        self.dataset_size = self.mnist.train._images.shape[0]
        
    def next_batch(self, batch_size, class_id=None):
        
        if class_id is None:
            image_batch, labels = self.mnist.train.next_batch(batch_size)
            new_image_batch = np.array([(image_batch[n]*2. - 1.).reshape((28, 28, 1))
                                        for n in range(image_batch.shape[0])])

            return new_image_batch, labels
        else:
            class_id_batch = np.array([])
            while class_id_batch.shape[0] < batch_size:
                image_batch, labels = self.mnist.train.next_batch(batch_size)
                image_batch = np.array([(image_batch[n]*2. - 1.).reshape((28, 28, 1))
                                       for n in range(image_batch.shape[0])])
                class_id_idx = np.argmax(labels, axis=1) == class_id
                if len(class_id_idx) > 0:
                    if class_id_batch.shape[0] == 0:
                        class_id_batch = image_batch[class_id_idx]
                    else:
                        class_id_batch = np.concatenate([class_id_batch, image_batch[class_id_idx]])
            labels = np.zeros((batch_size, 10))
            labels[:, class_id] = 1.0
            return class_id_batch[:batch_size], labels

    def test_batch(self, batch_size):
        
        image_batch, labels = self.mnist.test.next_batch(batch_size)
        new_image_batch = np.array([(image_batch[n]*2. - 1.).reshape((28, 28, 1))
                                        for n in range(image_batch.shape[0])])
        return new_image_batch, labels

    def get_test_set(self):
        test_imgs = self.mnist.test._images
        test_images = np.array([(test_imgs[n]*2. - 1.).reshape((28, 28, 1))
                                for n in range(test_imgs.shape[0])])
        test_labels = self.mnist.test.labels
        return test_images, test_labels

    def ld_data(self, data_dir, data_share):
        source_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        local_file = base.maybe_download(TRAIN_IMAGES, data_dir,
                                             source_url + TRAIN_IMAGES)
        with gfile.Open(local_file, 'rb') as f:
            test_data = extract_images(f)
            '''读取真实数据的标签
            local_file = base.maybe_download(TRAIN_LABELS, data_dir,
                                             source_url + TRAIN_LABELS)
            with gfile.Open(local_file, 'rb') as f:
                test_data_labels = extract_labels(f, one_hot=one_hot)
            '''
        all_index = [i for i in range(len(test_data))]
        test_index, train_index = getRandomTrain(all_index, data_share)
        train_bool = np.full(len(test_data), False)
        test_bool = np.full(len(test_data), True)
        for i in train_index:
            train_bool[i] = True
            test_bool[i] = False
        train_images = test_data[train_bool]
        teachers_preds = ensemble_preds('mnist', nb_teachers, train_images)
        # 根据教师模型获得暂时的标签，需要对标签进行处理
        train_labels_t, clean_votes, labels_for_dump = aggregation.noisy_max(teachers_preds, lap_scale,
                                                                               return_clean_votes=True)  # NOLINT(long-line)
        # 对标签进行处理
        train_labels=self.deal(train_labels_t)

        return train_images,train_labels

    def deal(self,labels):
        train_labels=[]
        for i in labels:
            tmp=np.full(10,0)
            tmp[i]=1
            train_labels.append(tmp)
        return np.array(train_labels)

class CelebDataset():
        
    def __init__(self, path):
        self.path = path
        self.x_dim = [32, 32, 3]

        with open(os.path.join(path, "Anno/list_attr_celeba.txt")) as af:
            lines = [line.strip() for line in af.readlines()]
    
        self.attr_dict = {}
        for bb_idx, bb_line in enumerate(lines):
            if bb_idx < 2:
                continue
            info = [token for token in bb_line.split(" ") if len(token)]
            self.attr_dict[info[0]] = [int(tk) for tk in info[1:]]

        self.salient_features = [9, 15, 20, 39] # blond, glasses, male, young
        
        self.num_classes = 2**len(self.salient_features)

        self.num_train = 75000
        self.num_test = 10000
        self.dataset_size = self.num_train


    def get_class_id(self, img_name):

        features = self.attr_dict[img_name]
        class_id = 0
        for (sfi, sf) in enumerate(self.salient_features):
            if features[sf] == 1:
                class_id += 2**sfi
        return class_id
            

    def get_batch(self, rand_idx):
        
        new_image_batch = []; new_lbl_batch = []
        for ridx in rand_idx:
            orig_name = "%06d.jpg" % (ridx + 1)
            img_name = "%06d_cropped.jpg" % (ridx + 1)
            img_path = os.path.join(self.path, "img_align_celeba/%s" % img_name)
            if not os.path.exists(img_path):
                continue
            X = imread(img_path)
            Xnorm = np.copy(X).astype(np.float64)
            Xg = np.zeros((X.shape[0], X.shape[1], 1))
            for i in range(3):
                Xnorm[:, :, i] /= 255.0
                Xnorm[:, :, i] = Xnorm[:, :, i] * 2. - 1.
            #Xg[:, :, 0] = 0.2126 * Xnorm[:, :, 0] + 0.7152 * Xnorm[:, :, 1] + 0.0722 * Xnorm[:, :, 2]
            new_image_batch.append(Xnorm)
            #new_image_batch.append(Xg)            

            y = self.get_class_id(orig_name)
            new_lbl_batch.append(y)

        return np.array(new_image_batch), one_hot_encoded(np.array(new_lbl_batch), self.num_classes)

    
    def next_batch(self, batch_size, class_id=None):
        got_batch = False
        while not got_batch:
            rand_idx = np.random.choice(range(self.num_train), size=(2*batch_size,), replace=False)
            X_batch, y_batch = self.get_batch(rand_idx)
            if X_batch.shape[0] >= batch_size:
                got_batch = True
                
        return X_batch[:batch_size], y_batch[:batch_size]
    

    def test_batch(self, batch_size):
        got_batch = False
        while not got_batch:
            rand_idx = np.random.choice(range(self.num_train, self.num_train + self.num_test),
                                        size=(2*batch_size,), replace=False)
            X_batch, y_batch = self.get_batch(rand_idx)
            if X_batch.shape[0] >= batch_size:
                got_batch = True
                
        return X_batch[:batch_size], y_batch[:batch_size]

    def get_test_set(self):
        return self.test_batch(1024*4)


class SVHN():

    def __init__(self, path, subsample=None):

        train_data = sio.loadmat(os.path.join(path, "train_32x32.mat"))
        test_data = sio.loadmat(os.path.join(path, "test_32x32.mat"))

        self.imgs = train_data["X"] / 255.
        self.imgs = self.imgs * 2 - 1.
        self.imgs = np.transpose(self.imgs, [3, 0, 1, 2])

        self.test_imgs = test_data["X"] / 255.
        self.test_imgs = self.test_imgs * 2 - 1.
        self.test_imgs = np.transpose(self.test_imgs, [3, 0, 1, 2])

        self.labels = np.array([yy[0]-1 for yy in train_data["y"]])
        self.labels = one_hot_encoded(self.labels, 10)

        self.test_labels = np.array([yy[0]-1 for yy in test_data["y"]])
        self.test_labels = one_hot_encoded(self.test_labels, 10)

        self.x_dim = [32, 32, 3]
        self.num_classes = 10
        self.dataset_size = self.imgs.shape[0]
        
        if subsample is not None:
            rand_idx = np.random.choice(range(self.imgs.shape[0]), 
                                        size=(int(self.imgs.shape[0]*subsample),), 
                                        replace=False)  
            self.imgs, self.labels = self.imgs[rand_idx], self.labels[rand_idx]

    def next_batch(self, batch_size, class_id=None):
        rand_idx = np.random.choice(range(self.imgs.shape[0]), size=(batch_size,), replace=False)    
        return self.imgs[rand_idx], self.labels[rand_idx]
    

    def test_batch(self, batch_size):
        rand_idx = np.random.choice(range(self.test_imgs.shape[0]),
                                    size=(batch_size,), replace=False)
        return self.test_imgs[rand_idx], self.test_labels[rand_idx]


    
def get_imagenet_val(path, x_dim, subsample=True):
    
    dirnames = [dn for dn in os.listdir(os.path.join(path, "val_256")) if dn[0] == "n"]
    assert len(dirnames), "invalid path %s given!" % (path)
    
    val_imgs = []; val_targets = []; class_dict = {}
    for dir_id, dirname in enumerate(dirnames):
        full_dirname = os.path.join(os.path.join(path, "val_256"), dirname)
        im_names = glob.glob(os.path.join(full_dirname, "*.JPEG"))
        assert len(im_names), "no images in dir %s, fix data" % full_dirname
        for im_file in im_names:
            if subsample and np.random.rand() < 0.8:
                continue
            X = imread(im_file)
            if X.shape != tuple([256, 256, 3]):
                continue
            val_imgs.append(X[None, ::4, ::4, :])
            val_targets.append(dir_id)
        class_dict[dirname] = dir_id
    
    return np.concatenate(val_imgs), np.array(val_targets), class_dict
          
    
class ImageNet():

    def __init__(self, path, num_classes, subsample=None):

        self.path = path
        self.x_dim = [64, 64, 3]
        self.num_classes = num_classes

        self.test_images, self.test_labels, self.class_dict = get_imagenet_val(self.path, self.x_dim)
        assert max(self.class_dict.values()) == self.num_classes - 1
        self.test_imgs = self.test_images / 255.
        self.test_imgs = self.test_imgs * 2 - 1.
        
        self.test_labels = one_hot_encoded(self.test_labels, self.num_classes)

        
    def supervised_batches(self, num_labeled, batch_size):

        print ("generating list of supervised examples")
        dirnames = [dn for dn in os.listdir(os.path.join(self.path, "train_256")) if dn[0] == "n"]
        rand_imgs = []
        while len(rand_imgs) < num_labeled:
            rdir_name = np.random.choice(dirnames)
            rdir = os.path.join(os.path.join(self.path, "train_256"), 
                                rdir_name)
            im_names = glob.glob(os.path.join(rdir, "*.JPEG"))
            assert len(im_names), "no images in dir %s, fix data" % rdir
            rand_im_name = np.random.choice(im_names)
            if rand_im_name not in [x[1] for x in rand_imgs]:
                X = imread(rand_im_name)
                if X.shape != tuple([256, 256, 3]):
                    continue
                rand_imgs.append((rdir_name, rand_im_name))

        num_batches = num_labeled / batch_size

        while True:
            batch_id = np.random.randint(num_batches-1)
            img_batch = rand_imgs[batch_id*batch_size:(batch_id+1)*batch_size]
            batch_imgs = []; batch_lbls = []
            for rdir_name, rand_im_name in img_batch:
                batch_imgs.append(X[None, ::4, ::4, :])
                batch_lbls.append(self.class_dict[rdir_name])

            batch_images = np.concatenate(batch_imgs)
            batch_imgs = batch_images / 255.
            batch_imgs = batch_imgs * 2 - 1.

            yield (batch_imgs, one_hot_encoded(np.array(batch_lbls), self.num_classes))

            
    def next_batch(self, batch_size, class_id=None):

        dirnames = [dn for dn in os.listdir(os.path.join(self.path, "train_256")) if dn[0] == "n"]
        rdir_name = np.random.choice(dirnames)

        batch_imgs, batch_lbls, rand_imgs = [], [], []
        while len(batch_imgs) < batch_size:
            rdir = os.path.join(os.path.join(self.path, "train_256"), 
                                rdir_name)
            im_names = glob.glob(os.path.join(rdir, "*.JPEG"))
            assert len(im_names), "no images in dir %s, fix data" % rdir
            rand_im_name = np.random.choice(im_names)
            if rand_im_name not in rand_imgs:
                X = imread(rand_im_name)
                if X.shape != tuple([256, 256, 3]):
                    continue
                batch_imgs.append(X[None, ::4, ::4, :])
                batch_lbls.append(self.class_dict[rdir_name])
                rand_imgs.append(rand_im_name)
        
        self.batch_images = np.concatenate(batch_imgs)
        self.batch_imgs = self.batch_images / 255.
        self.batch_imgs = self.batch_imgs * 2 - 1.
        
        self.batch_lbls = one_hot_encoded(np.array(batch_lbls),
                                          self.num_classes)

        return self.batch_imgs, self.batch_lbls
    

    
class Cifar10():
    
    def __init__(self, path):
    

        def _convert_images(raw):
            """
            Convert images from the CIFAR-10 format and
            return a 4-dim array with shape: [image_number, height, width, channel]
            where the pixels are floats between -1.0 and 1.0.
            """
            # Convert the raw images from the data-files to floating-points.
            raw_float = (np.array(raw, dtype=float) / 255.0) * 2.0 - 1.0

            # Reshape the array to 4-dimensions.
            images = raw_float.reshape([-1, 3, 32, 32])

            # Reorder the indices of the array.
            images = images.transpose([0, 2, 3, 1])

            return images

        def process_batch(fn):
            fo = open(fn, 'rb')
            data_dict = pickle.load(fo)
            fo.close()
            raw = data_dict["data"]
            images = _convert_images(raw)

            return images, data_dict["labels"]


        def process_meta(mfn):
            # Convert from binary strings.
            fo = open(mfn, 'rb')
            data_dict = pickle.load(fo)
            fo.close()
            raw = data_dict["label_names"]
            names = [x.decode('utf-8') for x in raw]

            return names
                
        
        meta_name = os.path.join(path, 'batches.meta')
        self.class_names = process_meta(meta_name)
        self.num_classes = len(self.class_names)
        
        self.imgs = []
        self.labels = [] 
        for i in range(1, 6):
            batch_name = os.path.join(path, 'data_batch_%i' % i)
            print (batch_name)
            images, labels = process_batch(batch_name)
            self.imgs.append(images)
            self.labels.append(labels)
            
        self.imgs = np.concatenate(self.imgs)
        self.labels = one_hot_encoded(np.concatenate(self.labels), len(self.class_names))

        self.dataset_size = self.imgs.shape[0]
            
        test_batch_name = os.path.join(path, 'test_batch')
        print (test_batch_name)
        self.test_imgs, self.test_labels = process_batch(test_batch_name)
        self.test_labels = one_hot_encoded(self.test_labels, len(self.class_names))
                
        self.x_dim = [32, 32, 3]
        

    def next_batch(self, batch_size, class_id=None):
        rand_idx = np.random.choice(range(self.imgs.shape[0]), size=(batch_size,), replace=False)    
        return self.imgs[rand_idx], self.labels[rand_idx]
    

    def test_batch(self, batch_size):
        rand_idx = np.random.choice(range(self.test_imgs.shape[0]),
                                    size=(batch_size,), replace=False)
        return self.test_imgs[rand_idx], self.test_labels[rand_idx]
    
