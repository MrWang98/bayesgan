import json
import time
import numpy as np
import os
import tensorflow as tf

labels_t=[[[1],2,3],[[2],4,5,6],[[3],7,8,9]]
labels=np.array(labels_t)
j_t=np.array(labels)
t_t=np.array(labels)

datafile="test.npy"
if os.path.exists(datafile):
    with tf.gfile.Open(datafile, mode='rb') as file_obj:
        jt,tt=np.load(file_obj)
else:
    t=(j_t,t_t)
    np.save(datafile,t)

print(j_t)