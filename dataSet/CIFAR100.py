import cPickle
import numpy as np
import os
import tarfile
from six.moves import urllib

URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
"""Download the cifar 100 dataset."""
if not os.path.exists('cifar-100-python.tar.gz'):
  print("Downloading...")
  urllib.request.urlretrieve(URL, 'cifar-100-python.tar.gz')
if not os.path.exists('cifar-100-python/test_batch'):
  print("Extracting files...")
  tar = tarfile.open('cifar-100-python.tar.gz')
  tar.extractall()
  tar.close()

trainX = np.zeros([50000,32,32,3], dtype=np.uint8)
trainY = np.zeros([50000,100], dtype=np.uint8)
testX  = np.zeros([10000,32,32,3], dtype=np.uint8)
testY  = np.zeros([10000,100], dtype=np.uint8)
#label = ['airplane', 'automoblie','bird','cat','deer', 'dog','frog','horse','ship','truck']

label = ['class_'+str(x+1) for x in xrange(100)]

trainFileName = ['cifar-100-python/train']

testFileName = ['cifar-100-python/test']

index = 0
for name in trainFileName:
    f = open(name,'rb')
    dict = cPickle.load(f)
    f.close()
    #print dict.keys()
    #print set(dict['batch_label']), set(dict['coarse_labels']), set(dict['fine_labels'])
    trainX[index:index + 50000, ...] = dict['data'].reshape([50000, 3, 32, 32]).transpose([0, 2, 3, 1])
    trainY[np.arange(index,index+50000), dict['fine_labels']] = 1
    index += 50000

index = 0
for name in testFileName:
    f = open(name, 'rb')
    dict = cPickle.load(f)
    f.close()
    testX[index:index + 10000, ...] = dict['data'].reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1])
    testY[np.arange(index,index+10000), dict['fine_labels']] = 1
    index += 10000

np.savez('CIFAR100.npz',trainX=trainX,trainY=trainY,testX=testX,testY=testY,label=label)

print 'dataset saved'

