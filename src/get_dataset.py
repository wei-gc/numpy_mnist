######################################################################
#### This file is used to download the MNIST dataset              ####
#### It's adopted from the following link:                        ####
## https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py ##
######################################################################


import numpy as np
from urllib import request
import gzip
import pickle
from random import shuffle

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

class DataLoader:
    def __init__(self, X, Y, batch_size=10000, shuffle=True):
        self.X = X.astype(np.float64) / 255
        self.Y = np.eye(10)[Y]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        assert self.num_samples == Y.shape[0]
        self._batch_sizes = np.arange(batch_size, self.num_samples, batch_size)
        self.indices = np.arange(self.num_samples, dtype=np.int32)

        self.reset()
    
    def reset(self):
        if self.shuffle:
            shuffle(self.indices)
        self.batch_index = np.array_split(self.indices, self._batch_sizes)

    def __iter__(self):
        for index in self.batch_index:
            yield self.X[index], self.Y[index]

if __name__ == '__main__':
    init()