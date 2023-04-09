import numpy as np

class SGD:
    def __init__(self, network, lr=0.0001, reg=0.01, lr_lambda=None):
        '''network: the network to be trained
           lr: learning rate
           lr_lambda: a function that takes in the current epoch and returns the learning rate
        '''
        self.network = network
        self.lr = lr
        self.reg = reg
        self.lr_lambda = lr_lambda if lr_lambda else lambda epoch: 1

    def update(self, epoch):
        '''
           epoch: current epoch
        '''
        lr = self.lr_lambda(epoch) * self.lr
        for key in self.network.params.keys():
            self.network.params[key] -= lr * (self.network.grads[key] + self.reg * self.network.params[key])

    def zero_grad(self):
        for key in self.network.grads.keys():
            self.network.grads[key] = 0

    def get_last_lr(self, epoch):
        return self.lr_lambda(epoch) * self.lr