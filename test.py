from train import train
import numpy as np
import os
import sys
sys.path.append('./src')
from MLP import MLP_2layer
from SGD import SGD
from get_dataset import load, DataLoader



if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load()
    train_loader = DataLoader(X_train, Y_train, batch_size=10000, shuffle=True)
    test_loader = DataLoader(X_test, Y_test, batch_size=10000, shuffle=False)

    network = MLP_2layer(784, 256, 10)
    network.load('results/params.npy')

    loss_epoch = []
    for x, y in train_loader:
        y_pred = network.forward(x)
        assert y.shape == y_pred.shape
        # calculate loss and log it
        loss = np.sum((y - y_pred)**2)
        loss_epoch.append(loss)
    loss_train_avg = np.sum(loss_epoch) / train_loader.num_samples


    loss_test = []
    acc_test = []
    for x, y in test_loader:
        y_pred = network.forward(x)
        loss = np.sum((y - y_pred)**2)
        loss_test.append(loss)
        acc_test.append(sum(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)))
    loss_test_avg = np.sum(loss_test) / test_loader.num_samples
    acc_test_avg = np.sum(acc_test) / test_loader.num_samples
    
    print('loss_train: {:.4f}, \tloss_test: {:.4f}, \tacc_test: {:.4f}'.format(
        loss_train_avg, loss_test_avg, acc_test_avg)
    )
