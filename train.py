import numpy as np
import os
import sys
sys.path.append('./src')
from MLP import MLP_2layer
from SGD import SGD
from get_dataset import load, DataLoader

def train(network, optimizer, 
          train_loader, test_loader,
          max_epoch=100,
          log_steps=1):
    train_loss_history = []
    test_loss_history = []
    test_acc_history = []
    for epoch in range(max_epoch):
        train_loader.reset()
        loss_epoch = []
        for x, y in train_loader:
            y_pred = network.forward(x)
            assert y.shape == y_pred.shape
            optimizer.zero_grad()
            network.backward(x, y)
            optimizer.update(epoch)
            # calculate loss and log it
            loss = np.sum((y - y_pred)**2)
            loss_epoch.append(loss)
        loss_train_avg = np.sum(loss_epoch) / train_loader.num_samples
        train_loss_history.append(loss_train_avg)

        if (epoch+1) % log_steps == 0 or epoch == max_epoch-1:
            loss_test = []
            acc_test = []
            for x, y in test_loader:
                y_pred = network.forward(x)
                loss = np.sum((y - y_pred)**2)
                loss_test.append(loss)
                acc_test.append(sum(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)))
            loss_test_avg = np.sum(loss_test) / test_loader.num_samples
            acc_test_avg = np.sum(acc_test) / test_loader.num_samples
            lr = optimizer.get_last_lr(epoch)
            print('epoch: {}, \tloss_train: {:.4f}, \tloss_test: {:.4f}, \tacc_test: {:.4f}, \tlr: {:.4f}'.format(
                epoch, loss_train_avg, loss_test_avg, acc_test_avg, lr)
            )
            test_loss_history.append(loss_test_avg)
            test_acc_history.append(acc_test_avg)

    train_history = {'train_loss': train_loss_history, 'test_loss': test_loss_history, 'test_acc': test_acc_history}
    return acc_test_avg, train_history


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load()
    train_loader = DataLoader(X_train, Y_train, batch_size=10000, shuffle=True)
    test_loader = DataLoader(X_test, Y_test, batch_size=10000, shuffle=False)

    network = MLP_2layer(784, 256, 10)
    optimizer = SGD(network, lr=0.03, reg=0.01, lr_lambda=lambda epoch: 0.98**epoch)
    acc_test_avg, train_history = train(network, optimizer, train_loader, test_loader, max_epoch=150, log_steps=1)
    # save results
    if not os.path.exists('results'):
        os.makedirs('results')
    np.save('results/train_history.npy', train_history, allow_pickle=True)
    network.save('results')
