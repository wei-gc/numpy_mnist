from train import train
import numpy as np
import os
import sys
sys.path.append('./src')
from MLP import MLP_2layer
from SGD import SGD
from get_dataset import load, DataLoader

LR_SHEET = [0.05, 0.03, 0.01, 0.005]
REG_SHEET = [0.1, 0.05, 0.03, 0.01]
HIDDEN_SHEET = [256, 128, 64]


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load()
    train_loader = DataLoader(X_train, Y_train, batch_size=10000, shuffle=True)
    test_loader = DataLoader(X_test, Y_test, batch_size=10000, shuffle=False)

    best_acc = 0
    sweep_count = 0
    for lr in LR_SHEET:
        for reg in REG_SHEET:
            for hidden in HIDDEN_SHEET:
                sweep_count += 1
                print("=========================================")
                print("Sweep: {}, lr: {}, reg: {}, hidden: {}".format(sweep_count, lr, reg, hidden))

                network = MLP_2layer(784, hidden, 10)
                optimizer = SGD(network, lr=lr, reg=reg, lr_lambda=lambda epoch: 0.98**epoch)
                acc_test_avg, train_history = train(network, optimizer, train_loader, test_loader, max_epoch=100, log_steps=20)
                if acc_test_avg > best_acc:
                    best_acc = acc_test_avg
                    best_lr = lr
                    best_reg = reg
                    best_hidden = hidden
                    with open('best_params.txt', 'w') as f:
                        f.write('best_lr: {}, best_reg: {}, best_hidden: {}, bext_acc: {}'.format(best_lr, best_reg, best_hidden, best_acc))