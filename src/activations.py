import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    total = np.sum(np.exp(x), axis=1, keepdims=True)
    out = np.exp(x) / total
    return out