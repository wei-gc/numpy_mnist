import numpy as np
import os
from src.activations import sigmoid, relu, softmax

class MLP_2layer:
    def __init__(self, input_size, hidden_size, output_size, act='relu'):
        assert act=='relu'
        act_dict = {'relu': relu, 'sigmoid': sigmoid}
        self.act = act_dict[act]
        self.params = {}
        # initialization
        self._xavier_uniform_init(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        self.part_result = {'z1': None, # z1 = W1x + b1
                            'a1': None, # a1 = relu(z1)
                            'z2': None, # z2 = W2a1 + b2
                            'a2': None, # a2 = softmax(z2)
                            }
        self.grads = {key: None for key in self.params.keys()}

    def _xavier_uniform_init(self, input_size, hidden_size, output_size):
        def _xavier_uniform(fan_in, fan_out):
            low = -np.sqrt(6 / (fan_in + fan_out))
            high = np.sqrt(6 / (fan_in + fan_out))
            return np.random.uniform(low, high, size=(fan_in, fan_out))
        W1 = _xavier_uniform(input_size, hidden_size)
        b1 = np.random.uniform(-1, 1, size=hidden_size)
        W2 = _xavier_uniform(hidden_size, output_size)
        b2 = np.random.uniform(-1, 1, size=output_size)
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

    def forward(self, x):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        z1 = np.dot(x, W1) + b1
        a1 = self.act(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)
        self.part_result['z1'] = z1
        self.part_result['a1'] = a1
        self.part_result['z2'] = z2
        self.part_result['a2'] = a2
        return a2
    
    def backward(self, x, y):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        z1, a1, z2, a2 = self.part_result['z1'], self.part_result['a1'], self.part_result['z2'], self.part_result['a2']
        # compute gradients
        dz2 = 2*(a2 - y)/x.shape[0]
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * (z1 > 0)
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0)
        self.grads['W1'] = dW1
        self.grads['b1'] = db1
        self.grads['W2'] = dW2
        self.grads['b2'] = db2

    def save(self, path='results'):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, 'params.npy'), self.params, allow_pickle=True)

    def load(self, path='results/params.npy'):
        self.params = np.load(path, allow_pickle=True).item()