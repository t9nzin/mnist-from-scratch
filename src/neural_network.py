import math
import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        for i in range(0, len(layers) - 1):
            self.weights.append(self.xavier_init(layers[i], layers[i+1]))
        self.biases = [np.zeros(self.layers[i]) for i in range(1, len(layers))]

    def xavier_init(self, n_in, n_out):
        limit = math.sqrt(6) / math.sqrt(n_in + n_out)
        return np.random.uniform(-limit, limit, (n_in, n_out))

    def feedforward(self, input_data):
        pass

    def backpropagation(self, input_data, target):
        pass

    def compute_cost(self, output, target):
        pass

    def update_mini_batch(self, mini_batch, learning_rate):
        pass

    def train(self, training_data, epochs, mini_batch_size, learning_rate):
        pass

    def predict(self, input_data):
        pass

    def accuracy(self, test_data):
        pass

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))