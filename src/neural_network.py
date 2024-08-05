import math
import sys
import random
import h5py
import numpy as np
import data


def progress(completed, total):
    """
    Display a progress bar in the console

    Parameters
    ----------
    completed : int
        Number of completed batches
    total : int
        Total number of batches in epoch

    Returns
    -------
    None
    """
    filled_length = int(round(20 * completed / float(total)))
    bar = '█' * filled_length + '.' * (20 - filled_length)
    sys.stdout.write(f'\r[{bar}]')
    sys.stdout.flush()


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.biases = [
            np.zeros((self.layers[i], 1))
            for i in range(1, len(self.layers))
        ]
        self.weights = [
            self.xavier_init(layers[i], layers[i + 1])
            for i in range(0, len(layers) - 1)
        ]
        self.batch_errors = [[] for _ in layers]
        self.batch_activations = [[] for _ in layers]
        self.train_accuracy = 0.0
        self.valid_accuracy = 0.0

    def xavier_init(self, n_in, n_out):
        """
        Initializes network weights using
        a Xavier uniform distribution scheme

        Parameters
        ----------
        n_in : int
            Number of neurons in nth layer
        n_out : int
            Number of neurons in (n+1)th layer

        Returns
        -------
        distribution_samples : numpy.ndarray
            Samples from a Xavier uniform distribution
        """
        limit = math.sqrt(6) / math.sqrt(n_in + n_out)
        return np.random.uniform(-limit, limit, (n_out, n_in))

    def feedforward(self, x):
        """
        Input data is passed forward
        through the layers until it
        reaches the output layer

        Parameters
        ----------
        x : numpy.ndarray
            A single training input

        Returns
        -------
        w_sums : list
            List of weighted sums
        """
        w_sums = []
        self.batch_activations[0].append(x)

        for l, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, x) + b
            w_sums.append(z)
            x = self.sigmoid(z)
            self.batch_activations[l+1].append(x)

        return w_sums

    def backprop(self, a_l, y, z):
        """
        Loss is propagated back through
        the network to update the weights
        and biases

        Parameters
        ----------
        a_l : numpy.ndarray
            Activation from output layer
        y : numpy.ndarray
            True label
        z : list
            Weighted sums

        Returns
        -------
        None
        """
        # compute output error
        cost_grad = a_l - y
        sigmoid_grad = self.sigmoid_prime(z[-1])
        error = np.multiply(cost_grad, sigmoid_grad)
        self.batch_errors[-1].append(error)

        # backpropagate the errors
        for l in range(len(self.layers) - 2, 0, -1):
            weight_t = np.transpose(self.weights[l])
            error = np.matmul(weight_t, error)
            sigmoid_grad = self.sigmoid_prime(z[::-1][l])
            error = np.multiply(error, sigmoid_grad)
            self.batch_errors[l].append(error)

    def sgd(self, lr, batch_size):
        """
        Updates the weights and biases

        Parameters
        ----------
        lr : float
            Learning rate
        batch_size : int
            Number of examples in batch

        Returns
        -------
        None
        """
        for l in range(len(self.layers) - 2, -1, -1):
            # compute sum for weights
            weight_sum = np.zeros_like(self.weights[l])
            biases_sum = np.zeros_like(self.biases[l])

            for error, a_l in zip(self.batch_errors[l+1], self.batch_activations[l]):
                product = np.matmul(error, np.transpose(a_l))
                weight_sum = np.add(weight_sum, product)
                biases_sum = np.add(biases_sum, error)

            # update weight & biases
            self.weights[l] -= (lr/batch_size)*weight_sum
            self.biases[l] -= (lr/batch_size)*biases_sum

    def train(self, training_data, epochs, lr, batch_size):
        """
        Trains the neural network

        Parameters
        ----------
        training_data : list
            Data to be trained on
        epochs : int
            Number of epochs to train for
        lr : float
            Learning rate
        batch_size : int
            Number of examples in batch

        Returns
        -------
        None
        """
        for epoch in range(1, epochs + 1):
            random.shuffle(training_data)
            batches = [training_data[i:i + batch_size]
                       for i in range(0, len(training_data), batch_size)]
            print(f"\nEpoch {epoch}/{epochs}")

            batch_accuracies = []

            for i, batch in enumerate(batches):
                correct_predict = 0

                for x, y in batch:
                    w_sums = self.feedforward(x)
                    a_l = self.sigmoid(w_sums[-1])
                    self.backprop(a_l, y, w_sums)

                    # compute accuracy of example
                    prediction = np.argmax(a_l)
                    truth = np.argmax(y)
                    if prediction == truth:
                        correct_predict += 1

                # compute batch accuracy
                batch_accuracy = (correct_predict / len(batch)) * 100
                batch_accuracies.append(batch_accuracy)

                self.sgd(lr, batch_size)
                self.batch_errors = [[] for _ in range(len(self.layers))]
                self.batch_activations = [[] for _ in range(len(self.layers))]

                self.train_accuracy = sum(batch_accuracies) / len(batch_accuracies)

                # show progress and accuracy
                progress(i + 1, len(batches))
                sys.stdout.write(f" - accuracy: {self.train_accuracy:.2f}%")
                sys.stdout.flush()

    def validate(self, data):
        """
        Validates the model accuracy

        Parameters
        ----------
        data : list
            Validation data

        Returns
        -------
        accuracy : float
            Model accuracy
        """
        correct_predict = 0

        for x, y in data:
            predicted, true = self.predict(x, y)
            if predicted == true:
                correct_predict += 1

        accuracy = (correct_predict / len(data)) * 100
        self.valid_accuracy = accuracy
        return accuracy

    def predict(self, x, y):
        """
        Predict the class of a single input example

        Parameters
        ----------
        x : numpy.ndarray
            The input example
        y : numpy.ndarray
            The true label (used for comparison)

        Returns
        ----------
        (prediction, y) : tuple
            Model prediction and true label
        """
        for w, b in zip(self.weights, self.biases):
            x = self.sigmoid(np.dot(w, x) + b)

        prediction = np.argmax(x)
        return prediction, y

    def sigmoid(self, z):
        """
        Computes the sigmoid function

        Parameters
        ----------
        z : numpy.ndarray
            The weighted sum of an example

        Returns
        ----------
        activation : numpy.ndarray
            Each element of z transformed by sigmoid
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        """
        Computes the sigmoid prime function

        Parameters
        ----------
        z : numpy.ndarray
            The weighted sum of an example

        Returns
        ----------
        activation prime : numpy.ndarray
            Each element of z transformed by sigmoid prime
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def load(self, file_name):
        """
        Loads saved weights & biases
        into a neural network object

        Parameters
        ----------
        file_name : string
            file to load from

        Returns
        ----------
        None
        """
        h5 = h5py.File(file_name, 'r')

        for i in range(len(self.layers) - 1):
            self.weights[i] = np.array(h5[f'weights_{i}'])
            self.biases[i] = np.array(h5[f'biases_{i}'])

        h5.close()

    def save(self, file_name):
        """
        Saves the network's weights
        & biases to a given file

        Parameters
        ----------
        file_name : string
            file to save to

        Returns
        ----------
        None
        """
        with h5py.File(file_name, 'w') as f:
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                f.create_dataset(f'weights_{i}', data=w)
                f.create_dataset(f'biases_{i}', data=b)


def main():
    training_data, testing_data = data.load()
    nn = NeuralNetwork([784, 128, 10])
    nn.train(training_data, 30, 3.0, 100)
    nn.save("model_w&b")

if __name__ == "__main__":
    main()
