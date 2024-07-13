import numpy as np
from matplotlib import pyplot
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras

MAPPING = {0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}


def load():
    """
    Loads the MNIST dataset

    Parameters
    ----------
    None

    Returns
    ----------
    training_data, testing_data : tuple
        MNIST dataset split into training and testing data
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x inputs
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = [np.reshape(x, (784, 1)) for x in x_train]
    y_train = [np.array(MAPPING[i]).reshape((10, 1)) for i in y_train]

    x_test = [np.reshape(x, (784, 1)) for x in x_test]
    y_test = [np.array(MAPPING[i]).reshape((10, 1)) for i in y_test]

    training_data = list(zip(x_train, y_train))
    testing_data = list(zip(x_test, y_test))

    return training_data, testing_data


def show(x):
    """
    Displays an image of a
    given training example

    Parameters
    ----------
    x : numpy.ndarray
        A training example

    Returns
    ----------
    None
    """
    image = np.reshape(x, (28, 28))

    pyplot.imshow(image)
    pyplot.show()
