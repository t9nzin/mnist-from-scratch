# mnist from scratch

A simple feedforward neural network coded from scratch.

![](https://github.com/t9nzin/mnist-from-scratch/blob/main/demo.gif)

## Description

This project implements a neural network classifier for the MNIST dataset of handwritten digits. 

1. Data Handling (```data.py```): This module is used for loading and preprocessing the MNIST dataset. 
It uses Keras to fetch the data and then normalizes and reshapes it to be used for training the network
or validating.

2. Neural Network (```neural_network.py```): the ```NeuralNetwork``` class is used to create flexible network
objects for a custom number of layers and neurons per layer. The Xavier initialization scheme is used for
initializing the weights. Feedforward, the backpropagation algorithm and stochastic gradient descent are all 
implemented as part of the neural network. Training loops include progress tracking and accuracy reporting.
Saving and loading model weights and biases is also supported.

## Getting Started

### Dependencies

* python 3.9
* h5py 3.11.0
* numpy 1.24.2 
* matplotlib 3.5.1 
* keras 3.3.3 (for downloading mnist data)

### How to Use

1. Clone the repository 
2.
```
python neural_network.py
```

## Authors

 [@t9nzin](https://github.com/t9nzin)

## License

MIT License

## Acknowledgments

This code was written while referencing Michael Nielsen's Neural Networks and 
Deep Learning, a fantastic free online book and resource that did a great job of
explaining the mechanics behind the backpropagation algorithm and neural networks
in general. You can read more about the book here: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
