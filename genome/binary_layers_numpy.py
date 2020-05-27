"""
An inefficient implementation of binary networks with numpy arrays, used for debugging.
Follows the same interface as `binary_layers.pyx`.

This file is intended as the source of truth for the behavior of a binary network.
Other implementations should match this interface and these computations at each step.
"""


import numpy as np


class BinaryLayer64:
    """
    A layer of a binary neural network, with 64 units.

    This layer has binary weights and integer biases.
    It is intended to be used as a hidden layer.

    Parameters
    ----------
    weights : np.ndarray
        The layer's weights, as a 64x64 ndarray of bools.
    biases : np.ndarray
        The layer's biases, as a 64-vector ndarray of ints.
    """

    weights: np.ndarray
    biases: np.ndarray

    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = (weights * 2 - 1).T
        self.biases = biases

    def forward(self, inputs):
        """
        Compute a forward pass through this layer.

        Computes the function:
            sign(weights^T X inputs + biases).

        Parameters
        ----------
        inputs : np.ndarray
            A 64-vector of inputs to the layer, as an ndarray of bools.

        Returns
        -------
        np.ndarray
            The 64 outputs from this layer, as an ndarray of bools.
        """
        inputs = inputs * 2 - 1
        return (self.weights @ inputs + self.biases) > 0

    def get_params(self):
        """
        Get the parameters of this layer in a Python-readable format.

        Returns
        -------
        weights : np.ndarray[(64, 64), bool]
            The weights of this layer.
        biases : np.ndarray[64, int32]
            The biases of this layer.
        """
        return (self.weights > 0, self.biases)


class LinearLayer64:
    """
    A linear layer for a binary neural network, with 64 units.

    This layer has binary weights and integer biases.
    It is intended to be used as the output layer.

    Parameters
    ----------
    weights : np.ndarray
        The layer's weights, as a 64xM ndarray of bools.
    biases : np.ndarray
        The layer's biases, as an ndarray of M ints.
    """

    def __init__(self, weights, biases):
        self.weights = (weights * 2 - 1).T
        self.biases = biases

    def forward(self, inputs):
        """
        Compute a forward pass through this layer.

        Computes the function:
            weights^T X inputs + biases.

        Parameters
        ----------
        inputs : np.ndarray
            A 64-vector of inputs to the layer, as an ndarray of bools.

        Returns
        -------
        np.ndarray
            The outputs from this layer, as an ndarray of ints.
        """
        inputs = inputs * 2 - 1
        return self.weights @ inputs + self.biases

    def get_params(self):
        """
        Get the parameters of this layer in a Python-readable format.

        Returns
        -------
        weights : np.ndarray[(64, num_outputs), bool]
            The weights of this layer.
        biases : np.ndarray[num_outputs, int32]
            The biases of this layer.
        """
        return (self.weights > 0, self.biases)


class BinaryNetwork64:
    """
    Represents an entire dense binary network, with 64-dimensional activations.

    This network has binary inputs, outputs, and weights, and uses integer biases
    and activations. It uses the sign function as its nonlinearity.

    Parameters
    ----------
    hidden_layers : [BinaryLayer64]
        A list of hidden layers.
    output_layer : LinearLayer64
        A linear output layer.
    """

    def __init__(self, hidden_layers, output_layer):
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def forward(self, inputs):
        """
        Compute a forward pass through the entire model.

        Parameters
        ----------
        inputs : np.ndarray
            An unpacked 64-vector of binary inputs to the model.

        Returns
        -------
        np.ndarray
            An ndarray of the network's outputs.
        """
        for layer in self.hidden_layers:
            inputs = layer.forward(inputs)
        return self.output_layer.forward(inputs).astype("int32")
