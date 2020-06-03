"""
An inefficient implementation of binary networks with numpy arrays, used for debugging.

This file is intended as the source of truth for the behavior of a binary network.
Other implementations should match this interface and these computations at each step.
"""

from typing import List

import numpy as np


class BinaryNetwork:
    n_layers: int

    def __init__(self, layer_dims: List[int]):
        self.n_layers = len(layer_dims) - 1

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        activations = np.copy(inputs)
        for i, (weights, biases) in enumerate(self.params):
            activations = (weights * 2 - 1).T @ (activations * 2 - 1) + biases

            if i != self.n_layers - 1:  # Hidden layer
                activations = activations > 0

        return activations

    def set_params(self, params):
        self.params = params
