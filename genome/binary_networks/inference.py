"""
Defines a generic interface to efficient implementations of binary networks.

There are 3 main ways to compile a binary network (given by a shape and collection of
binary weights & integer biases):
    - CPU64, which uses uint64 instructions for fast CPU inference, but all
      hidden layers must have exactly 64 units.
    - GPU, which parallelizes inference across many GPU threads with binary arithmetic.
    - Debug, which implements the network in single-threaded numpy.
      It is less efficient but acts as a source of truth for the other implementations.
"""

import enum
from typing import List, Optional, Tuple, Union

import numpy as np

from genome.binary_networks import debug
import cpu_inference
import gpu_inference

# Abstractions for parameters of a discretized but not "inference-ready" model
LayerParams = Tuple[np.ndarray, np.ndarray]  # Weights, biases
NetworkParams = List[LayerParams]

BITS_PER_CHUNK = 32


class InferenceStrategy(enum.Enum):
    """
    Ways of implementing a binary neural network for efficient inference.
    """

    CPU64 = enum.auto()
    GPU = enum.auto()
    DEBUG = enum.auto()


def bits_to_chunks(bits: int) -> int:
    if bits % BITS_PER_CHUNK == 0:
        return bits // BITS_PER_CHUNK
    raise ValueError("GPU inference requires hidden dimensions to be a multiple of 32")


class CompiledNetwork:
    """
    A binary neural network optimized for inference.

    To avoid unnecessary allocations, it's best to use the same CompiledNetwork
    many times and use set_params() to run inference with new networks.

    Parameters
    ----------
    layer_dims : [int]
        A list of all of the widths that will appear in the network. The first element
        is the input dimensionality, and the last is the number of outputs.
    strategy : InferenceStrategy
        What strategy will be used to implement this network.
    """

    strategy: InferenceStrategy
    network: Union[
        debug.BinaryNetwork,
        cpu_inference.BinaryNetwork64,
        gpu_inference.CudaBinaryNetwork,
    ]

    def __init__(self, layer_dims: List[int], strategy: InferenceStrategy):
        self.strategy = strategy
        if strategy == InferenceStrategy.DEBUG:
            self.network = debug.BinaryNetwork(layer_dims)
        elif strategy == InferenceStrategy.CPU64:
            for dim in layer_dims[:-1]:
                if dim != 64:
                    raise ValueError("hidden layers in a CPU64 network must be 64 wide")
            self.network = cpu_inference.BinaryNetwork64(
                layer_dims[-1], len(layer_dims) - 2
            )
        elif strategy == InferenceStrategy.GPU:
            input_chunks = bits_to_chunks(layer_dims[0])
            output_dims = layer_dims[-1]
            hidden_chunks = np.array(
                list(map(bits_to_chunks, layer_dims[1:-1])), dtype=np.uint32
            )
            self.network = gpu_inference.CudaBinaryNetwork.__new__(
                gpu_inference.CudaBinaryNetwork,
                input_chunks,
                output_dims,
                hidden_chunks,
            )
        else:
            raise ValueError("unrecognized inference strategy")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.network.forward(inputs)

    def set_params(self, params: NetworkParams) -> None:
        if self.strategy == InferenceStrategy.GPU:
            for i, (weights, biases) in enumerate(params[:-1]):
                self.network.set_hidden_params(i, weights, biases)
            weights, biases = params[-1]
            self.network.set_output_params(weights, biases)
        else:
            self.network.set_params(params)
