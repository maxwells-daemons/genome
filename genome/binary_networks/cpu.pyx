# cython: language_level=3, boundscheck=False, initializedcheck=False
"""
Provides efficient CPU implementations of binary neural networks and layers.
These rely on bit arithmetic on uint64_t and so can only have inputs and hidden layers
of width 64.

BinaryLayer64 and LinearLayer64 both conform to the same general interface, but
are provided separately for efficiency (to avoid dispatch overhead, etc).
They should be accessed in Python through duck typing.
"""

import numpy as np

cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdint cimport uint64_t, uint16_t

# See: https://gist.github.com/craffel/e470421958cad33df550
cdef extern int __builtin_popcountll(unsigned long long) nogil

cdef uint64_t pack_bits_64(np.npy_bool[:] bits) nogil:
    """
    Pack a vector of 64 bits into a uint64_t.

    Given a memoryview containing at least 64 bits, return a single uint64_t
    with those bits set.
    """
    cdef uint64_t result = 0
    for i in range(64):
        result <<= 1
        result |= bits[i]

    return result

cdef np.ndarray[np.npy_bool, ndim=1] unpack_bits_64(uint64_t packed):
    """
    Unpack a vector of 64 bits stored as a uint64_t into a numpy array.
    """
    cdef np.ndarray[np.npy_bool, ndim=1] result = np.ones(64, dtype="bool")
    for i in range(63, -1, -1):
        result[i] = packed & 0b1
        packed >>= 1
    return result

cpdef uint64_t concat_16bit(uint16_t a, uint16_t b, uint16_t c, uint16_t d):
    """
    Concatenate the bit-representations of four uint16_t into a uint64_t.
    """
    # Somehow this code block is really pretty
    return (
        <uint64_t> a
        | (<uint64_t> b) << 16
        | (<uint64_t> c) << 32
        | (<uint64_t> d) << 48
    )


cdef class BinaryLayer64:
    """
    A layer of a binary neural network, with 64 units.

    This layer has binary weights and integer biases.
    It is intended to be used as a hidden layer.

    Notes
    -----
    Computes the activation with this formula:
    - num_negative_bits = popcount(inputs XOR weights)
    - dot_product = num_positive_bits - num_negative_bits
                  = 64 - 2 * num_negative_bits
    - sign(dot_product + biases) = sign(64 - 2 * num_negative_bits + biases)
                                 = sign(biases/2 + 32 - num_negative_bits)
    """

    cdef uint64_t weights[64]
    cdef int[:] biases
    cdef int thresholds[64]

    def __cinit__(self):
        pass

    cpdef uint64_t forward(self, const uint64_t inputs):
        """
        Compute a forward pass through this layer.

        Computes the function:
            sign(weights^T X inputs + biases).

        Parameters
        ----------
        inputs : const uint64_t
            A packed 64-vector of inputs to this layer.

        Returns
        -------
        uint64_t
            A packed 64-vector of outputs from this layer.
        """
        # Explicitly inline bit-packing for efficiency
        cdef uint64_t result = 0

        for i in range(64):
            result <<= 1
            result |= <int> __builtin_popcountll(inputs ^ self.weights[i]) < \
                self.thresholds[i]

        return result

    def set_params(self, tuple params):
        weights, biases = params
        self.biases = biases

        for i in range(64):
            self.weights[i] = pack_bits_64(weights[:, i])
            self.thresholds[i] = (biases[i] >> 1) + 32


cdef class LinearLayer64:
    """
    A linear layer for a binary neural network, with 64 units.

    This layer has binary weights and integer biases.
    It is intended to be used as the output layer.

    Parameters
    ----------
    weights : bool[64, n_outputs]
        The layer's weights, as a memoryview-compatible array of 64xM bits.
    biases : int[n_outputs]
        The layer's biases, strored as a memoryview-compatible array of M ints.

    Notes
    -----
    Computes the activation with this formula:
    - num_negative_bits = popcount(inputs XOR weights)
    - dot_product = num_positive_bits - num_negative_bits
                  = 64 - 2 * num_negative_bits
    - dot_product + biases = (biases + 64) - (2 * num_negative_bits)
    """

    cdef uint64_t[:] weights
    cdef int[:] biases
    cdef unsigned int num_outputs

    def __cinit__(self, unsigned int num_outputs):
        self.num_outputs = num_outputs
        self.weights = np.empty(num_outputs, dtype=np.uint64)
        self.biases = np.empty(num_outputs, dtype=np.int32)

    cdef np.ndarray[int, ndim=1] forward(self, const uint64_t inputs):
        """
        Compute a forward pass through this layer.

        Computes the function:
            weights^T X inputs + biases.

        Parameters
        ----------
        inputs : const uint64_t
            A packed 64-vector of inputs to this layer.

        Returns
        -------
        uint64_t
            An unpacked vector of integers from this layer.
        """
        cdef np.ndarray[int, ndim=1] result = np.empty(self.num_outputs, dtype=np.int32)
        for i in range(self.num_outputs):
            result[i] = self.biases[i] - \
                (<int> __builtin_popcountll(inputs ^ self.weights[i]) << 1)

        return result

    def set_params(self, tuple params):
        weights, biases = params

        for i in range(self.num_outputs):
            self.weights[i] = pack_bits_64(weights[:, i])

            # Precompute offset biases; see `forward` for derivation
            self.biases[i] = biases[i] + 64


cdef class BinaryNetwork64:
    """
    Represents an entire dense binary network, with 64-dimensional activations.

    This network has binary inputs, outputs, and weights, and uses integer biases
    and activations. It uses the sign function as its nonlinearity.
    """

    cdef readonly unsigned int num_hidden_layers
    cdef readonly list hidden_layers
    cdef readonly LinearLayer64 output_layer

    def __cinit__(self, unsigned int output_dims, unsigned int num_hidden_layers):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = [BinaryLayer64.__new__(BinaryLayer64)
                              for _ in range(num_hidden_layers)]
        self.output_layer = LinearLayer64.__new__(LinearLayer64, output_dims)

    cpdef np.ndarray[int, ndim=1] forward_raw(self, uint64_t inputs):
        """
        Compute a forward pass through the entire model, on pre-packed inputs.

        Parameters
        ----------
        inputs : uint64_t
            A packed 64-vector of binary inputs to the model.

        Returns
        -------
        np.ndarray[int, num_outputs]
            An ndarray of the network's outputs.
        """
        # Workaround to call functions without Python lookup overhead.
        # See: https://stackoverflow.com/questions/31119510/cython-have-sequence-of-
        #      extension-types-as-attribute-of-another-extension-type
        cdef BinaryLayer64 layer

        for i in range(self.num_hidden_layers):
            layer = self.hidden_layers[i]
            inputs = layer.forward(inputs)
        return self.output_layer.forward(inputs)

    cpdef np.ndarray[int, ndim=1] forward(self, np.npy_bool[:] inputs):
        """
        Compute a forward pass through the entire model.

        Parameters
        ----------
        inputs : np.ndarray[64, bool]
            An unpacked 64-vector of binary inputs to the model.

        Returns
        -------
        np.ndarray[int, num_outputs]
            An ndarray of the network's outputs.
        """
        return self.forward_raw(pack_bits_64(inputs))

    def set_params(self, list params):
        for (layer, param) in zip(self.hidden_layers, params[:-1]):
            layer.set_params(param)
        self.output_layer.set_params(params[-1])
