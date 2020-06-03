# cython: language_level=3, boundscheck=False, initializedcheck=False
import numpy as np

cimport numpy as np
from libcpp.vector cimport vector
from binary_network_wrapper cimport BinaryNetwork

cdef class CudaBinaryNetwork:
    cdef BinaryNetwork* _cuda_network
    cdef unsigned int output_dims

    def __cinit__(self,
                  unsigned int input_chunks,
                  unsigned int output_dims,
                  unsigned int[:] hidden_chunks):
        cdef vector[unsigned int] cpp_hidden_chunks
        for chunk in hidden_chunks:
            cpp_hidden_chunks.push_back(chunk)
        self._cuda_network = new BinaryNetwork(input_chunks,
                                               output_dims,
                                               cpp_hidden_chunks)
        self.output_dims = output_dims

    def __dealloc__(self):
        del self._cuda_network

    cpdef np.ndarray[int, ndim=1] forward(self, np.npy_bool[:] inputs):
        cdef const unsigned char[:] packed = np.packbits(inputs, bitorder='little')
        cdef np.ndarray[int, ndim=1] result = np.empty(self.output_dims, dtype=np.int32)
        cdef int* result_ptr = self._cuda_network.forward(&packed[0])
        for i in range(self.output_dims):
            result[i] = result_ptr[i]
        return result

    def set_hidden_params(self, index, np.npy_bool[:, :] weights, int[:] biases):
        cdef np.ndarray[unsigned char, ndim=1] flat = np.packbits(weights,
                                                                  bitorder='little')
        self._cuda_network.set_hidden_params(index, &flat[0], &biases[0])

    def set_output_params(self, np.npy_bool[:, :] weights, int[:] biases):
        cdef np.ndarray[unsigned char, ndim=1] flat = np.packbits(weights,
                                                                  bitorder='little')
        self._cuda_network.set_output_params(&flat[0], &biases[0])
