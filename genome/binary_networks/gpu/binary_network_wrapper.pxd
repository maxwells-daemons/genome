from libcpp.vector cimport vector

cdef extern from "binary_network.hpp":
    cdef cppclass BinaryNetwork:
        BinaryNetwork(unsigned int, unsigned int, vector[unsigned int]) except +
        int* forward(const unsigned char* host_input)
        void set_hidden_params(unsigned int index,
                               unsigned char* host_weights,
                               int* host_biases);
        void set_output_params(unsigned char* host_weights, int* host_biases);
