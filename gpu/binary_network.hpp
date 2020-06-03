#pragma once

#include <tuple>
#include <vector>

#include "linear_layer.cuh"
#include "sign_layer.cuh"

/*
 * A binary neural network with a given shape.
 *
 * NOTE: when changing networks, it is much more efficient to set the parameters
 * of each layer than it is to de-allocate and re-allocate the whole network!
 */
class BinaryNetwork {
    const size_t input_bytes;
    const size_t output_bytes;
    void* device_input;
    int* host_output;
    std::vector<SignLayer> hidden_layers;
    LinearLayer output_layer;

   public:
    BinaryNetwork(unsigned int input_chunks, unsigned int output_dims,
                  const std::vector<unsigned int> hidden_chunks);
    ~BinaryNetwork();

    int* forward(const unsigned char* host_input);
    void set_hidden_params(unsigned int index,
                           const unsigned char* host_weights,
                           const int* host_biases);
    void set_output_params(const unsigned char* host_weights,
                           const int* host_biases);
};
