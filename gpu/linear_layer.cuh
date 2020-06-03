#pragma once

// A layer that computes the function W^T X + b.
// Has binary inputs, and weights, and integer biases and outputs.
class LinearLayer {
    void* weights;
    int* biases;
    int* activations;

   public:
    const unsigned int input_chunks;
    const unsigned int input_dims;
    const unsigned int output_dims;

    // NOTE: input_chunks must be a power of two
    LinearLayer(const unsigned int input_chunks, const unsigned int output_dims);
    ~LinearLayer();

    int* forward(void* device_input);
    void set_params(const unsigned char* host_weights, const int* host_biases);
};
