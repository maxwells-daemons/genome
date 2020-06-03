#pragma once

// A layer that computes the function sign(W^T X + b).
// Has binary inputs, activations, and weights, and integer biases.
class SignLayer {
    void* weights;
    int* thresholds;  // (biases + num_outputs) / 2
    void* activations;

   public:
    const unsigned int input_chunks;
    const unsigned int output_chunks;
    const unsigned int input_dims;
    const unsigned int output_dims;

    // NOTE: input_chunks must be a power of two
    SignLayer(const unsigned int input_chunks,
                const unsigned int output_chunks);
    ~SignLayer();

    void* forward(void* device_input);
    void set_params(const unsigned char* host_weights, const int* host_biases);
};
