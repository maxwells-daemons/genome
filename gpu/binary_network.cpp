#include "binary_network.hpp"

#include "common.hpp"

BinaryNetwork::BinaryNetwork(unsigned int input_chunks,
                             unsigned int output_dims,
                             const std::vector<unsigned int> hidden_chunks)
    : input_bytes(input_chunks * CHUNK_BYTE),
      output_bytes(output_dims * sizeof(int)),
      hidden_layers(std::vector<SignLayer>()),
      output_layer(LinearLayer(hidden_chunks.back(), output_dims)) {
    // Necessary to prevent moving hidden layers
    hidden_layers.reserve(hidden_chunks.size());
    for (unsigned int activation_chunks : hidden_chunks) {
        hidden_layers.emplace_back(input_chunks, activation_chunks);
        input_chunks = activation_chunks;
    }

    gpuErrChk(cudaMalloc(&device_input, input_bytes));
    host_output = (int*)malloc(output_bytes);
}

BinaryNetwork::~BinaryNetwork() {
    gpuErrChk(cudaFree(device_input));
    free(host_output);
}

// TODO: remove prints
int* BinaryNetwork::forward(const unsigned char* host_input) {
    cudaMemcpy(device_input, (const void*)host_input, input_bytes,
               cudaMemcpyHostToDevice);
    void* device_activation = device_input;
    for (SignLayer& layer : hidden_layers) {
        device_activation = layer.forward(device_activation);
    }
    device_activation = output_layer.forward(device_activation);
    cudaMemcpy(host_output, device_activation, output_bytes,
               cudaMemcpyDeviceToHost);
    return host_output;
}

void BinaryNetwork::set_hidden_params(unsigned int index,
                                      const unsigned char* host_weights,
                                      const int* host_biases) {
    hidden_layers[index].set_params(host_weights, host_biases);
}

void BinaryNetwork::set_output_params(const unsigned char* host_weights,
                                      const int* host_biases) {
    output_layer.set_params(host_weights, host_biases);
}
