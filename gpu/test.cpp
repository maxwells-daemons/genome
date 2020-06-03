#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "binary_network.hpp"
#include "common.hpp"
#include "linear_layer.cuh"
#include "sign_layer.cuh"

#define INPUT_CHUNKS 128
#define OUTPUT_CHUNKS 4
#define NET_OUTPUT_DIMS 1
#define REPEATS 10000

using namespace std;

void test_sign_layer() {
    cout << "SIGN LAYER TEST" << endl;

    size_t weight_bytes = INPUT_CHUNKS * OUTPUT_CHUNKS * CHUNK_BIT * CHUNK_BYTE;
    unsigned char* weights = (unsigned char*)malloc(weight_bytes);
    std::fill(weights, weights + weight_bytes, 0);

    size_t num_biases = OUTPUT_CHUNKS * CHUNK_BIT;
    int* biases = (int*)malloc(num_biases * sizeof(int));
    std::fill(biases, biases + num_biases, 0);

    size_t input_bytes = INPUT_CHUNKS * CHUNK_BYTE;
    unsigned char* inputs = (unsigned char*)malloc(input_bytes);
    std::fill(inputs, inputs + input_bytes, 0);

    void* dev_inputs;
    cudaMalloc(&dev_inputs, INPUT_CHUNKS * CHUNK_BYTE);

    cout << "Creating layer..." << endl;
    SignLayer layer = SignLayer(INPUT_CHUNKS, OUTPUT_CHUNKS);
    cout << "Setting params..." << endl;
    layer.set_params(weights, biases);
    cout << "Moving inputs to device..." << endl;
    cudaMemcpy(dev_inputs, inputs, INPUT_CHUNKS * CHUNK_BYTE,
               cudaMemcpyHostToDevice);
    cout << "Computing " << REPEATS << " forward passes..." << endl;
    for (int i = 0; i < REPEATS; i++) {
        layer.forward(dev_inputs);
    }
    cout << "Done!" << endl;

    free(weights);
    free(biases);
    free(inputs);
    cudaFree(dev_inputs);
}

void test_linear_layer() {
    cout << "LINEAR LAYER TEST" << endl;

    size_t weight_bytes = INPUT_CHUNKS * OUTPUT_CHUNKS * CHUNK_BIT * CHUNK_BYTE;
    unsigned char* weights = (unsigned char*)malloc(weight_bytes);
    std::fill(weights, weights + weight_bytes, 0);

    size_t num_biases = OUTPUT_CHUNKS * CHUNK_BIT;
    int* biases = (int*)malloc(num_biases * sizeof(int));
    std::fill(biases, biases + num_biases, 0);

    size_t input_bytes = INPUT_CHUNKS * CHUNK_BYTE;
    unsigned char* inputs = (unsigned char*)malloc(input_bytes);
    std::fill(inputs, inputs + input_bytes, 0);

    void* dev_inputs;
    cudaMalloc(&dev_inputs, INPUT_CHUNKS * CHUNK_BYTE);

    cout << "Creating layer..." << endl;
    LinearLayer layer = LinearLayer(INPUT_CHUNKS, OUTPUT_CHUNKS);
    cout << "Setting params..." << endl;
    layer.set_params(weights, biases);
    cout << "Moving inputs to device..." << endl;
    cudaMemcpy(dev_inputs, inputs, INPUT_CHUNKS * CHUNK_BYTE,
               cudaMemcpyHostToDevice);
    cout << "Computing " << REPEATS << " forward passes..." << endl;
    for (int i = 0; i < REPEATS; i++) {
        layer.forward(dev_inputs);
    }
    cout << "Done!" << endl;

    free(weights);
    free(biases);
    free(inputs);
    cudaFree(dev_inputs);
}

void test_binary_network() {
    cout << "BINARY NETWORK TEST" << endl;

    size_t input_bytes = INPUT_CHUNKS * CHUNK_BYTE;
    unsigned char* inputs = (unsigned char*)malloc(input_bytes);
    std::fill(inputs, inputs + input_bytes, 0);

    cout << "Creating network..." << endl;
    std::vector<unsigned int> layer_sizes{128, 128};
    BinaryNetwork network =
        BinaryNetwork(INPUT_CHUNKS, NET_OUTPUT_DIMS, layer_sizes);
    cout << "Setting params..." << endl;
    // TODO
    cout << "Computing " << REPEATS << " forward passes..." << endl;
    for (int i = 0; i < REPEATS; i++) {
        network.forward(inputs);
    }
    cout << "Done!" << endl;

    free(inputs);
}

int main(int argc, char* argv[]) {
    /* test_sign_layer(); */
    /* test_linear_layer(); */
    test_binary_network();
}
