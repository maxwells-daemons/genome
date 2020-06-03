#include <stdexcept>

#include "common.hpp"
#include "linear_layer.cuh"

/*
 * Kernel to compute activations <- weights^T inputs + b.
 *
 * Each block i creates a single output value by computing W[:, i] . X + b[i].
 * Each thread in a block is responsible for one or more "chunks" of 4 bytes
 * of input and weight data.
 *
 * Preconditions:
 *  - Dynamically allocated one unsigned int of shared space per thread.
 *  - inputs contains (CHUNK_BYTE * n_chunks) bytes of input.
 *  - weights contains (CHUNK_BYTE * n_chunks * blockDim.x) bytes of weights.
 *  - biases contains blockDim.x integers.
 *  - activations has space for blockDim.x * sizeof(int) bytes of output.
 */
__global__ void linear_forward(const unsigned int n_chunks,
                               const unsigned int* inputs,
                               const unsigned int* weights, const int* biases,
                               int* activations) {
    // We compute the dot product by counting the -1 terms
    extern __shared__ unsigned int negatives[];

    // Begin fetching the bias so hopefully the latency is hidden.
    // Also, pre-offset the bias by the total number of bits.
    const int bias = biases[blockIdx.x] + n_chunks * CHUNK_BIT;

    // First, compute the number of -1 summands in the dot product of this
    // thread's chunks of the input and the block's weight column
    negatives[threadIdx.x] = 0;
    for (unsigned int idx = threadIdx.x; idx < n_chunks; idx += blockDim.x) {
        negatives[threadIdx.x] +=
            __popc(inputs[idx] ^ weights[blockIdx.x * n_chunks + idx]);
    }
    __syncthreads();

    // TODO: remove
    printf("(%d, %d): %u\n", blockIdx.x, threadIdx.x, negatives[threadIdx.x]);

    // Then, compute the sum across the block into the first index with a
    // parallel "tree" reduce with sequential addressing
    for (unsigned int active_threads = blockDim.x / 2; active_threads > 0;
         active_threads >>= 1) {
        // Use the contiguous low chunk of threads
        if (threadIdx.x < active_threads) {
            negatives[threadIdx.x] += negatives[threadIdx.x + active_threads];
        }

        // Before the next step, all memory values must be valid
        __syncthreads();
    }

    // Finally, each block writes its dot product to the output
    if (threadIdx.x == 0) {
        activations[blockIdx.x] = bias - (negatives[0] << 1);
    }
}

LinearLayer::LinearLayer(const unsigned int input_chunks,
                         const unsigned int output_dims)
    : input_chunks(input_chunks),
      input_dims(input_chunks * CHUNK_BIT),
      output_dims(output_dims) {
    if (!is_power_of_two(input_chunks)) {
        throw std::invalid_argument("input_chunks must be a power of 2");
    }
    if (input_chunks > 2048) {
        throw std::invalid_argument("input_chunks can be at most 2048");
    }

    gpuErrChk(cudaMalloc(&weights, input_chunks * output_dims * CHUNK_BYTE));
    gpuErrChk(cudaMalloc(&biases, output_dims * sizeof(int)));
    gpuErrChk(cudaMalloc(&activations, output_dims * sizeof(int)));
}

LinearLayer::~LinearLayer() {
    gpuErrChk(cudaFree(weights));
    gpuErrChk(cudaFree(biases));
    gpuErrChk(cudaFree(activations));
}

/*
 * Compute the forward pass on device data. Assumes that device_input contains
 * input_chunks * CHUNK_BYTE bytes of binary data.
 */
int* LinearLayer::forward(void* device_input) {
    const unsigned int n_threads = std::min(input_chunks, MAX_BLOCK_THREADS);
    linear_forward<<<output_dims, n_threads, n_threads * sizeof(int)>>>(
        input_chunks, (const unsigned int*)device_input,
        (const unsigned int*)weights, biases, activations);
    return activations;
}

void LinearLayer::set_params(const unsigned char* host_weights,
                             const int* host_biases) {
    gpuErrChk(cudaMemcpy(weights, (const void*)host_weights,
                         input_chunks * CHUNK_BYTE * output_dims,
                         cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(biases, (const void*)host_biases,
                         output_dims * sizeof(int), cudaMemcpyHostToDevice));
}
