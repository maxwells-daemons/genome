#include <cstdio>
#include <stdexcept>

#include "common.hpp"
#include "sign_layer.cuh"

/*
 * Kernel to convert biases to thresholds on the number of allowable negative
 * bits, equal to (biases + num_outputs) / 2. This saves some arithmetic at
 * inference time. Biases are converted in place.
 */
__global__ void biases_to_thresholds(int* biases,
                                     const unsigned int num_outputs) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_outputs;
         idx += blockDim.x * gridDim.x) {
        biases[idx] += num_outputs;
        biases[idx] >>= 1;
    }
}

/*
 * Kernel to compute activations <- sign(weights^T inputs + b).
 *
 * Each block i creates a single bit of output by computing
 * sign(W[:, i] . X + b[i]). Each thread in a block is responsible for
 * one or more "chunks" of 4 bytes of input and weight data.
 *
 * Preconditions:
 *  - Dynamically allocated one unsigned int of shared space per thread.
 *  - inputs contains (CHUNK_BYTE * n_chunks) bytes of input.
 *  - weights contains (CHUNK_BYTE * n_chunks * blockDim.x) bytes of weights.
 *  - thresholds contains blockDim.x integers.
 *  - activations has space for (blockDim.x / CHUNK_BYTE) bytes of output.
 */
__global__ void sign_forward(const unsigned int n_chunks,
                             const unsigned int* inputs,
                             const unsigned int* weights, const int* thresholds,
                             unsigned int* activations) {
    // We implicitly compute the dot product by counting the -1 terms
    extern __shared__ unsigned int negatives[];

    // Begin fetching the threshold so hopefully the latency is hidden
    const int threshold = thresholds[blockIdx.x];

    // We'll write the `blockIdx.x`-th bit of the output atomically,
    // through a chunk index and a bit offset.
    const unsigned int output_chunk = blockIdx.x / CHUNK_BIT;
    const unsigned int output_offset = blockIdx.x % CHUNK_BIT;
    activations[output_chunk] = 0;

    // First, compute the number of -1 summands in the dot product of this
    // thread's chunks of the input and the block's weight column
    negatives[threadIdx.x] = 0;
    for (unsigned int idx = threadIdx.x; idx < n_chunks; idx += blockDim.x) {
        negatives[threadIdx.x] +=
            __popc(inputs[idx] ^ weights[blockIdx.x * n_chunks + idx]);
    }
    __syncthreads();

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

    // Finally, each block writes a single bit to the output
    if (threadIdx.x == 0) {
        const unsigned int write_bit = (negatives[0] < threshold)
                                       << output_offset;
        atomicOr(activations + output_chunk, write_bit);
    }
}

SignLayer::SignLayer(const unsigned int input_chunks,
                     const unsigned int output_chunks)
    : input_chunks(input_chunks),
      output_chunks(output_chunks),
      input_dims(input_chunks * CHUNK_BIT),
      output_dims(output_chunks * CHUNK_BIT) {
    if (!is_power_of_two(input_chunks)) {
        throw std::invalid_argument("input_chunks must be a power of 2");
    }
    if (input_chunks > 2048) {
        throw std::invalid_argument("input_chunks can be at most 2048");
    }

    gpuErrChk(cudaMalloc(&weights, input_chunks * output_dims * CHUNK_BYTE));
    gpuErrChk(cudaMalloc(&thresholds, output_dims * sizeof(int)));
    gpuErrChk(cudaMalloc(&activations, output_chunks * CHUNK_BYTE));
}

SignLayer::~SignLayer() {
    gpuErrChk(cudaFree(weights));
    gpuErrChk(cudaFree(thresholds));
    gpuErrChk(cudaFree(activations));
}

/*
 * Compute the forward pass on device data. Assumes that device_input contains
 * input_chunks * CHUNK_BYTE bytes of binary data.
 */
void* SignLayer::forward(void* device_input) {
    const unsigned int n_threads = std::min(input_chunks, MAX_BLOCK_THREADS);
    sign_forward<<<output_dims, n_threads, n_threads * sizeof(int)>>>(
        input_chunks, (const unsigned int*)device_input,
        (const unsigned int*)weights, thresholds, (unsigned int*)activations);
    return activations;
}

void SignLayer::set_params(const unsigned char* host_weights,
                           const int* host_biases) {
    gpuErrChk(cudaMemcpy(weights, (const void*)host_weights,
                         input_chunks * CHUNK_BYTE * output_dims,
                         cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(thresholds, (const void*)host_biases,
                         output_dims * sizeof(int), cudaMemcpyHostToDevice));

    // Hardcoded, since this is a minor operation
    biases_to_thresholds<<<32, 256>>>(thresholds, output_dims);
}
