#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

#include <iostream>

#define CHUNK_BYTE sizeof(int)
#define CHUNK_BIT (CHUNK_BYTE * CHAR_BIT)
#define MAX_BLOCK_THREADS ((unsigned int)1024)

/*
 * Macro for error-checking CUDA calls. Taken from Lab 2.
 *
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 * what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrChk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        exit(code);
    }
}

// Snippet taken from:
// https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
inline bool is_power_of_two(unsigned int x) { return x && !(x & (x - 1)); }
