#ifndef TEST_GPU_HIP_H 
#define TEST_GPU_HIP_H

#include "common_gpu.h"

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPU KERNELS ---------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------


// GPU Kernel for reduction using warp (uses appropriate warp for NVIDIA vs AMD devices i. e. "portable wave aware code")
__device__ void warp_reduce(volatile double *sdata, size_t thread_idx) {
    if (warpSize == 64) { if (GPU_BLOCK_SIZE >= 128) sdata[thread_idx] += sdata[thread_idx + 64]; }
    if (GPU_BLOCK_SIZE >= 64) sdata[thread_idx] += sdata[thread_idx + 32];
    if (GPU_BLOCK_SIZE >= 32) sdata[thread_idx] += sdata[thread_idx + 16];
    if (GPU_BLOCK_SIZE >= 16) sdata[thread_idx] += sdata[thread_idx + 8];
    if (GPU_BLOCK_SIZE >= 8) sdata[thread_idx] += sdata[thread_idx + 4];
    if (GPU_BLOCK_SIZE >= 4) sdata[thread_idx] += sdata[thread_idx + 2];
    if (GPU_BLOCK_SIZE >= 2) sdata[thread_idx] += sdata[thread_idx + 1];
}

__global__
void gpu_dot(double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
    __shared__ double _c[GPU_BLOCK_SIZE];
    size_t _j = hipThreadIdx_x;
    if (_j < N) {
        _c[hipThreadIdx_x] = A[_j]*B[_j];
    } else {
        _c[hipThreadIdx_x] = 0.0;
    }

    for (size_t i = 1; i < (N + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE; i++) {
        size_t j = GPU_BLOCK_SIZE*i + local_idx;
        if (j < N) {
            _c[_j] += A[j]*B[j];
        }
    }
    __syncthreads();

    // NEED TO REDUCE _c ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (hipThreadIdx_x < 512) {
            _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (hipThreadIdx_x < 256) {
            _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (hipThreadIdx_x < 128) {
            _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (hipThreadIdx_x < 64) {
                _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 64];
            }
            __syncthreads();
        } 
    }

    if (hipThreadIdx_x < warpSize) {
        warp_reduce(_c, hipThreadIdx_x);
    }

    if (hipThreadIdx_x == 0) {
        C[0] = _c[0];
    }
}

#endif

