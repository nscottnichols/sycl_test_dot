#ifndef TEST_GPU_CUDA_H 
#define TEST_GPU_CUDA_H

#include "common_gpu.h"

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPU KERNELS ---------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------


__device__ void warp_reduce(volatile double *_slm, size_t local_idx) {
    #if (SUB_GROUP_SIZE >= 64)
        _slm[local_idx] += _slm[local_idx + 64];
    #endif
    #if (SUB_GROUP_SIZE >= 32)
        _slm[local_idx] += _slm[local_idx + 32];
    #endif
    #if (SUB_GROUP_SIZE >= 16)
        _slm[local_idx] += _slm[local_idx + 16];
    #endif
    #if (SUB_GROUP_SIZE >= 8)
        _slm[local_idx] += _slm[local_idx + 8];
    #endif
    #if (SUB_GROUP_SIZE >= 4)
        _slm[local_idx] += _slm[local_idx + 4];
    #endif
    #if (SUB_GROUP_SIZE >= 2)
        _slm[local_idx] += _slm[local_idx + 2];
    #endif
    #if (SUB_GROUP_SIZE >= 1)
        _slm[local_idx] += _slm[local_idx + 1];
    #endif
}

__global__
void gpu_dot(double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
    __shared__ double _c[GPU_BLOCK_SIZE];
    size_t _j = threadIdx.x;
    if (_j < N) {
        _c[threadIdx.x] = A[_j]*B[_j];
    } else {
        _c[threadIdx.x] = 0.0;
    }

    for (size_t i = 1; i < (N + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE; i++) {
        size_t j = GPU_BLOCK_SIZE*i + _j;
        if (j < N) {
            _c[_j] += A[j]*B[j];
        }
    }
    __syncthreads();

    // NEED TO REDUCE _c ON SHARED MEMORY AND ADD TO GLOBAL isf
    #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
        if (threadIdx.x < 512) {
            _c[threadIdx.x] += _c[threadIdx.x + 512];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
        if (threadIdx.x < 256) {
            _c[threadIdx.x] += _c[threadIdx.x + 256];
        }
        __syncthreads();
    } 
    #endif

    #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
        if (threadIdx.x < 128) {
            _c[threadIdx.x] += _c[threadIdx.x + 128];
        }
        __syncthreads();
    } 

    #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
        if (GPU_BLOCK_SIZE >= 128) {
            if (threadIdx.x < 64) {
                _c[threadIdx.x] += _c[threadIdx.x + 64];
            }
            __syncthreads();
        } 
    #endif

    if (threadIdx.x < warpSize) {
        warp_reduce(_c, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        C[0] = _c[0];
    }
}


// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPU KERNEL WRAPPER --------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
namespace cuda_wrapper {
    void gpu_dot_wrapper(dim3 grid_size, dim3 group_size, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
        gpu_dot <<<grid_size, group_size, 0, 0>>> ( 
                C, B, A, N
                );
    }
    void gpu_dot_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
        gpu_dot <<<grid_size, group_size, 0, stream>>> ( 
                C, B, A, N
                );
    }
}
#endif

