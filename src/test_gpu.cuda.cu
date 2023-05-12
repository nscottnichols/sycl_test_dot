#ifndef TEST_GPU_CUDA_H 
#define TEST_GPU_CUDA_H

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
void gpu_matmul_short(double* C, double* B, double* A, size_t N) {
    __shared__ double _c[GPU_BLOCK_SIZE];
    size_t _j = blockDim.x * blockIdx.x + threadIdx.x;
    if (_j < N) {
        _c[threadIdx.x] = A[_j]*B[_j];
    } else {
        _c[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _c ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (threadIdx.x < 512) {
            _c[threadIdx.x] += _c[threadIdx.x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (threadIdx.x < 256) {
            _c[threadIdx.x] += _c[threadIdx.x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) {
            _c[threadIdx.x] += _c[threadIdx.x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (threadIdx.x < 64) {
                _c[threadIdx.x] += _c[threadIdx.x + 64];
            }
            __syncthreads();
        } 
    }

    if (threadIdx.x < warpSize) {
        warp_reduce(_c, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        C[0] = _c[0];
    }
}

__global__
void gpu_matmul(double* C, double* B, double* A, size_t N) {
    // C = B*A where [B] = 1xN and [A] = Nx1
    __shared__ double _c[GPU_BLOCK_SIZE];
    // Set shared local memory _c
    auto global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto local_idx = threadIdx.x;
    if (global_idx < N) {
        _c[local_idx] = A[global_idx]*B[global_idx];
    } else {
        _c[local_idx] = 0.0;
    }
    __syncthreads();

    // Reduce _c (using shared local memory)
    #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
        if (local_idx < 512) {
            _c[local_idx] += _c[local_idx + 512];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
        if (local_idx < 256) {
            _c[local_idx] += _c[local_idx + 256];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
        if (local_idx < 128) {
            _c[local_idx] += _c[local_idx + 128];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
        if (local_idx < 64) {
            _c[local_idx] += _c[local_idx + 64];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
        if (local_idx < 32) {
            _c[local_idx] += _c[local_idx + 32];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
        if (local_idx < 16) {
            _c[local_idx] += _c[local_idx + 16];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
        if (local_idx < 8) {
            _c[local_idx] += _c[local_idx + 8];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
        if (local_idx < 4) {
            _c[local_idx] += _c[local_idx + 4];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
        if (local_idx < 2) {
            _c[local_idx] += _c[local_idx + 2];
        }
        __syncthreads();
    #endif

    //Sub-group simultaneous work-item tasks (warp/wavefront/sub-group)
    //for (size_t j = SUB_GROUP_SIZE; j > 0; j /= 2) {
    //    _c[local_idx] += _c[local_idx + j];
    //}
    #if (SUB_GROUP_SIZE >= 512)
        _c[local_idx] += _c[local_idx + 512];
    #endif
    #if (SUB_GROUP_SIZE >= 256)
        _c[local_idx] += _c[local_idx + 256];
    #endif
    #if (SUB_GROUP_SIZE >= 128)
        _c[local_idx] += _c[local_idx + 128];
    #endif
    #if (SUB_GROUP_SIZE >= 64)
        _c[local_idx] += _c[local_idx + 64];
    #endif
    #if (SUB_GROUP_SIZE >= 32)
        _c[local_idx] += _c[local_idx + 32];
    #endif
    #if (SUB_GROUP_SIZE >= 16)
        _c[local_idx] += _c[local_idx + 16];
    #endif
    #if (SUB_GROUP_SIZE >= 8)
        _c[local_idx] += _c[local_idx + 8];
    #endif
    #if (SUB_GROUP_SIZE >= 4)
        _c[local_idx] += _c[local_idx + 4];
    #endif
    #if (SUB_GROUP_SIZE >= 2)
        _c[local_idx] += _c[local_idx + 2];
    #endif
    #if (SUB_GROUP_SIZE >= 1)
        _c[local_idx] += _c[local_idx + 1];
    #endif

    //Set C
    if (local_idx == 0) {
         C[0] = _c[0];
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPU KERNEL WRAPPER --------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
namespace cuda_wrapper {
    void gpu_matmul_short_wrapper(dim3 grid_size, dim3 group_size, double* C, double* B, double * A, size_t N) {
        gpu_matmul_short <<<grid_size, group_size, 0, 0>>> ( 
                C, B, A, N
                );
    }
    void gpu_matmul_short_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double* C, double* B, double* A, size_t N) {
        gpu_matmul_short <<<grid_size, group_size, 0, stream>>> ( 
                C, B, A, N
                );
    }

    void gpu_matmul_wrapper(dim3 grid_size, dim3 group_size, double* C, double* B, double * A, size_t N) {
        gpu_matmul <<<grid_size, group_size, 0, 0>>> ( 
                C, B, A, N
                );
    }
    void gpu_matmul_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double* C, double* B, double* A, size_t N) {
        gpu_matmul <<<grid_size, group_size, 0, stream>>> ( 
                C, B, A, N
                );
    }
}
#endif

