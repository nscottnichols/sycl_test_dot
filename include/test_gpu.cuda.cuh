#pragma once
#include "common_gpu.h"
#include "device_launch_parameters.h"

namespace cuda_wrapper {
    void gpu_dot_short_wrapper(dim3, dim3, double*, double*, double*, size_t);
    void gpu_dot_short_wrapper(dim3, dim3, cudaStream_t, double*, double*, double*, size_t);

    void gpu_dot_wrapper(dim3, dim3, double*, double*, double*, size_t);
    void gpu_dot_wrapper(dim3, dim3, cudaStream_t, double*, double*, double*, size_t);
}

