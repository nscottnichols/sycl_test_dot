#ifdef USE_SYCL
    #include "test_gpu.sycl.h"
#endif
#ifdef USE_HIP
    #include "test_gpu.hip.h"
#endif
#ifdef USE_CUDA
    #include "test_gpu.cuda.cuh"
#endif
#include <iostream>
#include <cassert>

int usage(char* argv0, int ret = 1) {
    std::cout << "Usage: " << argv0
              << " [-h] [-N] [-L]" << std::endl << std::endl;
    std::cout << "Optional arguments:"                                                                          << std::endl;
    std::cout << "  -h, --help                          shows help message and exits"                           << std::endl;
    std::cout << "  -N, --number_of_elements            number of elements in arrays (default: 8)"              << std::endl;
    std::cout << "  -L, --number_of_kernel_launches     number of dot product kernel launches (default: 1)"     << std::endl;
    std::cout << "  -R, --number_of_repititions         number of times to repeat kernel launches (default: 1)" << std::endl;
    return ret;
}

int main(int argc, char **argv) {
    //Parse arguments
    size_t N = 8;
    size_t L = 1;
    size_t R = 1;
    for (int argn = 1; argn < argc; ++argn) {
        std::string arg = argv[argn];
        if ((arg == "--help") || (arg == "-h")) {
            return usage(argv[0]);
        } else if ((arg == "--number_of_elements") || (arg == "-N")) {
            N = static_cast<size_t>(strtoul(argv[argn + 1], NULL, 0));
            argn++;
        } else if ((arg == "--number_of_kernel_launches") || (arg == "-L")) {
            L = static_cast<size_t>(strtoul(argv[argn + 1], NULL, 0));
            argn++;
        } else if ((arg == "--number_of_repititions") || (arg == "-R")) {
            R = static_cast<size_t>(strtoul(argv[argn + 1], NULL, 0));
            argn++;
        }
    }

    std::cout << "N = " << N << std::endl;
    std::cout << "L = " << L << std::endl;
    std::cout << "R = " << R << std::endl;
    #ifdef USE_SYCL
        //Setup device
        auto devices = sycl::device::get_devices();
        sycl::queue q = sycl::queue(devices[0]);
        
        //Test for valid subgroup size
        auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
        if (std::none_of(sg_sizes.cbegin(), sg_sizes.cend(), [](auto i) { return i  == SUB_GROUP_SIZE; })) {
            std::stringstream ss;
            ss << "Invalid SUB_GROUP_SIZE. Please select from: ";
            for (auto it = sg_sizes.cbegin(); it != sg_sizes.cend(); it++) {
                if (it != sg_sizes.begin()) {
                    ss << " ";
                }
                ss << *it;
            }
            throw std::runtime_error(ss.str());
        }
    #endif

    //test void gpu_dot
    auto h_a  = (double*) malloc(sizeof(double)*N); 
    auto h_b  = (double*) malloc(sizeof(double)*N); 
    auto h_c  = (double*) malloc(sizeof(double)*L); 
    auto h_c2 = (double*) malloc(sizeof(double)*L); 

    //initialize host memory to zero
    for (size_t i=0; i<L; i++) {
        h_c[i] = 0.0;
    }

    //set host arrays
    for (size_t i=0; i<N; i++) {
        h_a[i] = static_cast<double>(i);
        h_b[i] = static_cast<double>(N - i - 1);
        for (size_t j=0; j<L; j++) {
            h_c[j] += h_a[i]*h_b[i];
        }
    }

    #ifdef USE_SYCL
        //Set device memory so reads are coming from separate memory locations
        auto d_a = sycl::malloc_device<double>(N*L, q); 
        auto d_b = sycl::malloc_device<double>(N*L, q); 
        auto d_c = sycl::malloc_device<double>(L  , q); 

        for (size_t i=0; i<L; i++) {
            q.memcpy(d_a + i*N, h_a, sizeof(double)*N);
            q.memcpy(d_b + i*N, h_b, sizeof(double)*N);
        }
        q.wait();

        for (size_t j=0; j<R; j++) {
            for (size_t i=0; i<L; i++) {
                #ifdef SYCL_USE_WGDP
                    gpu_dot_wgdp(q, d_c + i, d_a + i*N, d_b + i*N, N);
                #else
                    gpu_dot(q, d_c + i, d_a + i*N, d_b + i*N, N);
                #endif
            }
            q.wait();
        }

        q.memcpy(h_c2, d_c, sizeof(double)*L).wait();

        sycl::free(d_a, q);
        sycl::free(d_b, q);
        sycl::free(d_c, q);
    #endif

    #ifdef USE_HIP
        #if MAX_GPU_STREAMS > 1
            hipStream_t stream_array[MAX_GPU_STREAMS];
            for (size_t i = 0; i < MAX_GPU_STREAMS; i++) {
                HIP_ASSERT(hipStreamCreate(&stream_array[i]));
            }
        #endif
        double* d_a;
        double* d_b;
        double* d_c;
        HIP_ASSERT(hipMalloc(&d_a, sizeof(double)*N*L));
        HIP_ASSERT(hipMalloc(&d_b, sizeof(double)*N*L));
        HIP_ASSERT(hipMalloc(&d_c, sizeof(double)*L  ));

        for (size_t i=0; i<L; i++) {
            HIP_ASSERT(hipMemcpy(d_a + i*N, h_a, sizeof(double)*N, hipMemcpyHostToDevice));
            HIP_ASSERT(hipMemcpy(d_b + i*N, h_b, sizeof(double)*N, hipMemcpyHostToDevice));
        }
        HIP_ASSERT(hipDeviceSynchronize());

        for (size_t j=0; j<R; j++) {
            for (size_t i=0; i<L; i++) {
                #if MAX_GPU_STREAMS > 1
                    size_t stream_idx = i % MAX_GPU_STREAMS;
                    hipLaunchKernelGGL(gpu_dot, dim3(1), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                            d_c + i, d_a + i*N, d_b + i*N, N); 
                #else
                    hipLaunchKernelGGL(gpu_dot, dim3(1), dim3(GPU_BLOCK_SIZE), 0, 0,
                            d_c + i, d_a + i*N, d_b + i*N, N); 
                #endif
            }
            HIP_ASSERT(hipDeviceSynchronize());
        }

        HIP_ASSERT(hipMemcpy(h_c2, d_c, sizeof(double)*L, hipMemcpyDeviceToHost));
        HIP_ASSERT(hipDeviceSynchronize());

        HIP_ASSERT(hipFree(d_a));
        HIP_ASSERT(hipFree(d_b));
        HIP_ASSERT(hipFree(d_c));

        #if MAX_GPU_STREAMS > 1
            for (size_t i = 0; i < MAX_GPU_STREAMS; i++) {
                HIP_ASSERT(hipStreamDestroy(stream_array[i]));
            }
        #endif
    #endif

    #ifdef USE_CUDA
        #if MAX_GPU_STREAMS > 1
            cudaStream_t stream_array[MAX_GPU_STREAMS];
            for (size_t i = 0; i < MAX_GPU_STREAMS; i++) {
                CUDA_ASSERT(cudaStreamCreate(&stream_array[i]));
            }
        #endif
        double* d_a;
        double* d_b;
        double* d_c;
        CUDA_ASSERT(cudaMalloc(&d_a, sizeof(double)*N*L));
        CUDA_ASSERT(cudaMalloc(&d_b, sizeof(double)*N*L));
        CUDA_ASSERT(cudaMalloc(&d_c, sizeof(double)*L  ));

        for (size_t i=0; i<L; i++) {
            CUDA_ASSERT(cudaMemcpy(d_a + i*N, h_a, sizeof(double)*N, cudaMemcpyHostToDevice));
            CUDA_ASSERT(cudaMemcpy(d_b + i*N, h_b, sizeof(double)*N, cudaMemcpyHostToDevice));
        }
        CUDA_ASSERT(cudaDeviceSynchronize());

        for (size_t j=0; j<R; j++) {
            for (size_t i=0; i<L; i++) {
                #if MAX_GPU_STREAMS > 1
                    size_t stream_idx = i % MAX_GPU_STREAMS;
                    cuda_wrapper::gpu_dot_wrapper(dim3(1), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                            d_c + i, d_a + i*N, d_b + i*N, N);
                #else
                    cuda_wrapper::gpu_dot_wrapper(dim3(1), dim3(GPU_BLOCK_SIZE),
                            d_c + i, d_a + i*N, d_b + i*N, N);
                #endif
            }
            CUDA_ASSERT(cudaDeviceSynchronize());
        }

        CUDA_ASSERT(cudaMemcpy(h_c2, d_c, sizeof(double)*L, cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaDeviceSynchronize());

        CUDA_ASSERT(cudaFree(d_a));
        CUDA_ASSERT(cudaFree(d_b));
        CUDA_ASSERT(cudaFree(d_c));

        #if MAX_GPU_STREAMS > 1
            for (size_t i = 0; i < MAX_GPU_STREAMS; i++) {
                CUDA_ASSERT(cudaStreamDestroy(stream_array[i]));
            }
        #endif
    #endif

    for (size_t i=0; i < L; i++) {
        std::cout << "h_c[" << i << "]:  " << h_c[i]  << std::endl; 
        std::cout << "h_c2[" << i << "]: " << h_c2[i] << std::endl; 
    }

    free(h_a );
    free(h_b );
    free(h_c );
    free(h_c2);
}
