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

int main(int argc, char **argv) {
    //Parse arguments
    size_t N = 8;
    for (int argn = 1; argn < argc; ++argn) {
        std::string arg = argv[argn];
        if ((arg == "--number_of_elements") || (arg == "-N")) {
            N = static_cast<size_t>(strtoul(argv[argn + 1]));
            argn++;
        } else if ((arg == "--help") || (arg == "-h")) {
            return usage(argv[0]);
        }
    }

    std::cout << "N = " << N << std::endl;
    #ifdef USE_SYCL
        //Setup device
        auto devices = sycl::device::get_devices();
        sycl::queue q = sycl::queue(devices[0]);
        
        //Test for vaild subgroup size
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

    //test void gpu_matmul
    size_t grid_size = (N + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
    assert(grid_size == 1);
    auto h_a  = (double*) malloc(sizeof(double)*N); 
    auto h_b  = (double*) malloc(sizeof(double)*N); 
    auto h_c  = (double*) malloc(sizeof(double)  ); 
    auto h_c2 = (double*) malloc(sizeof(double)  ); 
    h_c[0] = 0.0;
    for (size_t i=0; i<M; i++) {
        h_a[i] = static_cast<double>(i);
        h_b[i] = static_cast<double>(i);
        h_c[0] += h_a[i]*h_b[i];
    }

    #ifdef USE_SYCL
        auto d_a = sycl::malloc_device<double>(M, q); 
        auto d_b = sycl::malloc_device<double>(M, q); 
        auto d_c = sycl::malloc_device<double>(1, q); 

        q.memcpy(d_a, h_a, sizeof(double)*M);
        q.memcpy(d_b, h_b, sizeof(double)*M).wait();
        gpu_matmul(q, grid_size, d_c, d_a, d_b, N);
        q.wait();
        q.memcpy(h_c2, d_c, sizeof(double)).wait();

        sycl::free(d_a, q);
        sycl::free(d_b, q);
        sycl::free(d_c, q);
    #endif

    #ifdef USE_HIP
        double* d_a;
        double* d_b;
        double* d_c;
        HIP_ASSERT(hipMalloc(&d_a, sizeof(double)*M));
        HIP_ASSERT(hipMalloc(&d_b, sizeof(double)*M));
        HIP_ASSERT(hipMalloc(&d_c, sizeof(double)  ));
        auto d_a = sycl::malloc_device<double>(M, q); 
        auto d_b = sycl::malloc_device<double>(M, q); 
        auto d_c = sycl::malloc_device<double>(1, q); 

        HIP_ASSERT(hipMemcpy(d_a, h_a, sizeof(double)*M, hipMemcpyHostToDevice));
        HIP_ASSERT(hipMemcpy(d_b, h_b, sizeof(double)*M, hipMemcpyHostToDevice));
        gpu_matmul(q, grid_size, d_c, d_a, d_b, N);
        hipLaunchKernelGGL(gpu_matmul, dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, 0,
                d_c, d_a, d_b, N); 
        HIP_ASSERT(hipDeviceSynchronize());
        HIP_ASSERT(hipMemcpy(h_c2, d_c, sizeof(double), hipMemcpyDeviceToHost));
        HIP_ASSERT(hipDeviceSynchronize());

        HIP_ASSERT(hipFree(d_a));
        HIP_ASSERT(hipFree(d_b));
        HIP_ASSERT(hipFree(d_c));
    #endif

    #ifdef USE_CUDA
        double* d_a;
        double* d_b;
        double* d_c;
        CUDA_ASSERT(cudaMalloc(&d_a, sizeof(double)*M));
        CUDA_ASSERT(cudaMalloc(&d_b, sizeof(double)*M));
        CUDA_ASSERT(cudaMalloc(&d_c, sizeof(double)  ));
        auto d_a = sycl::malloc_device<double>(M, q); 
        auto d_b = sycl::malloc_device<double>(M, q); 
        auto d_c = sycl::malloc_device<double>(1, q); 

        CUDA_ASSERT(cudaMemcpy(d_a, h_a, sizeof(double)*M, cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_b, h_b, sizeof(double)*M, cudaMemcpyHostToDevice));
        cuda_wrapper::gpu_matmul_wrapper(dim3(grid_size), dim3(GPU_BLOCK_SIZE),
                d_c, d_a, d_b, N);
        CUDA_ASSERT(cudaDeviceSynchronize());
        CUDA_ASSERT(cudaMemcpy(h_c2, d_c, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaDeviceSynchronize());

        CUDA_ASSERT(cudaFree(d_a));
        CUDA_ASSERT(cudaFree(d_b));
        CUDA_ASSERT(cudaFree(d_c));
    #endif

    std::cout << "h_c[0]:  " << h_c[0]  << std::endl; 
    std::cout << "h_c2[0]: " << h_c2[0] << std::endl; 

    free(h_a );
    free(h_b );
    free(h_c );
    free(h_c2);
}
