cmake_minimum_required(VERSION 3.18.0)
project(test_matmul.e LANGUAGES CXX)

# Use GNUInstallDirs to install libraries into correct locations on all platforms.
include(GNUInstallDirs)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_STANDARD_REQUIRED TRUE)

# Check for workaround flag
if(DEFINED SYCL_MATMUL_WA)
    add_definitions(-D SYCL_MATMUL_WA=1)
endif()

# Set GPU native support backend
add_definitions(-D USE_GPU=1)
set(GPU_BACKEND "sycl" CACHE STRING "Set GPU native backend: sycl, hip, cuda (default: sycl)")
if (NOT ${GPU_BACKEND} STREQUAL "sycl")
    if (${GPU_BACKEND} STREQUAL "cuda")
        enable_language(CUDA)
        add_definitions(-D USE_CUDA=1)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 70)
        endif()
        message("CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}.")
        include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    elseif(${GPU_BACKEND} STREQUAL "hip")
        add_definitions(-D USE_HIP=1)
    else()
        message(FATAL_ERROR "Unknown GPU implementation '${GPU_BACKEND}',"
            " please select from 'cuda, hip, sycl, none'.")
    endif()
else()
    add_definitions(-D USE_SYCL=1)
endif()

# Set number of threads per block
set(GPU_BLOCK_SIZE "1024" CACHE STRING "Set gpu block size (default: 1024)")
add_definitions(-D GPU_BLOCK_SIZE=${GPU_BLOCK_SIZE})

# Set sub-group size (warpsize/wavefront/SIMD lanes/etc.)
set(SUB_GROUP_SIZE "32" CACHE STRING "Set sub-group size (default: 32)")
add_definitions(-D SUB_GROUP_SIZE=${SUB_GROUP_SIZE})

# Set number of GPU streams
set(MAX_GPU_STREAMS "1" CACHE STRING "Set number of gpu streams (default: 1)")
add_definitions(-D MAX_GPU_STREAMS=${MAX_GPU_STREAMS})

# Set default build flags
if(${GPU_BACKEND} STREQUAL "sycl")
    if(NOT DEFINED SYCL_FLAGS)
        message(FATAL_ERROR "Please set SYCL_FLAGS CMake variable for target architecture and device,"
            " e.g. to target an NVIDIA A100 GPU:"
            "     '-DSYCL_FLAGS=\"-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80\"'")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS} -march=native -Wall -fno-math-errno -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -fno-math-errno -O3")
endif()

# Determine executable name
set(exe test_matmul.e)

# Find source files ( better to list explicitly https://stackoverflow.com/questions/1027247/specify-source-files-globally-with-glob )
file( GLOB TEST_MATMUL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp )
if (${GPU_BACKEND} STREQUAL "cuda")
    file( GLOB TEST_MATMUL_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu )
    add_executable(${exe} ${TEST_MATMUL_SRC} ${TEST_MATMUL_CUDA})
    set_target_properties( ${exe} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    #if(APPLE)
    #  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
    #  # so that the static cuda runtime can find it at runtime.
    #  set_property(TARGET ${exe} 
    #               PROPERTY
    #               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    #endif()
else()
    add_executable(${exe} ${TEST_MATMUL_SRC})
endif()

# Define headers for target
target_include_directories(${exe} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)

# Set debug flags
if(${GPU_BACKEND} STREQUAL "sycl")
    if(NOT DEFINED SYCL_FLAGS)
        message(FATAL_ERROR "Please set SYCL_FLAGS CMake variable for target architecture and device,"
            " e.g. to target an NVIDIA A100 GPU:"
            "     '-DSYCL_FLAGS=\"-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80\"'")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${SYCL_FLAGS} -DTEST_MATMUL_DEBUG=1")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE None
       CACHE STRING "Choose the type of build : None Debug Release."
       FORCE)
endif()

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS ${exe} EXPORT ${exe}Config
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${exe}Config DESTINATION share/${exe}/cmake)

# This makes the project importable from the build directory
export(TARGETS ${exe} FILE ${exe}Config.cmake)
