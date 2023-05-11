#ifndef TEST_GPU_SYCL_H 
#define TEST_GPU_SYCL_H
#include "common_gpu.h"

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPU KERNELS ---------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void gpu_matmul(sycl::queue q, size_t grid_size, double* C, double* B, double* A, size_t N) {
    // C = B*A where [B] = 1xN and [A] = Nx1
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            double _c[GPU_BLOCK_SIZE];
            // Set shared local memory _c
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < N) {
                    _c[local_idx] = A[global_idx]*B[global_idx];
                } else {
                    _c[local_idx] = 0.0;
                }
            });

            // Reduce _c (using shared local memory)
            #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < 512) {
                        _c[local_idx] += _c[local_idx + 512];
                    }
                });
            #endif

            #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < 256) {
                        _c[local_idx] += _c[local_idx + 256];
                    }
                });
            #endif

            #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < 128) {
                        _c[local_idx] += _c[local_idx + 128];
                    }
                });
            #endif

            #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < 64) {
                        _c[local_idx] += _c[local_idx + 64];
                    }
                });
            #endif

            #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < 32) {
                        _c[local_idx] += _c[local_idx + 32];
                    }
                });
            #endif

            #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < 16) {
                        _c[local_idx] += _c[local_idx + 16];
                    }
                });
            #endif

            #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < 8) {
                        _c[local_idx] += _c[local_idx + 8];
                    }
                });
            #endif

            #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < 4) {
                        _c[local_idx] += _c[local_idx + 4];
                    }
                });
            #endif

            #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < 2) {
                        _c[local_idx] += _c[local_idx + 2];
                    }
                });
            #endif

            //Workaround fix
            #ifdef SYCL_MATMUL_WA
                #if (SUB_GROUP_SIZE >= 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _c[local_idx] += _c[local_idx + 512];
                        }
                    });
		#endif

                #if (SUB_GROUP_SIZE >= 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _c[local_idx] += _c[local_idx + 256];
                        }
                    });
		#endif

                #if (SUB_GROUP_SIZE >= 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _c[local_idx] += _c[local_idx + 128];
                        }
                    });
		#endif

                #if (SUB_GROUP_SIZE >= 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _c[local_idx] += _c[local_idx + 64];
                        }
                    });
		#endif

                #if (SUB_GROUP_SIZE >= 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _c[local_idx] += _c[local_idx + 32];
                        }
                    });
		#endif

                #if (SUB_GROUP_SIZE >= 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _c[local_idx] += _c[local_idx + 16];
                        }
                    });
		#endif

                #if (SUB_GROUP_SIZE >= 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _c[local_idx] += _c[local_idx + 8];
                        }
                    });
		#endif

                #if (SUB_GROUP_SIZE >= 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _c[local_idx] += _c[local_idx + 4];
                        }
                    });
		#endif

                #if (SUB_GROUP_SIZE >= 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _c[local_idx] += _c[local_idx + 2];
                        }
                    });
		#endif

                #if (SUB_GROUP_SIZE >= 1)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 1) {
                            _c[local_idx] += _c[local_idx + 1];
                        }
                    });
		#endif

            #else

                //Sub-group simultaneous work-item tasks (warp/wavefront/sub-group)
                //wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                //    size_t local_idx = index.get_local_id(0);
                //    for (size_t j = SUB_GROUP_SIZE; j > 0; j /= 2) {
                //        //FIXME does hard coding GPU_BLOCK_SIZE >= 2*SUB_GROUP_SIZE acutally save anything here?
                //        //if (GPU_BLOCK_SIZE >= 2*j) _c[local_idx] += _c[local_idx + j];
                //        _c[local_idx] += _c[local_idx + j];
                //    }
                //});
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
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
                });
            #endif

            //Set C
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t group_idx = wGroup.get_group_id(0);
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     //C_tmp[group_idx] = _c[0];
                     C[0] = _c[0];
                }
            });
        }));
    });
}

#endif
