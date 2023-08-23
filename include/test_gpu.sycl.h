#ifndef TEST_GPU_SYCL_H 
#define TEST_GPU_SYCL_H
#include "common_gpu.h"

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPU KERNELS ---------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void sub_group_reduce_add(volatile double* _c, size_t local_idx) {
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
}

template <int Dimensions = 1>
void gpu_reduce_add(double* _c, sycl::group<Dimensions> work_group) {
    #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 512) {
                _c[local_idx] += _c[local_idx + 512];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 256) {
                _c[local_idx] += _c[local_idx + 256];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 128) {
                _c[local_idx] += _c[local_idx + 128];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 64) {
                _c[local_idx] += _c[local_idx + 64];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 32) {
                _c[local_idx] += _c[local_idx + 32];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 16) {
                _c[local_idx] += _c[local_idx + 16];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 8) {
                _c[local_idx] += _c[local_idx + 8];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 4) {
                _c[local_idx] += _c[local_idx + 4];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 2) {
                _c[local_idx] += _c[local_idx + 2];
            }
        });
    #endif

    //Sub-group reduce
    work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
        size_t local_idx = index.get_local_id(0);
        if (local_idx < SUB_GROUP_SIZE) {
            sub_group_reduce_add(_c, local_idx);
        }
    });
}

template <int Dimensions = 1>
void gpu_reduce_add(double* _c, sycl::nd_item<Dimensions> item) {
    size_t local_idx = item.get_local_id(0);
    #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
        if (local_idx < 512) {
            _c[local_idx] += _c[local_idx + 512];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
        if (local_idx < 256) {
            _c[local_idx] += _c[local_idx + 256];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
        if (local_idx < 128) {
            _c[local_idx] += _c[local_idx + 128];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
        if (local_idx < 64) {
            _c[local_idx] += _c[local_idx + 64];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
        if (local_idx < 32) {
            _c[local_idx] += _c[local_idx + 32];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
        if (local_idx < 16) {
            _c[local_idx] += _c[local_idx + 16];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
        if (local_idx < 8) {
            _c[local_idx] += _c[local_idx + 8];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
        if (local_idx < 4) {
            _c[local_idx] += _c[local_idx + 4];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
        if (local_idx < 2) {
            _c[local_idx] += _c[local_idx + 2];
        }
        sycl::group_barrier(item.get_group());
    #endif

    //Sub-group reduce
    if (local_idx < SUB_GROUP_SIZE) {
        sub_group_reduce_add(_c, local_idx);
    }
    sycl::group_barrier(item.get_group());
}

void gpu_dot(sycl::queue q, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
    // C = B*A where [B] = 1xN and [A] = Nx1
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{1}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            double _c[GPU_BLOCK_SIZE];
            // Set shared local memory _c
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t local_idx = index.get_local_id(0);
                if (local_idx < N) {
                    _c[local_idx] = A[local_idx]*B[local_idx];
                } else {
                    _c[local_idx] = 0.0;
                }

                for (size_t i = 1; i < (N + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE; i++) {
                    size_t j = GPU_BLOCK_SIZE*i + local_idx;
                    if (j < N) {
                        _c[local_idx] += A[j]*B[j];
                    }
                }
            });

            // Reduce _c (using shared local memory)
            #ifdef SYCL_PASS_WORK_GROUP_WA
                // copy/paste code directly from gpu_reduce_add
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

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_add(_c, local_idx);
                    }
                });
            #else
                gpu_reduce_add(_c, wGroup);
            #endif

            //Set C
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     C[0] = _c[0];
                }
            });
        }));
    });
}

void gpu_dot_wgdp(sycl::queue q, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
    // C = B*A where [B] = 1xN and [A] = Nx1
    q.submit([&](sycl::handler& cgh) {
        // Shared Local Memory _c
        sycl::local_accessor<double, 1> _c(sycl::range(GPU_BLOCK_SIZE), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range(1), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            // Set shared local memory _c
            size_t local_idx = item.get_local_id(0);
            if (local_idx < N) {
                _c[local_idx] = A[local_idx]*B[local_idx];
            } else {
                _c[local_idx] = 0.0;
            }

            for (size_t i = 1; i < (N + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE; i++) {
                size_t j = GPU_BLOCK_SIZE*i + local_idx;
                if (j < N) {
                    _c[local_idx] += A[j]*B[j];
                }
            }
            sycl::group_barrier(item.get_group());

            // Reduce _c (using shared local memory)
            #ifdef SYCL_PASS_ITEM_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    if (local_idx < 512) {
                        _c[local_idx] += _c[local_idx + 512];
                    }
                    sycl::group_barrier(item.get_group());
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    if (local_idx < 256) {
                        _c[local_idx] += _c[local_idx + 256];
                    }
                    sycl::group_barrier(item.get_group());
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    if (local_idx < 128) {
                        _c[local_idx] += _c[local_idx + 128];
                    }
                    sycl::group_barrier(item.get_group());
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    if (local_idx < 64) {
                        _c[local_idx] += _c[local_idx + 64];
                    }
                    sycl::group_barrier(item.get_group());
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    if (local_idx < 32) {
                        _c[local_idx] += _c[local_idx + 32];
                    }
                    sycl::group_barrier(item.get_group());
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    if (local_idx < 16) {
                        _c[local_idx] += _c[local_idx + 16];
                    }
                    sycl::group_barrier(item.get_group());
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    if (local_idx < 8) {
                        _c[local_idx] += _c[local_idx + 8];
                    }
                    sycl::group_barrier(item.get_group());
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    if (local_idx < 4) {
                        _c[local_idx] += _c[local_idx + 4];
                    }
                    sycl::group_barrier(item.get_group());
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    if (local_idx < 2) {
                        _c[local_idx] += _c[local_idx + 2];
                    }
                    sycl::group_barrier(item.get_group());
                #endif

                //Sub-group reduce
                if (local_idx < SUB_GROUP_SIZE) {
                    //FIXME is get_pointer() right here?
                    //sub_group_reduce_add(_c, local_idx);
                    sub_group_reduce_add(_c.get_pointer(), local_idx);
                }
                sycl::group_barrier(item.get_group());
            #else
                //FIXME is get_pointer() right here?
                //gpu_reduce_add(_c, item);
                gpu_reduce_add(_c.get_pointer(), item);
            #endif

            //Set C
            if (local_idx == 0) {
                 C[0] = _c[0];
            }
        });
    });
}
#endif
