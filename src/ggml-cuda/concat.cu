#include "concat.cuh"
#include <cstdlib>

static bool ggml_cuda_concat_vec4_disabled() {
    static const bool disabled = []() {
        const char * env = std::getenv("ED_DISABLE_CUDA_CONCAT_VEC4");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return disabled;
}

// contiguous kernels
template <int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_f32_cont_4d(const float * x,
                                                                                    const float * y,
                                                                                    float *       dst,
                                                                                    int64_t       ne00,
                                                                                    int64_t       ne01,
                                                                                    int64_t       ne02,
                                                                                    int64_t       ne0,
                                                                                    int64_t       ne1,
                                                                                    int64_t       ne2,
                                                                                    int64_t       ne3) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t dst_block = ne0 * ne1 * ne2;
    const int64_t n         = dst_block * ne3;

    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t) blockDim.x * gridDim.x) {
        const int64_t i3 = idx / dst_block;
        const int64_t i  = idx - i3 * dst_block;

        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[idx] = x[i3 * (ne00 * ne1 * ne2) + row * ne00 + i0];
            } else {
                dst[idx] = y[i3 * ((ne0 - ne00) * ne1 * ne2) + row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[idx] = x[i3 * (src0_plane * ne2) + i2 * src0_plane + i01];
            } else {
                dst[idx] = y[i3 * (src1_plane * ne2) + i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[idx] = x[i3 * src0_size + i];
            } else {
                dst[idx] = y[i3 * (dst_block - src0_size) + (i - src0_size)];
            }
        }
    }
}

template <int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_f32_cont_4d_vec4(const float4 * x,
                                                                                         const float4 * y,
                                                                                         float4 *       dst,
                                                                                         int64_t        ne00,
                                                                                         int64_t        ne01,
                                                                                         int64_t        ne02,
                                                                                         int64_t        ne0,
                                                                                         int64_t        ne1,
                                                                                         int64_t        ne2,
                                                                                         int64_t        ne3) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t ne0_v      = ne0 / 4;
    const int64_t dst_block  = ne0_v * ne1 * ne2;
    const int64_t n          = dst_block * ne3;

    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t) blockDim.x * gridDim.x) {
        const int64_t i3 = idx / dst_block;
        const int64_t i  = idx - i3 * dst_block;

        if constexpr (dim == 0) {
            const int64_t ne00_v = ne00 / 4;
            const int64_t row    = i / ne0_v;
            const int64_t i0v    = i - row * ne0_v;

            if (i0v < ne00_v) {
                dst[idx] = x[i3 * (ne00_v * ne1 * ne2) + row * ne00_v + i0v];
            } else {
                dst[idx] = y[i3 * ((ne0_v - ne00_v) * ne1 * ne2) + row * (ne0_v - ne00_v) + (i0v - ne00_v)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0_v * ne1;
            const int64_t src0_plane = ne0_v * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[idx] = x[i3 * (src0_plane * ne2) + i2 * src0_plane + i01];
            } else {
                dst[idx] = y[i3 * (src1_plane * ne2) + i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0_v * ne1 * ne02;

            if (i < src0_size) {
                dst[idx] = x[i3 * src0_size + i];
            } else {
                dst[idx] = y[i3 * (dst_block - src0_size) + (i - src0_size)];
            }
        }
    }
}

static bool concat_f32_vec4_aligned(const float * x, const float * y, const float * dst) {
    return ((uintptr_t) x   % alignof(float4)) == 0 &&
           ((uintptr_t) y   % alignof(float4)) == 0 &&
           ((uintptr_t) dst % alignof(float4)) == 0;
}

static bool concat_f32_vec4_supported(int64_t ne00, int64_t ne0) {
    return (ne00 % 4) == 0 && (ne0 % 4) == 0;
}

static void concat_f32_cuda_4d(const float * x,
                               const float * y,
                               float *       dst,
                               int64_t       ne00,
                               int64_t       ne01,
                               int64_t       ne02,
                               int64_t       ne0,
                               int64_t       ne1,
                               int64_t       ne2,
                               int64_t       ne3,
                               int           dim,
                               cudaStream_t  stream) {
    const int64_t n          = ne0 * ne1 * ne2 * ne3;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (!ggml_cuda_concat_vec4_disabled() &&
        concat_f32_vec4_supported(ne00, ne0) &&
        concat_f32_vec4_aligned(x, y, dst)) {
        const int64_t n_v          = n / 4;
        const int     num_blocks_v = (n_v + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
        if (dim == 0) {
            concat_f32_cont_4d_vec4<0><<<num_blocks_v, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const float4 *) x, (const float4 *) y, (float4 *) dst, ne00, ne01, ne02, ne0, ne1, ne2, ne3);
            return;
        }
        if (dim == 1) {
            concat_f32_cont_4d_vec4<1><<<num_blocks_v, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const float4 *) x, (const float4 *) y, (float4 *) dst, ne00, ne01, ne02, ne0, ne1, ne2, ne3);
            return;
        }
        concat_f32_cont_4d_vec4<2><<<num_blocks_v, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
            (const float4 *) x, (const float4 *) y, (float4 *) dst, ne00, ne01, ne02, ne0, ne1, ne2, ne3);
        return;
    }

    if (dim == 0) {
        concat_f32_cont_4d<0>
            <<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2, ne3);
        return;
    }
    if (dim == 1) {
        concat_f32_cont_4d<1>
            <<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2, ne3);
        return;
    }
    concat_f32_cont_4d<2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2, ne3);
}

// non-contiguous kernel (slow)
template <int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_f32_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const float * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const float *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const float *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        float * y = (float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const float * src0_d = (const float *)src0->data;
        const float * src1_d = (const float *)src1->data;

        float * dst_d = (float *)dst->data;

        if (dim != 3) {
            concat_f32_cuda_4d(src0_d, src1_d, dst_d,
                               src0->ne[0], src0->ne[1], src0->ne[2],
                               dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                               dim, stream);
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync(dst_d,           src0_d, size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dst_d + size0/4, src1_d, size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_f32_non_cont<dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}
