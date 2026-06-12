#include "qwen-fused-qkv-pack.cuh"

#include "../ggml-impl.h"

#include <algorithm>
#include <cstring>

static constexpr uint64_t QWEN_FUSED_QKV_PACK_MAGIC = 0x5157454e46514b56ULL;

struct qwen_fused_qkv_pack_cuda_params {
    uint64_t magic;
    int64_t txt_real_seq;
    int64_t img_real_seq;
    int64_t mode;
    int64_t txt_padded_seq;
    int64_t img_padded_seq;
    int64_t world_size;
    int64_t stream_index;
};

bool ggml_cuda_is_qwen_fused_qkv_pack(const ggml_tensor* dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM ||
        (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
        return false;
    }
    ggml_custom_op_params params;
    memcpy(&params, dst->op_params, sizeof(params));
    if (params.userdata == nullptr) {
        return false;
    }
    const auto* qwen_params = static_cast<const qwen_fused_qkv_pack_cuda_params*>(params.userdata);
    return qwen_params->magic == QWEN_FUSED_QKV_PACK_MAGIC;
}

static __global__ void qwen_fused_qkv_pack_f32_kernel(const char* __restrict__ txt_q,
                                                      const char* __restrict__ img_q,
                                                      const char* __restrict__ txt_k,
                                                      const char* __restrict__ img_k,
                                                      const char* __restrict__ txt_v,
                                                      const char* __restrict__ img_v,
                                                      float* __restrict__ dst,
                                                      int64_t txt_seq,
                                                      int64_t img_seq,
                                                      int64_t heads,
                                                      int64_t head_dim,
                                                      size_t txt_q_nb0,
                                                      size_t txt_q_nb1,
                                                      size_t txt_q_nb2,
                                                      size_t txt_q_nb3,
                                                      size_t img_q_nb0,
                                                      size_t img_q_nb1,
                                                      size_t img_q_nb2,
                                                      size_t img_q_nb3,
                                                      size_t txt_k_nb0,
                                                      size_t txt_k_nb1,
                                                      size_t txt_k_nb2,
                                                      size_t txt_k_nb3,
                                                      size_t img_k_nb0,
                                                      size_t img_k_nb1,
                                                      size_t img_k_nb2,
                                                      size_t img_k_nb3,
                                                      size_t txt_v_nb0,
                                                      size_t txt_v_nb1,
                                                      size_t txt_v_nb2,
                                                      size_t txt_v_nb3,
                                                      size_t img_v_nb0,
                                                      size_t img_v_nb1,
                                                      size_t img_v_nb2,
                                                      size_t img_v_nb3) {
    const int64_t seq_total   = txt_seq + img_seq;
    const int64_t qk_half_dim = head_dim / 2;
    const int64_t plane_elems = qk_half_dim * seq_total * heads;
    const int64_t total       = 6 * plane_elems;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t plane = linear / plane_elems;
        int64_t rem         = linear - plane * plane_elems;
        const int64_t half  = rem % qk_half_dim;
        rem /= qk_half_dim;
        const int64_t tok   = rem % seq_total;
        const int64_t head  = rem / seq_total;
        const bool is_txt   = tok < txt_seq;
        const int64_t src_t = is_txt ? tok : tok - txt_seq;
        const int64_t qkv_plane = plane / 2;
        const int64_t part      = plane % 2;

        const char* src = nullptr;
        size_t nb0 = 0;
        size_t nb1 = 0;
        size_t nb2 = 0;
        size_t nb3 = 0;
        if (qkv_plane == 0) {
            src = is_txt ? txt_q : img_q;
            nb0 = is_txt ? txt_q_nb0 : img_q_nb0;
            nb1 = is_txt ? txt_q_nb1 : img_q_nb1;
            nb2 = is_txt ? txt_q_nb2 : img_q_nb2;
            nb3 = is_txt ? txt_q_nb3 : img_q_nb3;
        } else if (qkv_plane == 1) {
            src = is_txt ? txt_k : img_k;
            nb0 = is_txt ? txt_k_nb0 : img_k_nb0;
            nb1 = is_txt ? txt_k_nb1 : img_k_nb1;
            nb2 = is_txt ? txt_k_nb2 : img_k_nb2;
            nb3 = is_txt ? txt_k_nb3 : img_k_nb3;
        } else {
            src = is_txt ? txt_v : img_v;
            nb0 = is_txt ? txt_v_nb0 : img_v_nb0;
            nb1 = is_txt ? txt_v_nb1 : img_v_nb1;
            nb2 = is_txt ? txt_v_nb2 : img_v_nb2;
            nb3 = is_txt ? txt_v_nb3 : img_v_nb3;
        }

        float value = 0.0f;
        if (qkv_plane < 2) {
            value = *reinterpret_cast<const float*>(src + half * nb0 + src_t * nb1 + head * nb2 + part * nb3);
        } else {
            const int64_t d = half + part * qk_half_dim;
            value = *reinterpret_cast<const float*>(src + d * nb0 + src_t * nb1 + head * nb2);
        }
        dst[linear] = value;
    }
}

static __global__ void qwen_fused_qkv_pack_from_recv_f32_kernel(const char* __restrict__ txt_recv,
                                                                const char* __restrict__ img_recv,
                                                                float* __restrict__ dst,
                                                                int64_t txt_real_seq,
                                                                int64_t img_real_seq,
                                                                int64_t heads,
                                                                int64_t head_dim) {
    const int64_t seq_total   = txt_real_seq + img_real_seq;
    const int64_t qk_half_dim = head_dim / 2;
    const int64_t plane_elems = qk_half_dim * seq_total * heads;
    const int64_t total       = 6 * plane_elems;
    const int64_t recv_stride = 3 * head_dim * heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t plane = linear / plane_elems;
        int64_t rem         = linear - plane * plane_elems;
        const int64_t half  = rem % qk_half_dim;
        rem /= qk_half_dim;
        const int64_t tok   = rem % seq_total;
        const int64_t head  = rem / seq_total;
        const bool is_txt   = tok < txt_real_seq;
        const int64_t src_t = is_txt ? tok : tok - txt_real_seq;
        const int64_t qkv_plane = plane / 2;
        const int64_t part      = plane % 2;

        int64_t src_d = qkv_plane * head_dim;
        if (qkv_plane < 2) {
            src_d += part + 2 * half;
        } else {
            src_d += half + part * qk_half_dim;
        }
        const char* src = is_txt ? txt_recv : img_recv;
        const int64_t src_idx = src_d + head * (3 * head_dim) + src_t * recv_stride;
        dst[linear] = *reinterpret_cast<const float*>(src + src_idx * sizeof(float));
    }
}

static __global__ void qwen_fused_qk_pack_f32_kernel(const char* __restrict__ txt_q,
                                                     const char* __restrict__ img_q,
                                                     const char* __restrict__ txt_k,
                                                     const char* __restrict__ img_k,
                                                     float* __restrict__ dst,
                                                     int64_t txt_seq,
                                                     int64_t img_seq,
                                                     int64_t heads,
                                                     int64_t head_dim,
                                                     size_t txt_q_nb0,
                                                     size_t txt_q_nb1,
                                                     size_t txt_q_nb2,
                                                     size_t txt_q_nb3,
                                                     size_t img_q_nb0,
                                                     size_t img_q_nb1,
                                                     size_t img_q_nb2,
                                                     size_t img_q_nb3,
                                                     size_t txt_k_nb0,
                                                     size_t txt_k_nb1,
                                                     size_t txt_k_nb2,
                                                     size_t txt_k_nb3,
                                                     size_t img_k_nb0,
                                                     size_t img_k_nb1,
                                                     size_t img_k_nb2,
                                                     size_t img_k_nb3) {
    const int64_t seq_total   = txt_seq + img_seq;
    const int64_t qk_half_dim = head_dim / 2;
    const int64_t plane_elems = qk_half_dim * seq_total * heads;
    const int64_t total       = 4 * plane_elems;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t plane = linear / plane_elems;
        int64_t rem         = linear - plane * plane_elems;
        const int64_t half  = rem % qk_half_dim;
        rem /= qk_half_dim;
        const int64_t tok   = rem % seq_total;
        const int64_t head  = rem / seq_total;
        const bool is_txt   = tok < txt_seq;
        const int64_t src_t = is_txt ? tok : tok - txt_seq;
        const int64_t qk_plane = plane / 2;
        const int64_t part     = plane % 2;

        const char* src = nullptr;
        size_t nb0 = 0;
        size_t nb1 = 0;
        size_t nb2 = 0;
        size_t nb3 = 0;
        if (qk_plane == 0) {
            src = is_txt ? txt_q : img_q;
            nb0 = is_txt ? txt_q_nb0 : img_q_nb0;
            nb1 = is_txt ? txt_q_nb1 : img_q_nb1;
            nb2 = is_txt ? txt_q_nb2 : img_q_nb2;
            nb3 = is_txt ? txt_q_nb3 : img_q_nb3;
        } else {
            src = is_txt ? txt_k : img_k;
            nb0 = is_txt ? txt_k_nb0 : img_k_nb0;
            nb1 = is_txt ? txt_k_nb1 : img_k_nb1;
            nb2 = is_txt ? txt_k_nb2 : img_k_nb2;
            nb3 = is_txt ? txt_k_nb3 : img_k_nb3;
        }

        dst[linear] = *reinterpret_cast<const float*>(src + half * nb0 + src_t * nb1 + head * nb2 + part * nb3);
    }
}

static __global__ void qwen_fused_v_pack_f32_kernel(const char* __restrict__ txt_v,
                                                    const char* __restrict__ img_v,
                                                    float* __restrict__ dst,
                                                    int64_t txt_seq,
                                                    int64_t img_seq,
                                                    int64_t heads,
                                                    int64_t head_dim,
                                                    size_t txt_v_nb0,
                                                    size_t txt_v_nb1,
                                                    size_t txt_v_nb2,
                                                    size_t txt_v_nb3,
                                                    size_t img_v_nb0,
                                                    size_t img_v_nb1,
                                                    size_t img_v_nb2,
                                                    size_t img_v_nb3) {
    const int64_t seq_total = txt_seq + img_seq;
    const int64_t total     = head_dim * seq_total * heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem        = linear;
        const int64_t d    = rem % head_dim;
        rem /= head_dim;
        const int64_t tok  = rem % seq_total;
        const int64_t head = rem / seq_total;
        const bool is_txt  = tok < txt_seq;
        const int64_t src_t = is_txt ? tok : tok - txt_seq;
        const char* src = is_txt ? txt_v : img_v;
        const size_t nb0 = is_txt ? txt_v_nb0 : img_v_nb0;
        const size_t nb1 = is_txt ? txt_v_nb1 : img_v_nb1;
        const size_t nb2 = is_txt ? txt_v_nb2 : img_v_nb2;
        const size_t nb3 = is_txt ? txt_v_nb3 : img_v_nb3;

        dst[linear] = *reinterpret_cast<const float*>(src + d * nb0 + src_t * nb1 + head * nb2 + 0 * nb3);
    }
}

static __global__ void qwen_fused_qkv_send_pack_f32_kernel(const char* __restrict__ q,
                                                           const char* __restrict__ k,
                                                           const char* __restrict__ v,
                                                           float* __restrict__ dst,
                                                           int64_t world_size,
                                                           int64_t head_dim,
                                                           int64_t heads,
                                                           int64_t shard_sequence,
                                                           size_t q_nb0,
                                                           size_t q_nb1,
                                                           size_t q_nb2,
                                                           size_t q_nb3,
                                                           size_t k_nb0,
                                                           size_t k_nb1,
                                                           size_t k_nb2,
                                                           size_t k_nb3,
                                                           size_t v_nb0,
                                                           size_t v_nb1,
                                                           size_t v_nb2,
                                                           size_t v_nb3) {
    const int64_t shard_heads    = heads / world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t total          = total_head_dim * shard_heads * shard_sequence * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem         = linear;
        const int64_t d_all = rem % total_head_dim;
        rem /= total_head_dim;
        const int64_t head_local = rem % shard_heads;
        rem /= shard_heads;
        const int64_t seq = rem % shard_sequence;
        rem /= shard_sequence;
        const int64_t peer = rem;
        const int64_t head = head_local + peer * shard_heads;
        const int64_t qkv_plane = d_all / head_dim;
        const int64_t d = d_all - qkv_plane * head_dim;

        const char* src = q;
        size_t nb0 = q_nb0;
        size_t nb1 = q_nb1;
        size_t nb2 = q_nb2;
        size_t nb3 = q_nb3;
        if (qkv_plane == 1) {
            src = k;
            nb0 = k_nb0;
            nb1 = k_nb1;
            nb2 = k_nb2;
            nb3 = k_nb3;
        } else if (qkv_plane == 2) {
            src = v;
            nb0 = v_nb0;
            nb1 = v_nb1;
            nb2 = v_nb2;
            nb3 = v_nb3;
        }

        dst[linear] = *reinterpret_cast<const float*>(src + d * nb0 + head * nb1 + seq * nb2 + 0 * nb3);
    }
}

static __global__ void qwen_fused_qkv_send_pack_mixed_kernel(const char* __restrict__ q,
                                                             const char* __restrict__ k,
                                                             const char* __restrict__ v,
                                                             uint32_t* __restrict__ dst,
                                                             int64_t world_size,
                                                             int64_t head_dim,
                                                             int64_t heads,
                                                             int64_t shard_sequence,
                                                             size_t q_nb0,
                                                             size_t q_nb1,
                                                             size_t q_nb2,
                                                             size_t q_nb3,
                                                             size_t k_nb0,
                                                             size_t k_nb1,
                                                             size_t k_nb2,
                                                             size_t k_nb3,
                                                             size_t v_nb0,
                                                             size_t v_nb1,
                                                             size_t v_nb2,
                                                             size_t v_nb3) {
    const int64_t shard_heads = heads / world_size;
    const int64_t packed_dim = head_dim * 2;
    const int64_t total = packed_dim * shard_heads * shard_sequence * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int64_t d_all = rem % packed_dim;
        rem /= packed_dim;
        const int64_t head_local = rem % shard_heads;
        rem /= shard_heads;
        const int64_t seq = rem % shard_sequence;
        rem /= shard_sequence;
        const int64_t peer = rem;
        const int64_t head = head_local + peer * shard_heads;

        if (d_all < head_dim) {
            const float value = *reinterpret_cast<const float*>(q + d_all * q_nb0 + head * q_nb1 + seq * q_nb2 + 0 * q_nb3);
            dst[linear] = __float_as_uint(value);
        } else {
            const int64_t d = d_all - head_dim;
            const float k_value = *reinterpret_cast<const float*>(k + d * k_nb0 + head * k_nb1 + seq * k_nb2 + 0 * k_nb3);
            const float v_value = *reinterpret_cast<const float*>(v + d * v_nb0 + head * v_nb1 + seq * v_nb2 + 0 * v_nb3);
            const uint32_t k_half = static_cast<uint32_t>(__half_as_ushort(__float2half(k_value)));
            const uint32_t v_half = static_cast<uint32_t>(__half_as_ushort(__float2half(v_value)));
            dst[linear] = k_half | (v_half << 16);
        }
    }
}

static __global__ void qwen_fused_attn_head_to_seq_send_pack_f32_kernel(const char* __restrict__ attn,
                                                                        float* __restrict__ dst,
                                                                        int64_t txt_real_seq,
                                                                        int64_t img_real_seq,
                                                                        int64_t txt_padded_seq,
                                                                        int64_t img_padded_seq,
                                                                        int64_t world_size,
                                                                        int64_t head_dim,
                                                                        int64_t shard_heads,
                                                                        size_t attn_nb0,
                                                                        size_t attn_nb1,
                                                                        size_t attn_nb2,
                                                                        size_t attn_nb3) {
    const int64_t txt_shard_seq = txt_padded_seq / world_size;
    const int64_t img_shard_seq = img_padded_seq / world_size;
    const int64_t txt_chunk = head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk = head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    const int64_t total = count_per_peer * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int64_t peer = rem / count_per_peer;
        rem -= peer * count_per_peer;
        const bool is_img = rem >= txt_chunk;
        if (is_img) {
            rem -= txt_chunk;
        }
        const int64_t shard_seq = is_img ? img_shard_seq : txt_shard_seq;
        const int64_t stream_real_seq = is_img ? img_real_seq : txt_real_seq;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % shard_heads;
        rem /= shard_heads;
        const int64_t local_tok = rem;
        const int64_t stream_tok = peer * shard_seq + local_tok;

        float value = 0.0f;
        if (stream_tok < stream_real_seq) {
            const int64_t total_tok = is_img ? txt_real_seq + stream_tok : stream_tok;
            value = *reinterpret_cast<const float*>(attn +
                                                    d * attn_nb0 +
                                                    head * attn_nb1 +
                                                    total_tok * attn_nb2 +
                                                    0 * attn_nb3);
        }
        dst[linear] = value;
    }
}

static __global__ void qwen_fused_attn_head_to_seq_recv_unpack_f32_kernel(const char* __restrict__ recv_flat,
                                                                          float* __restrict__ dst,
                                                                          int64_t stream_index,
                                                                          int64_t txt_padded_seq,
                                                                          int64_t img_padded_seq,
                                                                          int64_t world_size,
                                                                          int64_t head_dim,
                                                                          int64_t heads) {
    const int64_t shard_heads = heads / world_size;
    const int64_t txt_shard_seq = txt_padded_seq / world_size;
    const int64_t img_shard_seq = img_padded_seq / world_size;
    const int64_t out_shard_seq = stream_index == 0 ? txt_shard_seq : img_shard_seq;
    const int64_t txt_chunk = head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk = head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    const int64_t stream_offset = stream_index == 0 ? 0 : txt_chunk;
    const int64_t total = head_dim * heads * out_shard_seq;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % heads;
        rem /= heads;
        const int64_t local_tok = rem;
        const int64_t src_peer = head / shard_heads;
        const int64_t local_head = head - src_peer * shard_heads;
        const int64_t src_idx =
            src_peer * count_per_peer +
            stream_offset +
            d +
            local_head * head_dim +
            local_tok * head_dim * shard_heads;
        dst[linear] = *reinterpret_cast<const float*>(recv_flat + src_idx * sizeof(float));
    }
}

static __global__ void qwen_fused_attn_head_to_seq_send_pack_f16_kernel(const char* __restrict__ attn,
                                                                        half* __restrict__ dst,
                                                                        int64_t txt_real_seq,
                                                                        int64_t img_real_seq,
                                                                        int64_t txt_padded_seq,
                                                                        int64_t img_padded_seq,
                                                                        int64_t world_size,
                                                                        int64_t head_dim,
                                                                        int64_t shard_heads,
                                                                        size_t attn_nb0,
                                                                        size_t attn_nb1,
                                                                        size_t attn_nb2,
                                                                        size_t attn_nb3) {
    const int64_t txt_shard_seq = txt_padded_seq / world_size;
    const int64_t img_shard_seq = img_padded_seq / world_size;
    const int64_t txt_chunk = head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk = head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    const int64_t total = count_per_peer * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int64_t peer = rem / count_per_peer;
        rem -= peer * count_per_peer;
        const bool is_img = rem >= txt_chunk;
        if (is_img) {
            rem -= txt_chunk;
        }
        const int64_t shard_seq = is_img ? img_shard_seq : txt_shard_seq;
        const int64_t stream_real_seq = is_img ? img_real_seq : txt_real_seq;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % shard_heads;
        rem /= shard_heads;
        const int64_t local_tok = rem;
        const int64_t stream_tok = peer * shard_seq + local_tok;

        float value = 0.0f;
        if (stream_tok < stream_real_seq) {
            const int64_t total_tok = is_img ? txt_real_seq + stream_tok : stream_tok;
            value = *reinterpret_cast<const float*>(attn +
                                                    d * attn_nb0 +
                                                    head * attn_nb1 +
                                                    total_tok * attn_nb2 +
                                                    0 * attn_nb3);
        }
        dst[linear] = __float2half(value);
    }
}

static __global__ void qwen_fused_attn_head_to_seq_recv_unpack_f16_kernel(const half* __restrict__ recv_flat,
                                                                          float* __restrict__ dst,
                                                                          int64_t stream_index,
                                                                          int64_t txt_padded_seq,
                                                                          int64_t img_padded_seq,
                                                                          int64_t world_size,
                                                                          int64_t head_dim,
                                                                          int64_t heads) {
    const int64_t shard_heads = heads / world_size;
    const int64_t txt_shard_seq = txt_padded_seq / world_size;
    const int64_t img_shard_seq = img_padded_seq / world_size;
    const int64_t out_shard_seq = stream_index == 0 ? txt_shard_seq : img_shard_seq;
    const int64_t txt_chunk = head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk = head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    const int64_t stream_offset = stream_index == 0 ? 0 : txt_chunk;
    const int64_t total = head_dim * heads * out_shard_seq;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % heads;
        rem /= heads;
        const int64_t local_tok = rem;
        const int64_t src_peer = head / shard_heads;
        const int64_t local_head = head - src_peer * shard_heads;
        const int64_t src_idx =
            src_peer * count_per_peer +
            stream_offset +
            d +
            local_head * head_dim +
            local_tok * head_dim * shard_heads;
        dst[linear] = __half2float(recv_flat[src_idx]);
    }
}

static __global__ void qwen_fused_two_stream_flat_send_pack_f32_kernel(const char* __restrict__ first,
                                                                       const char* __restrict__ second,
                                                                       float* __restrict__ dst,
                                                                       int64_t first_chunk,
                                                                       int64_t second_chunk,
                                                                       int64_t world_size) {
    const int64_t count_per_peer = first_chunk + second_chunk;
    const int64_t total          = count_per_peer * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t peer = linear / count_per_peer;
        const int64_t rem  = linear - peer * count_per_peer;
        if (rem < first_chunk) {
            dst[linear] = *reinterpret_cast<const float*>(first + (peer * first_chunk + rem) * sizeof(float));
        } else {
            const int64_t second_rem = rem - first_chunk;
            dst[linear] = *reinterpret_cast<const float*>(second + (peer * second_chunk + second_rem) * sizeof(float));
        }
    }
}

static __global__ void qwen_fused_two_stream_flat_recv_unpack_f32_kernel(const char* __restrict__ recv_flat,
                                                                         float* __restrict__ dst,
                                                                         int64_t stream_index,
                                                                         int64_t first_chunk,
                                                                         int64_t second_chunk,
                                                                         int64_t world_size) {
    const int64_t stream_chunk   = stream_index == 0 ? first_chunk : second_chunk;
    const int64_t stream_offset  = stream_index == 0 ? 0 : first_chunk;
    const int64_t count_per_peer = first_chunk + second_chunk;
    const int64_t total          = stream_chunk * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t peer = linear / stream_chunk;
        const int64_t rem  = linear - peer * stream_chunk;
        const int64_t src  = peer * count_per_peer + stream_offset + rem;
        dst[linear] = *reinterpret_cast<const float*>(recv_flat + src * sizeof(float));
    }
}

static __device__ __forceinline__ int64_t mmdit_joint_qkv_src_index(int64_t d,
                                                                    int64_t local_head,
                                                                    int64_t local_seq,
                                                                    int64_t peer,
                                                                    int64_t plane,
                                                                    int64_t head_dim,
                                                                    int64_t shard_heads,
                                                                    int64_t stream_offset,
                                                                    int64_t count_per_peer) {
    const int64_t total_head_dim = head_dim * 3;
    return peer * count_per_peer +
           stream_offset +
           plane * head_dim +
           d +
           local_head * total_head_dim +
           local_seq * total_head_dim * shard_heads;
}

static __global__ void mmdit_fused_joint_qkv_to_seq_major_f32_kernel(const char* __restrict__ recv_flat,
                                                                     float* __restrict__ dst,
                                                                     int64_t plane,
                                                                     int64_t context_real_seq,
                                                                     int64_t x_real_seq,
                                                                     int64_t context_full_seq,
                                                                     int64_t x_full_seq,
                                                                     int64_t world_size,
                                                                     int64_t head_dim,
                                                                     int64_t shard_heads) {
    const int64_t total_real_seq    = context_real_seq + x_real_seq;
    const int64_t context_shard_seq = context_full_seq / world_size;
    const int64_t x_shard_seq       = x_full_seq / world_size;
    const int64_t total_head_dim    = head_dim * 3;
    const int64_t context_chunk     = total_head_dim * shard_heads * context_shard_seq;
    const int64_t x_chunk           = total_head_dim * shard_heads * x_shard_seq;
    const int64_t count_per_peer    = context_chunk + x_chunk;
    const int64_t total             = head_dim * total_real_seq * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem           = linear;
        const int64_t d       = rem % head_dim;
        rem /= head_dim;
        const int64_t out_tok = rem % total_real_seq;
        rem /= total_real_seq;
        const int64_t out_head = rem;

        const bool is_x = out_tok >= context_real_seq;
        const int64_t stream_tok = is_x ? out_tok - context_real_seq : out_tok;
        const int64_t stream_shard_seq = is_x ? x_shard_seq : context_shard_seq;
        const int64_t stream_offset = is_x ? context_chunk : 0;

        const int64_t peer = stream_tok / stream_shard_seq;
        const int64_t local_seq = stream_tok - peer * stream_shard_seq;
        const int64_t local_head = out_head;
        if (local_seq >= stream_shard_seq || peer >= world_size) {
            dst[linear] = 0.0f;
            continue;
        }

        const int64_t src_idx = mmdit_joint_qkv_src_index(d,
                                                          local_head,
                                                          local_seq,
                                                          peer,
                                                          plane,
                                                          head_dim,
                                                          shard_heads,
                                                          stream_offset,
                                                          count_per_peer);
        dst[linear] = *reinterpret_cast<const float*>(recv_flat + src_idx * sizeof(float));
    }
}

static __global__ void mmdit_fused_joint_qkv_v_to_seq_major_f32_kernel(const char* __restrict__ recv_flat,
                                                                       float* __restrict__ dst,
                                                                       int64_t context_real_seq,
                                                                       int64_t x_real_seq,
                                                                       int64_t context_full_seq,
                                                                       int64_t x_full_seq,
                                                                       int64_t world_size,
                                                                       int64_t head_dim,
                                                                       int64_t shard_heads) {
    const int64_t total_real_seq    = context_real_seq + x_real_seq;
    const int64_t context_shard_seq = context_full_seq / world_size;
    const int64_t x_shard_seq       = x_full_seq / world_size;
    const int64_t total_head_dim    = head_dim * 3;
    const int64_t context_chunk     = total_head_dim * shard_heads * context_shard_seq;
    const int64_t x_chunk           = total_head_dim * shard_heads * x_shard_seq;
    const int64_t count_per_peer    = context_chunk + x_chunk;
    const int64_t total             = head_dim * shard_heads * total_real_seq;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem            = linear;
        const int64_t d        = rem % head_dim;
        rem /= head_dim;
        const int64_t out_tok  = rem % total_real_seq;
        rem /= total_real_seq;
        const int64_t out_head = rem;

        const bool is_x = out_tok >= context_real_seq;
        const int64_t stream_tok = is_x ? out_tok - context_real_seq : out_tok;
        const int64_t stream_shard_seq = is_x ? x_shard_seq : context_shard_seq;
        const int64_t stream_offset = is_x ? context_chunk : 0;

        const int64_t local_head = out_head;
        const int64_t local_seq = stream_tok % stream_shard_seq;
        const int64_t peer = stream_tok / stream_shard_seq;
        if (peer >= world_size) {
            dst[linear] = 0.0f;
            continue;
        }

        const int64_t src_idx = mmdit_joint_qkv_src_index(d,
                                                          local_head,
                                                          local_seq,
                                                          peer,
                                                          2,
                                                          head_dim,
                                                          shard_heads,
                                                          stream_offset,
                                                          count_per_peer);
        dst[linear] = *reinterpret_cast<const float*>(recv_flat + src_idx * sizeof(float));
    }
}

static __global__ void mmdit_fused_joint_mixed_q_to_seq_major_kernel(const uint32_t* __restrict__ recv_flat,
                                                                     float* __restrict__ dst,
                                                                     int64_t context_real_seq,
                                                                     int64_t x_real_seq,
                                                                     int64_t context_full_seq,
                                                                     int64_t x_full_seq,
                                                                     int64_t world_size,
                                                                     int64_t head_dim,
                                                                     int64_t shard_heads) {
    const int64_t total_real_seq = context_real_seq + x_real_seq;
    const int64_t context_shard_seq = context_full_seq / world_size;
    const int64_t x_shard_seq = x_full_seq / world_size;
    const int64_t packed_dim = head_dim * 2;
    const int64_t context_chunk = packed_dim * shard_heads * context_shard_seq;
    const int64_t x_chunk = packed_dim * shard_heads * x_shard_seq;
    const int64_t count_per_peer = context_chunk + x_chunk;
    const int64_t total = head_dim * total_real_seq * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t out_tok = rem % total_real_seq;
        rem /= total_real_seq;
        const int64_t out_head = rem;

        const bool is_x = out_tok >= context_real_seq;
        const int64_t stream_tok = is_x ? out_tok - context_real_seq : out_tok;
        const int64_t stream_full_seq = is_x ? x_full_seq : context_full_seq;
        const int64_t stream_offset = is_x ? context_chunk : 0;

        const int64_t row = stream_tok + out_head * stream_full_seq;
        const int64_t peer = row % world_size;
        int64_t row_rem = row / world_size;
        const int64_t local_head = row_rem % shard_heads;
        const int64_t local_seq = row_rem / shard_heads;

        const int64_t src_idx =
            peer * count_per_peer +
            stream_offset +
            d +
            local_head * packed_dim +
            local_seq * packed_dim * shard_heads;
        dst[linear] = __uint_as_float(recv_flat[src_idx]);
    }
}

static __global__ void mmdit_fused_joint_mixed_kv_to_seq_major_kernel(const uint32_t* __restrict__ recv_flat,
                                                                      half* __restrict__ dst,
                                                                      bool unpack_v,
                                                                      int64_t context_real_seq,
                                                                      int64_t x_real_seq,
                                                                      int64_t context_full_seq,
                                                                      int64_t x_full_seq,
                                                                      int64_t world_size,
                                                                      int64_t head_dim,
                                                                      int64_t shard_heads) {
    const int64_t total_real_seq = context_real_seq + x_real_seq;
    const int64_t context_shard_seq = context_full_seq / world_size;
    const int64_t x_shard_seq = x_full_seq / world_size;
    const int64_t packed_dim = head_dim * 2;
    const int64_t context_chunk = packed_dim * shard_heads * context_shard_seq;
    const int64_t x_chunk = packed_dim * shard_heads * x_shard_seq;
    const int64_t count_per_peer = context_chunk + x_chunk;
    const int64_t total = head_dim * total_real_seq * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t out_tok = rem % total_real_seq;
        rem /= total_real_seq;
        const int64_t out_head = rem;

        const bool is_x = out_tok >= context_real_seq;
        const int64_t stream_tok = is_x ? out_tok - context_real_seq : out_tok;
        const int64_t stream_shard_seq = is_x ? x_shard_seq : context_shard_seq;
        const int64_t stream_offset = is_x ? context_chunk : 0;

        int64_t local_head;
        int64_t local_seq;
        int64_t peer;
        if (unpack_v) {
            const int64_t row = out_head + stream_tok * shard_heads;
            local_head = row % shard_heads;
            int64_t row_rem = row / shard_heads;
            local_seq = row_rem % stream_shard_seq;
            peer = row_rem / stream_shard_seq;
        } else {
            peer = stream_tok / stream_shard_seq;
            local_seq = stream_tok - peer * stream_shard_seq;
            local_head = out_head;
        }

        const int64_t src_idx =
            peer * count_per_peer +
            stream_offset +
            head_dim + d +
            local_head * packed_dim +
            local_seq * packed_dim * shard_heads;
        const uint32_t packed = recv_flat[src_idx];
        dst[linear] = __ushort_as_half(static_cast<unsigned short>(unpack_v ? (packed >> 16) : (packed & 0xffffu)));
    }
}

static __device__ __forceinline__ int64_t mmdit_joint_mixed_linear_src_index(int64_t d,
                                                                             int64_t local_head,
                                                                             int64_t local_seq,
                                                                             int64_t peer,
                                                                             int64_t head_dim,
                                                                             int64_t shard_heads,
                                                                             int64_t stream_offset,
                                                                             int64_t count_per_peer) {
    const int64_t packed_dim = head_dim * 2;
    return peer * count_per_peer +
           stream_offset +
           d +
           local_head * packed_dim +
           local_seq * packed_dim * shard_heads;
}

static __global__ void mmdit_fused_joint_mixed_linear_q_to_seq_major_kernel(const uint32_t* __restrict__ recv_flat,
                                                                           float* __restrict__ dst,
                                                                           int64_t context_real_seq,
                                                                           int64_t x_real_seq,
                                                                           int64_t context_full_seq,
                                                                           int64_t x_full_seq,
                                                                           int64_t world_size,
                                                                           int64_t head_dim,
                                                                           int64_t shard_heads) {
    const int64_t total_real_seq    = context_real_seq + x_real_seq;
    const int64_t context_shard_seq = context_full_seq / world_size;
    const int64_t x_shard_seq       = x_full_seq / world_size;
    const int64_t packed_dim        = head_dim * 2;
    const int64_t context_chunk     = packed_dim * shard_heads * context_shard_seq;
    const int64_t x_chunk           = packed_dim * shard_heads * x_shard_seq;
    const int64_t count_per_peer    = context_chunk + x_chunk;
    const int64_t total             = head_dim * total_real_seq * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem            = linear;
        const int64_t d        = rem % head_dim;
        rem /= head_dim;
        const int64_t out_tok  = rem % total_real_seq;
        rem /= total_real_seq;
        const int64_t out_head = rem;

        const bool is_x = out_tok >= context_real_seq;
        const int64_t stream_tok       = is_x ? out_tok - context_real_seq : out_tok;
        const int64_t stream_shard_seq = is_x ? x_shard_seq : context_shard_seq;
        const int64_t stream_offset    = is_x ? context_chunk : 0;
        const int64_t peer             = stream_tok / stream_shard_seq;
        const int64_t local_seq        = stream_tok - peer * stream_shard_seq;
        const int64_t local_head       = out_head;

        const int64_t src_idx = mmdit_joint_mixed_linear_src_index(d,
                                                                   local_head,
                                                                   local_seq,
                                                                   peer,
                                                                   head_dim,
                                                                   shard_heads,
                                                                   stream_offset,
                                                                   count_per_peer);
        dst[linear] = __uint_as_float(recv_flat[src_idx]);
    }
}

static __global__ void mmdit_fused_joint_mixed_linear_kv_to_seq_major_kernel(const uint32_t* __restrict__ recv_flat,
                                                                             half* __restrict__ dst,
                                                                             bool unpack_v,
                                                                             int64_t context_real_seq,
                                                                             int64_t x_real_seq,
                                                                             int64_t context_full_seq,
                                                                             int64_t x_full_seq,
                                                                             int64_t world_size,
                                                                             int64_t head_dim,
                                                                             int64_t shard_heads) {
    const int64_t total_real_seq    = context_real_seq + x_real_seq;
    const int64_t context_shard_seq = context_full_seq / world_size;
    const int64_t x_shard_seq       = x_full_seq / world_size;
    const int64_t packed_dim        = head_dim * 2;
    const int64_t context_chunk     = packed_dim * shard_heads * context_shard_seq;
    const int64_t x_chunk           = packed_dim * shard_heads * x_shard_seq;
    const int64_t count_per_peer    = context_chunk + x_chunk;
    const int64_t total             = head_dim * total_real_seq * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem            = linear;
        const int64_t d        = rem % head_dim;
        rem /= head_dim;
        const int64_t out_tok  = rem % total_real_seq;
        rem /= total_real_seq;
        const int64_t out_head = rem;

        const bool is_x = out_tok >= context_real_seq;
        const int64_t stream_tok       = is_x ? out_tok - context_real_seq : out_tok;
        const int64_t stream_shard_seq = is_x ? x_shard_seq : context_shard_seq;
        const int64_t stream_offset    = is_x ? context_chunk : 0;
        const int64_t peer             = stream_tok / stream_shard_seq;
        const int64_t local_seq        = stream_tok - peer * stream_shard_seq;
        const int64_t local_head       = out_head;

        const int64_t src_idx = mmdit_joint_mixed_linear_src_index(head_dim + d,
                                                                   local_head,
                                                                   local_seq,
                                                                   peer,
                                                                   head_dim,
                                                                   shard_heads,
                                                                   stream_offset,
                                                                   count_per_peer);
        const uint32_t packed = recv_flat[src_idx];
        dst[linear] = __ushort_as_half(static_cast<unsigned short>(unpack_v ? (packed >> 16) : (packed & 0xffffu)));
    }
}

static __global__ void qwen_fused_qk_pack_from_recv_f32_kernel(const char* __restrict__ txt_recv,
                                                               const char* __restrict__ img_recv,
                                                               float* __restrict__ dst,
                                                               int64_t txt_real_seq,
                                                               int64_t img_real_seq,
                                                               int64_t heads,
                                                               int64_t head_dim) {
    const int64_t seq_total   = txt_real_seq + img_real_seq;
    const int64_t qk_half_dim = head_dim / 2;
    const int64_t plane_elems = qk_half_dim * seq_total * heads;
    const int64_t total       = 4 * plane_elems;
    const int64_t recv_stride = 3 * head_dim * heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t plane = linear / plane_elems;
        int64_t rem         = linear - plane * plane_elems;
        const int64_t half  = rem % qk_half_dim;
        rem /= qk_half_dim;
        const int64_t tok   = rem % seq_total;
        const int64_t head  = rem / seq_total;
        const bool is_txt   = tok < txt_real_seq;
        const int64_t src_t = is_txt ? tok : tok - txt_real_seq;
        const int64_t qkv_plane = plane / 2;
        const int64_t part      = plane % 2;
        const int64_t src_d = qkv_plane * head_dim + part + 2 * half;
        const char* src = is_txt ? txt_recv : img_recv;
        const int64_t src_idx = src_d + head * (3 * head_dim) + src_t * recv_stride;
        dst[linear] = *reinterpret_cast<const float*>(src + src_idx * sizeof(float));
    }
}

static __global__ void qwen_fused_v_pack_from_recv_f32_kernel(const char* __restrict__ txt_recv,
                                                              const char* __restrict__ img_recv,
                                                              float* __restrict__ dst,
                                                              int64_t txt_real_seq,
                                                              int64_t img_real_seq,
                                                              int64_t heads,
                                                              int64_t head_dim) {
    const int64_t seq_total   = txt_real_seq + img_real_seq;
    const int64_t total       = head_dim * seq_total * heads;
    const int64_t recv_stride = 3 * head_dim * heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem        = linear;
        const int64_t d    = rem % head_dim;
        rem /= head_dim;
        const int64_t tok  = rem % seq_total;
        const int64_t head = rem / seq_total;
        const bool is_txt  = tok < txt_real_seq;
        const int64_t src_t = is_txt ? tok : tok - txt_real_seq;
        const char* src = is_txt ? txt_recv : img_recv;
        const int64_t src_d = 2 * head_dim + d;
        const int64_t src_idx = src_d + head * (3 * head_dim) + src_t * recv_stride;
        dst[linear] = *reinterpret_cast<const float*>(src + src_idx * sizeof(float));
    }
}

void ggml_cuda_op_qwen_fused_qkv_pack(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    ggml_custom_op_params op_params;
    memcpy(&op_params, dst->op_params, sizeof(op_params));
    const auto* qwen_params = static_cast<const qwen_fused_qkv_pack_cuda_params*>(op_params.userdata);
    GGML_ASSERT(qwen_params != nullptr);
    GGML_ASSERT(qwen_params->magic == QWEN_FUSED_QKV_PACK_MAGIC);

    if (qwen_params->mode == 9) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* first  = dst->src[0];
        const ggml_tensor* second = dst->src[1];
        GGML_ASSERT(first && second);
        GGML_ASSERT(first->type == GGML_TYPE_F32 && second->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->txt_real_seq > 0);
        GGML_ASSERT(qwen_params->img_real_seq > 0);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(first->ne[0] == qwen_params->txt_real_seq * qwen_params->world_size);
        GGML_ASSERT(second->ne[0] == qwen_params->img_real_seq * qwen_params->world_size);
        GGML_ASSERT(ggml_nelements(dst) == (qwen_params->txt_real_seq + qwen_params->img_real_seq) *
                                           qwen_params->world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        qwen_fused_two_stream_flat_send_pack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(first->data),
            static_cast<const char*>(second->data),
            static_cast<float*>(dst->data),
            qwen_params->txt_real_seq,
            qwen_params->img_real_seq,
            qwen_params->world_size);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 10) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->stream_index == 0 || qwen_params->stream_index == 1);
        GGML_ASSERT(qwen_params->txt_real_seq > 0);
        GGML_ASSERT(qwen_params->img_real_seq > 0);
        GGML_ASSERT(qwen_params->world_size > 0);
        const int64_t stream_chunk = qwen_params->stream_index == 0 ? qwen_params->txt_real_seq :
                                                                    qwen_params->img_real_seq;
        GGML_ASSERT(ggml_nelements(dst) == stream_chunk * qwen_params->world_size);
        GGML_ASSERT(recv_flat->ne[0] == (qwen_params->txt_real_seq + qwen_params->img_real_seq) *
                                      qwen_params->world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        qwen_fused_two_stream_flat_recv_unpack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(recv_flat->data),
            static_cast<float*>(dst->data),
            qwen_params->stream_index,
            qwen_params->txt_real_seq,
            qwen_params->img_real_seq,
            qwen_params->world_size);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 11) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->stream_index == 0 || qwen_params->stream_index == 1);
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq >= qwen_params->txt_real_seq);
        GGML_ASSERT(qwen_params->img_padded_seq >= qwen_params->img_real_seq);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] == qwen_params->txt_real_seq + qwen_params->img_real_seq);

        const int64_t context_shard_seq = qwen_params->txt_padded_seq / qwen_params->world_size;
        const int64_t x_shard_seq = qwen_params->img_padded_seq / qwen_params->world_size;
        const int64_t total_head_dim = dst->ne[0] * 3;
        const int64_t expected = (total_head_dim * dst->ne[2] * context_shard_seq +
                                  total_head_dim * dst->ne[2] * x_shard_seq) *
                                 qwen_params->world_size;
        GGML_ASSERT(recv_flat->ne[0] == expected);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        mmdit_fused_joint_qkv_to_seq_major_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(recv_flat->data),
            static_cast<float*>(dst->data),
            qwen_params->stream_index,
            qwen_params->txt_real_seq,
            qwen_params->img_real_seq,
            qwen_params->txt_padded_seq,
            qwen_params->img_padded_seq,
            qwen_params->world_size,
            dst->ne[0],
            dst->ne[2]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 12) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq >= qwen_params->txt_real_seq);
        GGML_ASSERT(qwen_params->img_padded_seq >= qwen_params->img_real_seq);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] == qwen_params->txt_real_seq + qwen_params->img_real_seq);

        const int64_t context_shard_seq = qwen_params->txt_padded_seq / qwen_params->world_size;
        const int64_t x_shard_seq = qwen_params->img_padded_seq / qwen_params->world_size;
        const int64_t total_head_dim = dst->ne[0] * 3;
        const int64_t expected = (total_head_dim * dst->ne[2] * context_shard_seq +
                                  total_head_dim * dst->ne[2] * x_shard_seq) *
                                 qwen_params->world_size;
        GGML_ASSERT(recv_flat->ne[0] == expected);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        mmdit_fused_joint_qkv_v_to_seq_major_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(recv_flat->data),
            static_cast<float*>(dst->data),
            qwen_params->txt_real_seq,
            qwen_params->img_real_seq,
            qwen_params->txt_padded_seq,
            qwen_params->img_padded_seq,
            qwen_params->world_size,
            dst->ne[0],
            dst->ne[2]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 15) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq >= qwen_params->txt_real_seq);
        GGML_ASSERT(qwen_params->img_padded_seq >= qwen_params->img_real_seq);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] == qwen_params->txt_real_seq + qwen_params->img_real_seq);

        const int64_t context_shard_seq = qwen_params->txt_padded_seq / qwen_params->world_size;
        const int64_t x_shard_seq = qwen_params->img_padded_seq / qwen_params->world_size;
        const int64_t packed_dim = dst->ne[0] * 2;
        const int64_t expected = (packed_dim * dst->ne[2] * context_shard_seq +
                                  packed_dim * dst->ne[2] * x_shard_seq) *
                                 qwen_params->world_size;
        GGML_ASSERT(recv_flat->ne[0] == expected);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        mmdit_fused_joint_mixed_q_to_seq_major_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint32_t*>(recv_flat->data),
            static_cast<float*>(dst->data),
            qwen_params->txt_real_seq,
            qwen_params->img_real_seq,
            qwen_params->txt_padded_seq,
            qwen_params->img_padded_seq,
            qwen_params->world_size,
            dst->ne[0],
            dst->ne[2]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 16 || qwen_params->mode == 17) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq >= qwen_params->txt_real_seq);
        GGML_ASSERT(qwen_params->img_padded_seq >= qwen_params->img_real_seq);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] == qwen_params->txt_real_seq + qwen_params->img_real_seq);

        const int64_t context_shard_seq = qwen_params->txt_padded_seq / qwen_params->world_size;
        const int64_t x_shard_seq = qwen_params->img_padded_seq / qwen_params->world_size;
        const int64_t packed_dim = dst->ne[0] * 2;
        const int64_t expected = (packed_dim * dst->ne[2] * context_shard_seq +
                                  packed_dim * dst->ne[2] * x_shard_seq) *
                                 qwen_params->world_size;
        GGML_ASSERT(recv_flat->ne[0] == expected);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        mmdit_fused_joint_mixed_kv_to_seq_major_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint32_t*>(recv_flat->data),
            static_cast<half*>(dst->data),
            qwen_params->mode == 17,
            qwen_params->txt_real_seq,
            qwen_params->img_real_seq,
            qwen_params->txt_padded_seq,
            qwen_params->img_padded_seq,
            qwen_params->world_size,
            dst->ne[0],
            dst->ne[2]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 19) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq >= qwen_params->txt_real_seq);
        GGML_ASSERT(qwen_params->img_padded_seq >= qwen_params->img_real_seq);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] == qwen_params->txt_real_seq + qwen_params->img_real_seq);

        const int64_t context_shard_seq = qwen_params->txt_padded_seq / qwen_params->world_size;
        const int64_t x_shard_seq = qwen_params->img_padded_seq / qwen_params->world_size;
        const int64_t packed_dim = dst->ne[0] * 2;
        const int64_t expected = (packed_dim * dst->ne[2] * context_shard_seq +
                                  packed_dim * dst->ne[2] * x_shard_seq) *
                                 qwen_params->world_size;
        GGML_ASSERT(recv_flat->ne[0] == expected);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        mmdit_fused_joint_mixed_linear_q_to_seq_major_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint32_t*>(recv_flat->data),
            static_cast<float*>(dst->data),
            qwen_params->txt_real_seq,
            qwen_params->img_real_seq,
            qwen_params->txt_padded_seq,
            qwen_params->img_padded_seq,
            qwen_params->world_size,
            dst->ne[0],
            dst->ne[2]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 20 || qwen_params->mode == 21) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq >= qwen_params->txt_real_seq);
        GGML_ASSERT(qwen_params->img_padded_seq >= qwen_params->img_real_seq);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] == qwen_params->txt_real_seq + qwen_params->img_real_seq);

        const int64_t context_shard_seq = qwen_params->txt_padded_seq / qwen_params->world_size;
        const int64_t x_shard_seq = qwen_params->img_padded_seq / qwen_params->world_size;
        const int64_t packed_dim = dst->ne[0] * 2;
        const int64_t expected = (packed_dim * dst->ne[2] * context_shard_seq +
                                  packed_dim * dst->ne[2] * x_shard_seq) *
                                 qwen_params->world_size;
        GGML_ASSERT(recv_flat->ne[0] == expected);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        mmdit_fused_joint_mixed_linear_kv_to_seq_major_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint32_t*>(recv_flat->data),
            static_cast<half*>(dst->data),
            qwen_params->mode == 21,
            qwen_params->txt_real_seq,
            qwen_params->img_real_seq,
            qwen_params->txt_padded_seq,
            qwen_params->img_padded_seq,
            qwen_params->world_size,
            dst->ne[0],
            dst->ne[2]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 7) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* attn = dst->src[0];
        GGML_ASSERT(attn);
        GGML_ASSERT(attn->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq >= qwen_params->txt_real_seq);
        GGML_ASSERT(qwen_params->img_padded_seq >= qwen_params->img_real_seq);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(attn->ne[2] == qwen_params->txt_real_seq + qwen_params->img_real_seq);
        GGML_ASSERT(attn->ne[3] == 1);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        qwen_fused_attn_head_to_seq_send_pack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(attn->data),
            static_cast<float*>(dst->data),
            qwen_params->txt_real_seq,
            qwen_params->img_real_seq,
            qwen_params->txt_padded_seq,
            qwen_params->img_padded_seq,
            qwen_params->world_size,
            attn->ne[0],
            attn->ne[1],
            attn->nb[0], attn->nb[1], attn->nb[2], attn->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 8) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->stream_index == 0 || qwen_params->stream_index == 1);
        GGML_ASSERT(qwen_params->txt_padded_seq > 0 && qwen_params->img_padded_seq > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] % qwen_params->world_size == 0);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        qwen_fused_attn_head_to_seq_recv_unpack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(recv_flat->data),
            static_cast<float*>(dst->data),
            qwen_params->stream_index,
            qwen_params->txt_padded_seq,
            qwen_params->img_padded_seq,
            qwen_params->world_size,
            dst->ne[0],
            dst->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 13) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* attn = dst->src[0];
        GGML_ASSERT(attn);
        GGML_ASSERT(attn->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq >= qwen_params->txt_real_seq);
        GGML_ASSERT(qwen_params->img_padded_seq >= qwen_params->img_real_seq);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(attn->ne[2] == qwen_params->txt_real_seq + qwen_params->img_real_seq);
        GGML_ASSERT(attn->ne[3] == 1);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        qwen_fused_attn_head_to_seq_send_pack_f16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(attn->data),
            static_cast<half*>(dst->data),
            qwen_params->txt_real_seq,
            qwen_params->img_real_seq,
            qwen_params->txt_padded_seq,
            qwen_params->img_padded_seq,
            qwen_params->world_size,
            attn->ne[0],
            attn->ne[1],
            attn->nb[0], attn->nb[1], attn->nb[2], attn->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 14) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F16);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->stream_index == 0 || qwen_params->stream_index == 1);
        GGML_ASSERT(qwen_params->txt_padded_seq > 0 && qwen_params->img_padded_seq > 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] % qwen_params->world_size == 0);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        qwen_fused_attn_head_to_seq_recv_unpack_f16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const half*>(recv_flat->data),
            static_cast<float*>(dst->data),
            qwen_params->stream_index,
            qwen_params->txt_padded_seq,
            qwen_params->img_padded_seq,
            qwen_params->world_size,
            dst->ne[0],
            dst->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 6) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k = dst->src[1];
        const ggml_tensor* v = dst->src[2];
        GGML_ASSERT(q && k && v);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(q->ne[1] % world_size == 0);
        GGML_ASSERT(k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0]);
        GGML_ASSERT(k->ne[1] == q->ne[1] && v->ne[1] == q->ne[1]);
        GGML_ASSERT(k->ne[2] == q->ne[2] && v->ne[2] == q->ne[2]);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        qwen_fused_qkv_send_pack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(q->data),
            static_cast<const char*>(k->data),
            static_cast<const char*>(v->data),
            static_cast<float*>(dst->data),
            world_size,
            q->ne[0],
            q->ne[1],
            q->ne[2],
            q->nb[0], q->nb[1], q->nb[2], q->nb[3],
            k->nb[0], k->nb[1], k->nb[2], k->nb[3],
            v->nb[0], v->nb[1], v->nb[2], v->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 18) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k = dst->src[1];
        const ggml_tensor* v = dst->src[2];
        GGML_ASSERT(q && k && v);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(q->ne[1] % world_size == 0);
        GGML_ASSERT(k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0]);
        GGML_ASSERT(k->ne[1] == q->ne[1] && v->ne[1] == q->ne[1]);
        GGML_ASSERT(k->ne[2] == q->ne[2] && v->ne[2] == q->ne[2]);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        qwen_fused_qkv_send_pack_mixed_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(q->data),
            static_cast<const char*>(k->data),
            static_cast<const char*>(v->data),
            static_cast<uint32_t*>(dst->data),
            world_size,
            q->ne[0],
            q->ne[1],
            q->ne[2],
            q->nb[0], q->nb[1], q->nb[2], q->nb[3],
            k->nb[0], k->nb[1], k->nb[2], k->nb[3],
            v->nb[0], v->nb[1], v->nb[2], v->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode >= 3) {
        const ggml_tensor* txt_recv = dst->src[0];
        const ggml_tensor* img_recv = dst->src[1];
        GGML_ASSERT(txt_recv && img_recv);
        GGML_ASSERT(txt_recv->type == GGML_TYPE_F32 && img_recv->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->ne[1] == qwen_params->txt_real_seq + qwen_params->img_real_seq);
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq > 0);
        const int64_t head_dim = dst->ne[3] == 1 ? dst->ne[0] : dst->ne[0] * 2;
        const int64_t heads = dst->ne[2];
        GGML_ASSERT(txt_recv->ne[0] % (3 * head_dim * heads) == 0);
        GGML_ASSERT(img_recv->ne[0] % (3 * head_dim * heads) == 0);
        GGML_ASSERT(qwen_params->txt_real_seq <= txt_recv->ne[0] / (3 * head_dim * heads));
        GGML_ASSERT(qwen_params->img_real_seq <= img_recv->ne[0] / (3 * head_dim * heads));

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        if (dst->ne[3] == 6) {
            GGML_ASSERT(dst->type == GGML_TYPE_F32);
            qwen_fused_qkv_pack_from_recv_f32_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const char*>(txt_recv->data),
                static_cast<const char*>(img_recv->data),
                static_cast<float*>(dst->data),
                qwen_params->txt_real_seq,
                qwen_params->img_real_seq,
                heads,
                head_dim);
        } else if (dst->ne[3] == 4) {
            GGML_ASSERT(dst->type == GGML_TYPE_F32);
            qwen_fused_qk_pack_from_recv_f32_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const char*>(txt_recv->data),
                static_cast<const char*>(img_recv->data),
                static_cast<float*>(dst->data),
                qwen_params->txt_real_seq,
                qwen_params->img_real_seq,
                heads,
                head_dim);
        } else {
            GGML_ASSERT(dst->ne[3] == 1);
            qwen_fused_v_pack_from_recv_f32_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const char*>(txt_recv->data),
                static_cast<const char*>(img_recv->data),
                static_cast<float*>(dst->data),
                qwen_params->txt_real_seq,
                qwen_params->img_real_seq,
                heads,
                head_dim);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 1) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* txt_q = dst->src[0];
        const ggml_tensor* img_q = dst->src[1];
        const ggml_tensor* txt_k = dst->src[2];
        const ggml_tensor* img_k = dst->src[3];
        GGML_ASSERT(txt_q && img_q && txt_k && img_k);
        GGML_ASSERT(txt_q->type == GGML_TYPE_F32 && img_q->type == GGML_TYPE_F32);
        GGML_ASSERT(txt_k->type == GGML_TYPE_F32 && img_k->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->ne[0] == txt_q->ne[0] && dst->ne[0] == img_q->ne[0]);
        GGML_ASSERT(dst->ne[0] == txt_k->ne[0] && dst->ne[0] == img_k->ne[0]);
        GGML_ASSERT(dst->ne[1] == txt_q->ne[1] + img_q->ne[1]);
        GGML_ASSERT(dst->ne[1] == txt_k->ne[1] + img_k->ne[1]);
        GGML_ASSERT(dst->ne[2] == txt_q->ne[2] && dst->ne[2] == img_q->ne[2]);
        GGML_ASSERT(dst->ne[2] == txt_k->ne[2] && dst->ne[2] == img_k->ne[2]);
        GGML_ASSERT(dst->ne[3] == 4);
        GGML_ASSERT(txt_q->ne[3] == 2 && img_q->ne[3] == 2 && txt_k->ne[3] == 2 && img_k->ne[3] == 2);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        qwen_fused_qk_pack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(txt_q->data),
            static_cast<const char*>(img_q->data),
            static_cast<const char*>(txt_k->data),
            static_cast<const char*>(img_k->data),
            static_cast<float*>(dst->data),
            txt_q->ne[1],
            img_q->ne[1],
            dst->ne[2],
            dst->ne[0] * 2,
            txt_q->nb[0], txt_q->nb[1], txt_q->nb[2], txt_q->nb[3],
            img_q->nb[0], img_q->nb[1], img_q->nb[2], img_q->nb[3],
            txt_k->nb[0], txt_k->nb[1], txt_k->nb[2], txt_k->nb[3],
            img_k->nb[0], img_k->nb[1], img_k->nb[2], img_k->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 2) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* txt_v = dst->src[0];
        const ggml_tensor* img_v = dst->src[1];
        GGML_ASSERT(txt_v && img_v);
        GGML_ASSERT(txt_v->type == GGML_TYPE_F32 && img_v->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->ne[0] == txt_v->ne[0] && dst->ne[0] == img_v->ne[0]);
        GGML_ASSERT(dst->ne[1] == txt_v->ne[1] + img_v->ne[1]);
        GGML_ASSERT(dst->ne[2] == txt_v->ne[2] && dst->ne[2] == img_v->ne[2]);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(txt_v->ne[3] == 1 && img_v->ne[3] == 1);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        qwen_fused_v_pack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(txt_v->data),
            static_cast<const char*>(img_v->data),
            static_cast<float*>(dst->data),
            txt_v->ne[1],
            img_v->ne[1],
            dst->ne[2],
            dst->ne[0],
            txt_v->nb[0], txt_v->nb[1], txt_v->nb[2], txt_v->nb[3],
            img_v->nb[0], img_v->nb[1], img_v->nb[2], img_v->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const ggml_tensor* txt_q = dst->src[0];
    const ggml_tensor* img_q = dst->src[1];
    const ggml_tensor* txt_k = dst->src[2];
    const ggml_tensor* img_k = dst->src[3];
    const ggml_tensor* txt_v = dst->src[4];
    const ggml_tensor* img_v = dst->src[5];

    GGML_ASSERT(txt_q && img_q && txt_k && img_k && txt_v && img_v);
    GGML_ASSERT(txt_q->type == GGML_TYPE_F32 && img_q->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_k->type == GGML_TYPE_F32 && img_k->type == GGML_TYPE_F32);
    GGML_ASSERT(txt_v->type == GGML_TYPE_F32 && img_v->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[0] * 2 == txt_v->ne[0] && dst->ne[0] * 2 == img_v->ne[0]);
    GGML_ASSERT(dst->ne[0] == txt_q->ne[0] && dst->ne[0] == img_q->ne[0]);
    GGML_ASSERT(dst->ne[0] == txt_k->ne[0] && dst->ne[0] == img_k->ne[0]);
    GGML_ASSERT(dst->ne[1] == txt_q->ne[1] + img_q->ne[1]);
    GGML_ASSERT(dst->ne[1] == txt_k->ne[1] + img_k->ne[1]);
    GGML_ASSERT(dst->ne[1] == txt_v->ne[1] + img_v->ne[1]);
    GGML_ASSERT(dst->ne[2] == txt_q->ne[2] && dst->ne[2] == img_q->ne[2]);
    GGML_ASSERT(dst->ne[2] == txt_k->ne[2] && dst->ne[2] == img_k->ne[2]);
    GGML_ASSERT(dst->ne[2] == txt_v->ne[2] && dst->ne[2] == img_v->ne[2]);
    GGML_ASSERT(dst->ne[3] == 6);
    GGML_ASSERT(txt_q->ne[3] == 2 && img_q->ne[3] == 2 && txt_k->ne[3] == 2 && img_k->ne[3] == 2);
    GGML_ASSERT(txt_v->ne[3] == 1 && img_v->ne[3] == 1);

    const int64_t total = ggml_nelements(dst);
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    cudaStream_t stream = ctx.stream();

    qwen_fused_qkv_pack_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const char*>(txt_q->data),
        static_cast<const char*>(img_q->data),
        static_cast<const char*>(txt_k->data),
        static_cast<const char*>(img_k->data),
        static_cast<const char*>(txt_v->data),
        static_cast<const char*>(img_v->data),
        static_cast<float*>(dst->data),
        txt_q->ne[1],
        img_q->ne[1],
        dst->ne[2],
        dst->ne[0] * 2,
        txt_q->nb[0], txt_q->nb[1], txt_q->nb[2], txt_q->nb[3],
        img_q->nb[0], img_q->nb[1], img_q->nb[2], img_q->nb[3],
        txt_k->nb[0], txt_k->nb[1], txt_k->nb[2], txt_k->nb[3],
        img_k->nb[0], img_k->nb[1], img_k->nb[2], img_k->nb[3],
        txt_v->nb[0], txt_v->nb[1], txt_v->nb[2], txt_v->nb[3],
        img_v->nb[0], img_v->nb[1], img_v->nb[2], img_v->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}
