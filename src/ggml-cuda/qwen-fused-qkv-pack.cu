#include "qwen-fused-qkv-pack.cuh"

#include "../ggml-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static constexpr uint64_t QWEN_FUSED_QKV_PACK_MAGIC = 0x5157454e46514b56ULL;
static constexpr uint32_t ED_CHANNEL_RMS_NORM_CUSTOM_MAGIC = 0x4543524eU;
static constexpr uint32_t ED_RMS_NORM_MUL_F16_CUSTOM_MAGIC = 0x45524d48U;
static constexpr uint32_t ED_FUSED_MODULATE_CUSTOM_MAGIC = 0x45464d4fU;
static constexpr uint32_t ED_FUSED_RESIDUAL_GATE_CUSTOM_MAGIC = 0x45465247U;
static constexpr uint32_t ED_ROPE_CUSTOM_MAGIC = 0x45525250U;
static constexpr uint32_t ED_ATTENTION_V_PREP_CUSTOM_MAGIC = 0x45565050U;
static constexpr uint32_t ED_ATTENTION_PAIR_PACK_CUSTOM_MAGIC = 0x45505150U;
static constexpr uint32_t ED_ATTENTION_QKV_PAIR_PACK_CUSTOM_MAGIC = 0x45514b50U;
static constexpr uint32_t ED_SP_RECV_PLACEHOLDER_CUSTOM_MAGIC = 0x45535052U;
static constexpr uint32_t ED_FLUX_SP_QKV_RECV_PREP_CUSTOM_MAGIC = 0x45465152U;
static constexpr uint32_t ED_FLUX_SP_QKV_PAIR_RECV_PREP_CUSTOM_MAGIC = 0x45465150U;
static constexpr uint32_t ED_FLUX_SP_QKV_MIXED_RECV_PREP_CUSTOM_MAGIC = 0x45464d52U;
static constexpr uint32_t ED_FLUX_SP_QKV_PAIR_MIXED_RECV_PREP_CUSTOM_MAGIC = 0x45464d50U;
static constexpr uint32_t ED_FLUX_SP_CONCAT_LINEAR_CUSTOM_MAGIC = 0x45464c32U;

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

static bool ggml_cuda_qwen_fused_qkv_pack_name_matches(const ggml_tensor* dst) {
    if (dst == nullptr || dst->name[0] == '\0') {
        return false;
    }
    const char* name = dst->name;
    return std::strstr(name, "qwen.") != nullptr ||
           std::strstr(name, "mmdit.") != nullptr ||
           std::strstr(name, "wan.") != nullptr ||
           std::strstr(name, "_fused_qkv") != nullptr ||
           std::strstr(name, "_qkv_fused") != nullptr ||
           std::strstr(name, "_fused_attn") != nullptr ||
           std::strstr(name, "_attn_head_to_seq") != nullptr ||
           std::strstr(name, "_qkv_seq_to_head") != nullptr ||
           std::strstr(name, "_kv_attn_in") != nullptr ||
           std::strstr(name, "_q_rope") != nullptr ||
           std::strstr(name, "_k_rope") != nullptr;
}

static bool ggml_cuda_qwen_fused_profile_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("ED_PROFILE_FLUX_SP_CUSTOM");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv("ED_PROFILE_FLUX_SP_CUSTOM_OPS");
        }
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

static bool ggml_cuda_flux_sp_fast_qkv_send_pack_f16_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("ED_FLUX_SP_FAST_QKV_SEND_PACK_F16");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        return std::strcmp(env, "0") != 0 &&
               std::strcmp(env, "false") != 0 &&
               std::strcmp(env, "FALSE") != 0 &&
               std::strcmp(env, "off") != 0 &&
               std::strcmp(env, "OFF") != 0;
    }();
    return enabled;
}

static int ggml_cuda_qwen_fused_profile_rank_fallback() {
    const char* names[] = {
        "LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "PMI_LOCAL_RANK",
        "RANK",
    };
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return std::atoi(value);
        }
    }
    return -1;
}

static const char* ggml_cuda_qwen_fused_profile_category(const ggml_tensor* dst) {
    const char* name = (dst != nullptr && dst->name[0] != '\0') ? dst->name : "";
    if (std::strstr(name, "_qkv_seq_to_head") != nullptr) {
        return "qkv_seq_to_head";
    }
    if (std::strstr(name, "_txt_img_attn_head_to_seq") != nullptr) {
        return "double_head_to_seq";
    }
    if (std::strstr(name, "_attn_head_to_seq") != nullptr) {
        return "single_head_to_seq";
    }
    return "other";
}

static const char* ggml_cuda_qwen_fused_profile_mode_detail(int64_t mode) {
    switch (mode) {
        case 6:
            return "qkv_send_pack";
        case 7:
            return "head_to_seq_send_pack";
        case 8:
            return "head_to_seq_recv_unpack";
        case 9:
            return "two_stream_flat_send_pack";
        case 10:
            return "two_stream_flat_recv_unpack";
        case 13:
            return "head_to_seq_send_pack_f16";
        case 14:
            return "head_to_seq_recv_unpack_f16";
        case 45:
            return "head_to_seq_recv_unpack_keep_f16";
        case 48:
            return "head_to_seq_recv_unpack_keep_bf16";
        case 18:
            return "qkv_send_pack_mixed";
        case 46:
            return "qkv_send_pack_f16";
        case 47:
            return "double_qkv_send_pack_f16";
        case 33:
            return "wan_qkv_vhalf_send_pack";
        case 37:
            return "wan_qkv_roped_half_send_pack";
        default:
            return "other";
    }
}

static bool ggml_cuda_qwen_fused_profile_begin(cudaStream_t stream, cudaEvent_t* start, cudaEvent_t* stop) {
    if (!ggml_cuda_qwen_fused_profile_enabled()) {
        return false;
    }
    *start = nullptr;
    *stop = nullptr;
    if (cudaEventCreate(start) != cudaSuccess) {
        return false;
    }
    if (cudaEventCreate(stop) != cudaSuccess) {
        cudaEventDestroy(*start);
        *start = nullptr;
        return false;
    }
    if (cudaEventRecord(*start, stream) != cudaSuccess) {
        cudaEventDestroy(*start);
        cudaEventDestroy(*stop);
        *start = nullptr;
        *stop = nullptr;
        return false;
    }
    return true;
}

static __device__ __forceinline__ float qwen_load_f32_or_f16(const char* ptr, int type) {
    return type == GGML_TYPE_F16 ?
               __half2float(*reinterpret_cast<const half*>(ptr)) :
               *reinterpret_cast<const float*>(ptr);
}

static __device__ __forceinline__ half2 qwen_load_half2_or_convert(const char* ptr, size_t nb0, int type) {
    if (type == GGML_TYPE_F16 && nb0 == sizeof(half)) {
        return *reinterpret_cast<const half2*>(ptr);
    }
    const float v0 = qwen_load_f32_or_f16(ptr, type);
    const float v1 = qwen_load_f32_or_f16(ptr + nb0, type);
    return __floats2half2_rn(v0, v1);
}

static void ggml_cuda_qwen_fused_profile_end(cudaStream_t stream,
                                             cudaEvent_t start,
                                             cudaEvent_t stop,
                                             int64_t mode,
                                             const ggml_tensor* dst) {
    if (start == nullptr || stop == nullptr) {
        return;
    }
    float elapsed_ms = 0.0f;
    if (cudaEventRecord(stop, stream) == cudaSuccess &&
        cudaEventSynchronize(stop) == cudaSuccess &&
        cudaEventElapsedTime(&elapsed_ms, start, stop) == cudaSuccess) {
        const int64_t elems = ggml_nelements(dst);
        const int64_t elem_size = ggml_type_size(dst->type) / ggml_blck_size(dst->type);
        const char* name = (dst != nullptr && dst->name[0] != '\0') ? dst->name : "-";
        std::fprintf(stderr,
                     "ED_FLUX_SP_CUSTOM_PROFILE rank=%d kind=pack_unpack detail=%s category=%s name=%s elems=%lld bytes_mib=%.3f elapsed_ms=%.3f\n",
                     ggml_cuda_qwen_fused_profile_rank_fallback(),
                     ggml_cuda_qwen_fused_profile_mode_detail(mode),
                     ggml_cuda_qwen_fused_profile_category(dst),
                     name,
                     static_cast<long long>(elems),
                     static_cast<double>(elems * elem_size) / (1024.0 * 1024.0),
                     static_cast<double>(elapsed_ms));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

static bool ggml_cuda_qwen_fused_qkv_pack_userdata_is_packed(uintptr_t userdata) {
    const uint32_t low = static_cast<uint32_t>(userdata & 0xffffffffU);
    return low == ED_CHANNEL_RMS_NORM_CUSTOM_MAGIC ||
           low == ED_RMS_NORM_MUL_F16_CUSTOM_MAGIC ||
           low == ED_FUSED_MODULATE_CUSTOM_MAGIC ||
           low == ED_FUSED_RESIDUAL_GATE_CUSTOM_MAGIC ||
           low == ED_ROPE_CUSTOM_MAGIC ||
           low == ED_ATTENTION_V_PREP_CUSTOM_MAGIC ||
           low == ED_ATTENTION_PAIR_PACK_CUSTOM_MAGIC ||
           low == ED_ATTENTION_QKV_PAIR_PACK_CUSTOM_MAGIC ||
           low == ED_SP_RECV_PLACEHOLDER_CUSTOM_MAGIC ||
           low == ED_FLUX_SP_QKV_RECV_PREP_CUSTOM_MAGIC ||
           low == ED_FLUX_SP_QKV_PAIR_RECV_PREP_CUSTOM_MAGIC ||
           low == ED_FLUX_SP_QKV_MIXED_RECV_PREP_CUSTOM_MAGIC ||
           low == ED_FLUX_SP_QKV_PAIR_MIXED_RECV_PREP_CUSTOM_MAGIC ||
           low == ED_FLUX_SP_CONCAT_LINEAR_CUSTOM_MAGIC;
}

bool ggml_cuda_is_qwen_fused_qkv_pack(const ggml_tensor* dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM ||
        (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_BF16)) {
        return false;
    }
    if (!ggml_cuda_qwen_fused_qkv_pack_name_matches(dst)) {
        return false;
    }
    ggml_custom_op_params params;
    memcpy(&params, dst->op_params, sizeof(params));
    if (params.userdata == nullptr) {
        return false;
    }
    const uintptr_t userdata = reinterpret_cast<uintptr_t>(params.userdata);
    if (ggml_cuda_qwen_fused_qkv_pack_userdata_is_packed(userdata) ||
        (userdata % alignof(qwen_fused_qkv_pack_cuda_params)) != 0) {
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

static __global__ void qwen_fused_qkv_send_pack_f16_kernel(const char* __restrict__ q,
                                                           const char* __restrict__ k,
                                                           const char* __restrict__ v,
                                                           half* __restrict__ dst,
                                                           int64_t world_size,
                                                           int64_t head_dim,
                                                           int64_t heads,
                                                           int64_t shard_sequence,
                                                           int q_type,
                                                           int k_type,
                                                           int v_type,
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
    const int64_t total_dim2     = total_head_dim / 2;
    const int64_t total          = total_dim2 * shard_heads * shard_sequence * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem         = linear;
        const int64_t d_all = (rem % total_dim2) * 2;
        rem /= total_dim2;
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

        int src_type = q_type;
        if (qkv_plane == 1) {
            src_type = k_type;
        } else if (qkv_plane == 2) {
            src_type = v_type;
        }
        const char* src_pair = src + d * nb0 + head * nb1 + seq * nb2 + 0 * nb3;
        reinterpret_cast<half2*>(dst)[linear] = qwen_load_half2_or_convert(src_pair, nb0, src_type);
    }
}

static __global__ void qwen_fused_qkv_send_pack_f16_i32_kernel(const char* __restrict__ q,
                                                               const char* __restrict__ k,
                                                               const char* __restrict__ v,
                                                               half2* __restrict__ dst,
                                                               int world_size,
                                                               int head_dim,
                                                               int heads,
                                                               int shard_sequence,
                                                               int q_type,
                                                               int k_type,
                                                               int v_type,
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
    const int shard_heads    = heads / world_size;
    const int total_head_dim = head_dim * 3;
    const int total_dim2     = total_head_dim / 2;
    const int total          = total_dim2 * shard_heads * shard_sequence * world_size;

    for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += gridDim.x * blockDim.x) {
        int rem         = linear;
        const int d_all = (rem % total_dim2) * 2;
        rem /= total_dim2;
        const int head_local = rem % shard_heads;
        rem /= shard_heads;
        const int seq = rem % shard_sequence;
        rem /= shard_sequence;
        const int peer = rem;
        const int head = head_local + peer * shard_heads;
        const int qkv_plane = d_all / head_dim;
        const int d = d_all - qkv_plane * head_dim;

        const char* src = q;
        size_t nb0 = q_nb0;
        size_t nb1 = q_nb1;
        size_t nb2 = q_nb2;
        size_t nb3 = q_nb3;
        int src_type = q_type;
        if (qkv_plane == 1) {
            src = k;
            nb0 = k_nb0;
            nb1 = k_nb1;
            nb2 = k_nb2;
            nb3 = k_nb3;
            src_type = k_type;
        } else if (qkv_plane == 2) {
            src = v;
            nb0 = v_nb0;
            nb1 = v_nb1;
            nb2 = v_nb2;
            nb3 = v_nb3;
            src_type = v_type;
        }

        const char* src_pair = src + d * nb0 + head * nb1 + seq * nb2 + 0 * nb3;
        dst[linear] = qwen_load_half2_or_convert(src_pair, nb0, src_type);
    }
}

static __global__ void flux_sp_qkv_send_pack_f16_triplet_kernel(const char* __restrict__ q,
                                                                const char* __restrict__ k,
                                                                const char* __restrict__ v,
                                                                half2* __restrict__ dst,
                                                                int world_size,
                                                                int head_dim,
                                                                int heads,
                                                                int shard_sequence,
                                                                int q_type,
                                                                int k_type,
                                                                int v_type,
                                                                size_t q_nb0,
                                                                size_t q_nb1,
                                                                size_t q_nb2,
                                                                size_t k_nb0,
                                                                size_t k_nb1,
                                                                size_t k_nb2,
                                                                size_t v_nb0,
                                                                size_t v_nb1,
                                                                size_t v_nb2) {
    const int shard_heads = heads / world_size;
    const int half_dim = head_dim / 2;
    const int total = half_dim * shard_heads * shard_sequence * world_size;
    const int plane_stride_h2 = half_dim;
    const int head_stride_h2 = plane_stride_h2 * 3;

    for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += gridDim.x * blockDim.x) {
        int rem = linear;
        const int pair = rem % half_dim;
        rem /= half_dim;
        const int head_local = rem % shard_heads;
        rem /= shard_heads;
        const int seq = rem % shard_sequence;
        rem /= shard_sequence;
        const int peer = rem;
        const int head = head_local + peer * shard_heads;
        const int d = pair * 2;

        const int64_t base_h2 =
            static_cast<int64_t>(pair) +
            static_cast<int64_t>(head_local) * head_stride_h2 +
            static_cast<int64_t>(seq) * head_stride_h2 * shard_heads +
            static_cast<int64_t>(peer) * head_stride_h2 * shard_heads * shard_sequence;

        const char* q_pair = q + static_cast<int64_t>(d) * q_nb0 + static_cast<int64_t>(head) * q_nb1 + static_cast<int64_t>(seq) * q_nb2;
        const char* k_pair = k + static_cast<int64_t>(d) * k_nb0 + static_cast<int64_t>(head) * k_nb1 + static_cast<int64_t>(seq) * k_nb2;
        const char* v_pair = v + static_cast<int64_t>(d) * v_nb0 + static_cast<int64_t>(head) * v_nb1 + static_cast<int64_t>(seq) * v_nb2;

        dst[base_h2] = qwen_load_half2_or_convert(q_pair, q_nb0, q_type);
        dst[base_h2 + plane_stride_h2] = qwen_load_half2_or_convert(k_pair, k_nb0, k_type);
        dst[base_h2 + 2 * plane_stride_h2] = qwen_load_half2_or_convert(v_pair, v_nb0, v_type);
    }
}

static __global__ void flux_sp_double_qkv_send_pack_f16_kernel(
    const char* __restrict__ first_q,
    const char* __restrict__ first_k,
    const char* __restrict__ first_v,
    const char* __restrict__ second_q,
    const char* __restrict__ second_k,
    const char* __restrict__ second_v,
    half2* __restrict__ dst,
    int world_size,
    int head_dim,
    int heads,
    int first_shard_sequence,
    int second_shard_sequence,
    int first_q_type,
    int first_k_type,
    int first_v_type,
    int second_q_type,
    int second_k_type,
    int second_v_type,
    size_t first_q_nb0,
    size_t first_q_nb1,
    size_t first_q_nb2,
    size_t first_k_nb0,
    size_t first_k_nb1,
    size_t first_k_nb2,
    size_t first_v_nb0,
    size_t first_v_nb1,
    size_t first_v_nb2,
    size_t second_q_nb0,
    size_t second_q_nb1,
    size_t second_q_nb2,
    size_t second_k_nb0,
    size_t second_k_nb1,
    size_t second_k_nb2,
    size_t second_v_nb0,
    size_t second_v_nb1,
    size_t second_v_nb2) {
    const int shard_heads = heads / world_size;
    const int total_head_dim = head_dim * 3;
    const int total_dim2 = total_head_dim / 2;
    const int first_chunk_h2 = total_dim2 * shard_heads * first_shard_sequence;
    const int second_chunk_h2 = total_dim2 * shard_heads * second_shard_sequence;
    const int count_per_peer_h2 = first_chunk_h2 + second_chunk_h2;
    const int64_t total = static_cast<int64_t>(count_per_peer_h2) * world_size;

    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int peer = rem / count_per_peer_h2;
        rem -= static_cast<int64_t>(peer) * count_per_peer_h2;
        const bool is_second = rem >= first_chunk_h2;
        if (is_second) {
            rem -= first_chunk_h2;
        }
        const int d_all = static_cast<int>(rem % total_dim2) * 2;
        rem /= total_dim2;
        const int head_local = rem % shard_heads;
        rem /= shard_heads;
        const int seq = rem;
        const int head = head_local + peer * shard_heads;
        const int qkv_plane = d_all / head_dim;
        const int d = d_all - qkv_plane * head_dim;

        const char* src = first_q;
        int src_type = first_q_type;
        size_t nb0 = first_q_nb0;
        size_t nb1 = first_q_nb1;
        size_t nb2 = first_q_nb2;
        if (!is_second) {
            if (qkv_plane == 1) {
                src = first_k;
                src_type = first_k_type;
                nb0 = first_k_nb0;
                nb1 = first_k_nb1;
                nb2 = first_k_nb2;
            } else if (qkv_plane == 2) {
                src = first_v;
                src_type = first_v_type;
                nb0 = first_v_nb0;
                nb1 = first_v_nb1;
                nb2 = first_v_nb2;
            }
        } else {
            src = second_q;
            src_type = second_q_type;
            nb0 = second_q_nb0;
            nb1 = second_q_nb1;
            nb2 = second_q_nb2;
            if (qkv_plane == 1) {
                src = second_k;
                src_type = second_k_type;
                nb0 = second_k_nb0;
                nb1 = second_k_nb1;
                nb2 = second_k_nb2;
            } else if (qkv_plane == 2) {
                src = second_v;
                src_type = second_v_type;
                nb0 = second_v_nb0;
                nb1 = second_v_nb1;
                nb2 = second_v_nb2;
            }
        }
        const char* src_pair =
            src +
            static_cast<int64_t>(d) * nb0 +
            static_cast<int64_t>(head) * nb1 +
            static_cast<int64_t>(seq) * nb2;
        dst[linear] = qwen_load_half2_or_convert(src_pair, nb0, src_type);
    }
}

static __global__ void flux_sp_double_qkv_send_pack_f16_triplet_kernel(
    const char* __restrict__ first_q,
    const char* __restrict__ first_k,
    const char* __restrict__ first_v,
    const char* __restrict__ second_q,
    const char* __restrict__ second_k,
    const char* __restrict__ second_v,
    half2* __restrict__ dst,
    int world_size,
    int head_dim,
    int heads,
    int first_shard_sequence,
    int second_shard_sequence,
    int first_q_type,
    int first_k_type,
    int first_v_type,
    int second_q_type,
    int second_k_type,
    int second_v_type,
    size_t first_q_nb0,
    size_t first_q_nb1,
    size_t first_q_nb2,
    size_t first_k_nb0,
    size_t first_k_nb1,
    size_t first_k_nb2,
    size_t first_v_nb0,
    size_t first_v_nb1,
    size_t first_v_nb2,
    size_t second_q_nb0,
    size_t second_q_nb1,
    size_t second_q_nb2,
    size_t second_k_nb0,
    size_t second_k_nb1,
    size_t second_k_nb2,
    size_t second_v_nb0,
    size_t second_v_nb1,
    size_t second_v_nb2) {
    const int shard_heads = heads / world_size;
    const int half_dim = head_dim / 2;
    const int total_dim2 = half_dim * 3;
    const int head_stride_h2 = total_dim2;
    const int first_chunk_h2 = head_stride_h2 * shard_heads * first_shard_sequence;
    const int second_chunk_h2 = head_stride_h2 * shard_heads * second_shard_sequence;
    const int count_per_peer_h2 = first_chunk_h2 + second_chunk_h2;
    const int total_sequence = first_shard_sequence + second_shard_sequence;
    const int64_t total = static_cast<int64_t>(half_dim) * shard_heads * total_sequence * world_size;

    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int pair = static_cast<int>(rem % half_dim);
        rem /= half_dim;
        const int head_local = static_cast<int>(rem % shard_heads);
        rem /= shard_heads;
        const int combined_seq = static_cast<int>(rem % total_sequence);
        rem /= total_sequence;
        const int peer = static_cast<int>(rem);

        const bool is_second = combined_seq >= first_shard_sequence;
        const int seq = is_second ? combined_seq - first_shard_sequence : combined_seq;
        const int head = head_local + peer * shard_heads;
        const int d = pair * 2;

        const int64_t stream_base =
            static_cast<int64_t>(peer) * count_per_peer_h2 +
            (is_second ? first_chunk_h2 : 0);
        const int64_t base_h2 =
            stream_base +
            static_cast<int64_t>(pair) +
            static_cast<int64_t>(head_local) * head_stride_h2 +
            static_cast<int64_t>(seq) * head_stride_h2 * shard_heads;

        if (!is_second) {
            const char* q_pair = first_q + static_cast<int64_t>(d) * first_q_nb0 + static_cast<int64_t>(head) * first_q_nb1 + static_cast<int64_t>(seq) * first_q_nb2;
            const char* k_pair = first_k + static_cast<int64_t>(d) * first_k_nb0 + static_cast<int64_t>(head) * first_k_nb1 + static_cast<int64_t>(seq) * first_k_nb2;
            const char* v_pair = first_v + static_cast<int64_t>(d) * first_v_nb0 + static_cast<int64_t>(head) * first_v_nb1 + static_cast<int64_t>(seq) * first_v_nb2;

            dst[base_h2] = qwen_load_half2_or_convert(q_pair, first_q_nb0, first_q_type);
            dst[base_h2 + half_dim] = qwen_load_half2_or_convert(k_pair, first_k_nb0, first_k_type);
            dst[base_h2 + 2 * half_dim] = qwen_load_half2_or_convert(v_pair, first_v_nb0, first_v_type);
        } else {
            const char* q_pair = second_q + static_cast<int64_t>(d) * second_q_nb0 + static_cast<int64_t>(head) * second_q_nb1 + static_cast<int64_t>(seq) * second_q_nb2;
            const char* k_pair = second_k + static_cast<int64_t>(d) * second_k_nb0 + static_cast<int64_t>(head) * second_k_nb1 + static_cast<int64_t>(seq) * second_k_nb2;
            const char* v_pair = second_v + static_cast<int64_t>(d) * second_v_nb0 + static_cast<int64_t>(head) * second_v_nb1 + static_cast<int64_t>(seq) * second_v_nb2;

            dst[base_h2] = qwen_load_half2_or_convert(q_pair, second_q_nb0, second_q_type);
            dst[base_h2 + half_dim] = qwen_load_half2_or_convert(k_pair, second_k_nb0, second_k_type);
            dst[base_h2 + 2 * half_dim] = qwen_load_half2_or_convert(v_pair, second_v_nb0, second_v_type);
        }
    }
}

static __global__ void wan_fused_qk_recv_unpack_f32_kernel(const char* __restrict__ recv_flat,
                                                           float* __restrict__ dst,
                                                           int64_t world_size,
                                                           int64_t plane,
                                                           int64_t head_dim,
                                                           int64_t shard_heads,
                                                           int64_t shard_sequence) {
    const int64_t half_dim       = head_dim / 2;
    const int64_t sequence       = shard_sequence * world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t total          = half_dim * sequence * shard_heads * 2;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem          = linear;
        const int64_t half   = rem % half_dim;
        rem /= half_dim;
        const int64_t seq    = rem % sequence;
        rem /= sequence;
        const int64_t head   = rem % shard_heads;
        rem /= shard_heads;
        const int64_t part   = rem;
        const int64_t peer   = seq / shard_sequence;
        const int64_t local_seq = seq - peer * shard_sequence;
        const int64_t src_d  = plane * head_dim + part + 2 * half;
        const int64_t src_idx = src_d +
                                head * total_head_dim +
                                local_seq * total_head_dim * shard_heads +
                                peer * total_head_dim * shard_heads * shard_sequence;
        dst[linear] = *reinterpret_cast<const float*>(recv_flat + src_idx * sizeof(float));
    }
}

static __global__ void wan_fused_qkv_vhalf_send_pack_kernel(const char* __restrict__ q,
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
    const int64_t packed_dim  = head_dim * 2 + head_dim / 2;
    const int64_t total       = packed_dim * shard_heads * shard_sequence * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem         = linear;
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
        } else if (d_all < head_dim * 2) {
            const int64_t d = d_all - head_dim;
            const float value = *reinterpret_cast<const float*>(k + d * k_nb0 + head * k_nb1 + seq * k_nb2 + 0 * k_nb3);
            dst[linear] = __float_as_uint(value);
        } else {
            const int64_t half = d_all - head_dim * 2;
            const int64_t d0 = 2 * half;
            const int64_t d1 = d0 + 1;
            const float v0 = *reinterpret_cast<const float*>(v + d0 * v_nb0 + head * v_nb1 + seq * v_nb2 + 0 * v_nb3);
            const float v1 = *reinterpret_cast<const float*>(v + d1 * v_nb0 + head * v_nb1 + seq * v_nb2 + 0 * v_nb3);
            const uint32_t h0 = static_cast<uint32_t>(__half_as_ushort(__float2half(v0)));
            const uint32_t h1 = static_cast<uint32_t>(__half_as_ushort(__float2half(v1)));
            dst[linear] = h0 | (h1 << 16);
        }
    }
}

static __device__ __forceinline__ float wan_roped_local_value(const char* __restrict__ x,
                                                              const char* __restrict__ pe,
                                                              int64_t d,
                                                              int64_t head,
                                                              int64_t local_seq,
                                                              int64_t global_seq,
                                                              size_t x_nb0,
                                                              size_t x_nb1,
                                                              size_t x_nb2,
                                                              size_t x_nb3,
                                                              size_t pe_nb0,
                                                              size_t pe_nb1,
                                                              size_t pe_nb2,
                                                              size_t pe_nb3) {
    const int64_t part = d & 1;
    const int64_t half = d >> 1;
    const float x0 = *reinterpret_cast<const float*>(x + (2 * half) * x_nb0 + head * x_nb1 + local_seq * x_nb2 + 0 * x_nb3);
    const float x1 = *reinterpret_cast<const float*>(x + (2 * half + 1) * x_nb0 + head * x_nb1 + local_seq * x_nb2 + 0 * x_nb3);
    const float pe0 = *reinterpret_cast<const float*>(pe + part * pe_nb0 + half * pe_nb1 + global_seq * pe_nb2 + 0 * pe_nb3);
    const float pe1 = *reinterpret_cast<const float*>(pe + part * pe_nb0 + half * pe_nb1 + global_seq * pe_nb2 + 1 * pe_nb3);
    return __fadd_rn(__fmul_rn(x0, pe0), __fmul_rn(x1, pe1));
}

static __global__ void wan_fused_qkv_roped_half_send_pack_kernel(const char* __restrict__ q,
                                                                 const char* __restrict__ k,
                                                                 const char* __restrict__ v,
                                                                 const char* __restrict__ pe,
                                                                 uint32_t* __restrict__ dst,
                                                                 int64_t world_size,
                                                                 int64_t rank,
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
                                                                 size_t v_nb3,
                                                                 size_t pe_nb0,
                                                                 size_t pe_nb1,
                                                                 size_t pe_nb2,
                                                                 size_t pe_nb3) {
    const int64_t half_dim    = head_dim / 2;
    const int64_t shard_heads = heads / world_size;
    const int64_t packed_dim  = head_dim * 2;
    const int64_t total       = packed_dim * shard_heads * shard_sequence * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem         = linear;
        const int64_t d_all = rem % packed_dim;
        rem /= packed_dim;
        const int64_t head_local = rem % shard_heads;
        rem /= shard_heads;
        const int64_t seq = rem % shard_sequence;
        rem /= shard_sequence;
        const int64_t peer = rem;
        const int64_t head = head_local + peer * shard_heads;
        const int64_t global_seq = rank * shard_sequence + seq;

        if (d_all < head_dim) {
            const float value = wan_roped_local_value(q,
                                                      pe,
                                                      d_all,
                                                      head,
                                                      seq,
                                                      global_seq,
                                                      q_nb0, q_nb1, q_nb2, q_nb3,
                                                      pe_nb0, pe_nb1, pe_nb2, pe_nb3);
            dst[linear] = __float_as_uint(value);
        } else if (d_all < head_dim + half_dim) {
            const int64_t half = d_all - head_dim;
            const int64_t d0 = 2 * half;
            const int64_t d1 = d0 + 1;
            const float k0 = wan_roped_local_value(k,
                                                   pe,
                                                   d0,
                                                   head,
                                                   seq,
                                                   global_seq,
                                                   k_nb0, k_nb1, k_nb2, k_nb3,
                                                   pe_nb0, pe_nb1, pe_nb2, pe_nb3);
            const float k1 = wan_roped_local_value(k,
                                                   pe,
                                                   d1,
                                                   head,
                                                   seq,
                                                   global_seq,
                                                   k_nb0, k_nb1, k_nb2, k_nb3,
                                                   pe_nb0, pe_nb1, pe_nb2, pe_nb3);
            const uint32_t h0 = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(k0)));
            const uint32_t h1 = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(k1)));
            dst[linear] = h0 | (h1 << 16);
        } else {
            const int64_t half = d_all - head_dim - half_dim;
            const int64_t d0 = 2 * half;
            const int64_t d1 = d0 + 1;
            const float v0 = *reinterpret_cast<const float*>(v + d0 * v_nb0 + head * v_nb1 + seq * v_nb2 + 0 * v_nb3);
            const float v1 = *reinterpret_cast<const float*>(v + d1 * v_nb0 + head * v_nb1 + seq * v_nb2 + 0 * v_nb3);
            const uint32_t h0 = static_cast<uint32_t>(__half_as_ushort(__float2half(v0)));
            const uint32_t h1 = static_cast<uint32_t>(__half_as_ushort(__float2half(v1)));
            dst[linear] = h0 | (h1 << 16);
        }
    }
}

static __global__ void wan_fused_v_recv_unpack_f32_kernel(const char* __restrict__ recv_flat,
                                                          float* __restrict__ dst,
                                                          int64_t world_size,
                                                          int64_t head_dim,
                                                          int64_t shard_heads,
                                                          int64_t shard_sequence) {
    const int64_t sequence       = shard_sequence * world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t total          = head_dim * sequence * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem         = linear;
        const int64_t d     = rem % head_dim;
        rem /= head_dim;
        const int64_t seq   = rem % sequence;
        rem /= sequence;
        const int64_t head  = rem % shard_heads;
        const int64_t peer  = seq / shard_sequence;
        const int64_t local_seq = seq - peer * shard_sequence;
        const int64_t src_d = 2 * head_dim + d;
        const int64_t src_idx = src_d +
                                head * total_head_dim +
                                local_seq * total_head_dim * shard_heads +
                                peer * total_head_dim * shard_heads * shard_sequence;
        dst[linear] = *reinterpret_cast<const float*>(recv_flat + src_idx * sizeof(float));
    }
}

static __global__ void wan_fused_v_recv_unpack_f16_kernel(const char* __restrict__ recv_flat,
                                                          half* __restrict__ dst,
                                                          int64_t world_size,
                                                          int64_t head_dim,
                                                          int64_t shard_heads,
                                                          int64_t shard_sequence) {
    const int64_t sequence       = shard_sequence * world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t total          = head_dim * sequence * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem         = linear;
        const int64_t d     = rem % head_dim;
        rem /= head_dim;
        const int64_t seq   = rem % sequence;
        rem /= sequence;
        const int64_t head  = rem % shard_heads;
        const int64_t peer  = seq / shard_sequence;
        const int64_t local_seq = seq - peer * shard_sequence;
        const int64_t src_d = 2 * head_dim + d;
        const int64_t src_idx = src_d +
                                head * total_head_dim +
                                local_seq * total_head_dim * shard_heads +
                                peer * total_head_dim * shard_heads * shard_sequence;
        dst[linear] = __float2half(*reinterpret_cast<const float*>(recv_flat + src_idx * sizeof(float)));
    }
}

static __global__ void wan_fused_attn_head_to_seq_send_pack_f32_kernel(const char* __restrict__ attn,
                                                                       float* __restrict__ dst,
                                                                       int64_t world_size,
                                                                       int64_t head_dim,
                                                                       int64_t shard_heads,
                                                                       int64_t sequence,
                                                                       size_t attn_nb0,
                                                                       size_t attn_nb1,
                                                                       size_t attn_nb2,
                                                                       size_t attn_nb3) {
    const int64_t shard_sequence = sequence / world_size;
    const int64_t count_per_peer = head_dim * shard_heads * shard_sequence;
    const int64_t total          = count_per_peer * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem        = linear;
        const int64_t peer = rem / count_per_peer;
        rem -= peer * count_per_peer;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % shard_heads;
        rem /= shard_heads;
        const int64_t local_seq = rem;
        const int64_t seq = peer * shard_sequence + local_seq;
        dst[linear] = *reinterpret_cast<const float*>(attn +
                                                      d * attn_nb0 +
                                                      head * attn_nb1 +
                                                      seq * attn_nb2 +
                                                      0 * attn_nb3);
    }
}

static __global__ void wan_fused_attn_head_to_seq_recv_unpack_f32_kernel(const char* __restrict__ recv_flat,
                                                                         float* __restrict__ dst,
                                                                         int64_t world_size,
                                                                         int64_t head_dim,
                                                                         int64_t heads,
                                                                         int64_t shard_sequence) {
    const int64_t shard_heads = heads / world_size;
    const int64_t count_per_peer = head_dim * shard_heads * shard_sequence;
    const int64_t total = head_dim * heads * shard_sequence;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem    = linear;
        const int64_t d = rem % head_dim;
        rem /= head_dim;
        const int64_t head = rem % heads;
        rem /= heads;
        const int64_t local_seq  = rem;
        const int64_t src_peer   = head / shard_heads;
        const int64_t local_head = head - src_peer * shard_heads;
        const int64_t src_idx = src_peer * count_per_peer +
                                d +
                                local_head * head_dim +
                                local_seq * head_dim * shard_heads;
        dst[linear] = *reinterpret_cast<const float*>(recv_flat + src_idx * sizeof(float));
    }
}

static __global__ void wan_rope_seq_major_f16_kernel(const char* __restrict__ x,
                                                     const char* __restrict__ pe,
                                                     half* __restrict__ dst,
                                                     int64_t half_dim,
                                                     int64_t sequence,
                                                     int64_t heads,
                                                     size_t x_nb0,
                                                     size_t x_nb1,
                                                     size_t x_nb2,
                                                     size_t x_nb3,
                                                     size_t pe_nb0,
                                                     size_t pe_nb1,
                                                     size_t pe_nb2,
                                                     size_t pe_nb3) {
    const int64_t head_dim = half_dim * 2;
    const int64_t total    = head_dim * sequence * heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem        = linear;
        const int64_t d    = rem % head_dim;
        rem /= head_dim;
        const int64_t seq  = rem % sequence;
        rem /= sequence;
        const int64_t head = rem % heads;

        const int64_t part = d & 1;
        const int64_t half = d >> 1;
        const float x0     = *reinterpret_cast<const float*>(x + half * x_nb0 + seq * x_nb1 + head * x_nb2 + 0 * x_nb3);
        const float x1     = *reinterpret_cast<const float*>(x + half * x_nb0 + seq * x_nb1 + head * x_nb2 + 1 * x_nb3);
        const float pe0    = *reinterpret_cast<const float*>(pe + part * pe_nb0 + half * pe_nb1 + seq * pe_nb2 + 0 * pe_nb3);
        const float pe1    = *reinterpret_cast<const float*>(pe + part * pe_nb0 + half * pe_nb1 + seq * pe_nb2 + 1 * pe_nb3);
        const float value  = __fadd_rn(__fmul_rn(x0, pe0), __fmul_rn(x1, pe1));
        dst[linear]        = __float2half_rn(value);
    }
}

static __global__ void wan_rope_seq_major_f32_kernel(const char* __restrict__ x,
                                                     const char* __restrict__ pe,
                                                     float* __restrict__ dst,
                                                     int64_t half_dim,
                                                     int64_t sequence,
                                                     int64_t heads,
                                                     size_t x_nb0,
                                                     size_t x_nb1,
                                                     size_t x_nb2,
                                                     size_t x_nb3,
                                                     size_t pe_nb0,
                                                     size_t pe_nb1,
                                                     size_t pe_nb2,
                                                     size_t pe_nb3) {
    const int64_t head_dim = half_dim * 2;
    const int64_t total    = head_dim * sequence * heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem        = linear;
        const int64_t d    = rem % head_dim;
        rem /= head_dim;
        const int64_t seq  = rem % sequence;
        rem /= sequence;
        const int64_t head = rem % heads;

        const int64_t part = d & 1;
        const int64_t half = d >> 1;
        const float x0     = *reinterpret_cast<const float*>(x + half * x_nb0 + seq * x_nb1 + head * x_nb2 + 0 * x_nb3);
        const float x1     = *reinterpret_cast<const float*>(x + half * x_nb0 + seq * x_nb1 + head * x_nb2 + 1 * x_nb3);
        const float pe0    = *reinterpret_cast<const float*>(pe + part * pe_nb0 + half * pe_nb1 + seq * pe_nb2 + 0 * pe_nb3);
        const float pe1    = *reinterpret_cast<const float*>(pe + part * pe_nb0 + half * pe_nb1 + seq * pe_nb2 + 1 * pe_nb3);
        dst[linear]        = __fadd_rn(__fmul_rn(x0, pe0), __fmul_rn(x1, pe1));
    }
}

template <bool output_f16>
static __global__ void wan_fused_qk_recv_rope_kernel(const char* __restrict__ recv_flat,
                                                     const char* __restrict__ pe,
                                                     void* __restrict__ dst,
                                                     int64_t world_size,
                                                     int64_t plane,
                                                     int64_t head_dim,
                                                     int64_t shard_heads,
                                                     int64_t shard_sequence,
                                                     size_t pe_nb0,
                                                     size_t pe_nb1,
                                                     size_t pe_nb2,
                                                     size_t pe_nb3) {
    const int64_t sequence       = shard_sequence * world_size;
    const int64_t total_head_dim = head_dim * 3;
    const int64_t total          = head_dim * sequence * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem        = linear;
        const int64_t d    = rem % head_dim;
        rem /= head_dim;
        const int64_t seq  = rem % sequence;
        rem /= sequence;
        const int64_t head = rem % shard_heads;

        const int64_t part      = d & 1;
        const int64_t half_idx  = d >> 1;
        const int64_t peer      = seq / shard_sequence;
        const int64_t local_seq = seq - peer * shard_sequence;
        const int64_t src_d0    = plane * head_dim + 2 * half_idx;
        const int64_t src_d1    = src_d0 + 1;
        const int64_t base_idx  = head * total_head_dim +
                                  local_seq * total_head_dim * shard_heads +
                                  peer * total_head_dim * shard_heads * shard_sequence;
        const float x0  = *reinterpret_cast<const float*>(recv_flat + (src_d0 + base_idx) * sizeof(float));
        const float x1  = *reinterpret_cast<const float*>(recv_flat + (src_d1 + base_idx) * sizeof(float));
        const float pe0 = *reinterpret_cast<const float*>(pe + part * pe_nb0 + half_idx * pe_nb1 + seq * pe_nb2 + 0 * pe_nb3);
        const float pe1 = *reinterpret_cast<const float*>(pe + part * pe_nb0 + half_idx * pe_nb1 + seq * pe_nb2 + 1 * pe_nb3);
        const float value = __fadd_rn(__fmul_rn(x0, pe0), __fmul_rn(x1, pe1));
        if constexpr (output_f16) {
            static_cast<half*>(dst)[linear] = __float2half_rn(value);
        } else {
            static_cast<float*>(dst)[linear] = value;
        }
    }
}

template <bool output_f16>
static __global__ void wan_fused_qk_recv_rope_vhalf_kernel(const uint32_t* __restrict__ recv_flat,
                                                           const char* __restrict__ pe,
                                                           void* __restrict__ dst,
                                                           int64_t world_size,
                                                           int64_t plane,
                                                           int64_t head_dim,
                                                           int64_t shard_heads,
                                                           int64_t shard_sequence,
                                                           size_t pe_nb0,
                                                           size_t pe_nb1,
                                                           size_t pe_nb2,
                                                           size_t pe_nb3) {
    const int64_t sequence   = shard_sequence * world_size;
    const int64_t half_dim   = head_dim / 2;
    const int64_t packed_dim = head_dim * 2 + half_dim;
    const int64_t total      = head_dim * sequence * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem        = linear;
        const int64_t d    = rem % head_dim;
        rem /= head_dim;
        const int64_t seq  = rem % sequence;
        rem /= sequence;
        const int64_t head = rem % shard_heads;

        const int64_t part      = d & 1;
        const int64_t half_idx  = d >> 1;
        const int64_t peer      = seq / shard_sequence;
        const int64_t local_seq = seq - peer * shard_sequence;
        const int64_t src_d0    = plane * head_dim + 2 * half_idx;
        const int64_t src_d1    = src_d0 + 1;
        const int64_t base_idx  = head * packed_dim +
                                  local_seq * packed_dim * shard_heads +
                                  peer * packed_dim * shard_heads * shard_sequence;
        const float x0  = __uint_as_float(recv_flat[src_d0 + base_idx]);
        const float x1  = __uint_as_float(recv_flat[src_d1 + base_idx]);
        const float pe0 = *reinterpret_cast<const float*>(pe + part * pe_nb0 + half_idx * pe_nb1 + seq * pe_nb2 + 0 * pe_nb3);
        const float pe1 = *reinterpret_cast<const float*>(pe + part * pe_nb0 + half_idx * pe_nb1 + seq * pe_nb2 + 1 * pe_nb3);
        const float value = __fadd_rn(__fmul_rn(x0, pe0), __fmul_rn(x1, pe1));
        if constexpr (output_f16) {
            static_cast<half*>(dst)[linear] = __float2half_rn(value);
        } else {
            static_cast<float*>(dst)[linear] = value;
        }
    }
}

static __global__ void wan_fused_vhalf_recv_unpack_kernel(const uint32_t* __restrict__ recv_flat,
                                                          half* __restrict__ dst,
                                                          int64_t world_size,
                                                          int64_t head_dim,
                                                          int64_t shard_heads,
                                                          int64_t shard_sequence) {
    const int64_t sequence   = shard_sequence * world_size;
    const int64_t packed_dim = head_dim * 2 + head_dim / 2;
    const int64_t total      = head_dim * sequence * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem       = linear;
        const int64_t d   = rem % head_dim;
        rem /= head_dim;
        const int64_t seq = rem % sequence;
        rem /= sequence;
        const int64_t head = rem % shard_heads;
        const int64_t peer = seq / shard_sequence;
        const int64_t local_seq = seq - peer * shard_sequence;
        const int64_t src_d = head_dim * 2 + d / 2;
        const int64_t src_idx = src_d +
                                head * packed_dim +
                                local_seq * packed_dim * shard_heads +
                                peer * packed_dim * shard_heads * shard_sequence;
        const uint32_t packed = recv_flat[src_idx];
        const uint16_t bits = (d & 1) == 0 ?
                                  static_cast<uint16_t>(packed & 0xffffu) :
                                  static_cast<uint16_t>(packed >> 16);
        dst[linear] = __ushort_as_half(bits);
    }
}

static __global__ void wan_fused_roped_qkv_recv_unpack_kernel(const uint32_t* __restrict__ recv_flat,
                                                              void* __restrict__ dst,
                                                              int64_t world_size,
                                                              int64_t plane,
                                                              int64_t head_dim,
                                                              int64_t shard_heads,
                                                              int64_t shard_sequence) {
    const int64_t sequence   = shard_sequence * world_size;
    const int64_t half_dim   = head_dim / 2;
    const int64_t packed_dim = head_dim * 2;
    const int64_t total      = head_dim * sequence * shard_heads;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem       = linear;
        const int64_t d   = rem % head_dim;
        rem /= head_dim;
        const int64_t seq = rem % sequence;
        rem /= sequence;
        const int64_t head = rem % shard_heads;
        const int64_t peer = seq / shard_sequence;
        const int64_t local_seq = seq - peer * shard_sequence;
        const int64_t base_idx = head * packed_dim +
                                 local_seq * packed_dim * shard_heads +
                                 peer * packed_dim * shard_heads * shard_sequence;
        if (plane == 0) {
            static_cast<float*>(dst)[linear] = __uint_as_float(recv_flat[base_idx + d]);
        } else {
            const int64_t half_plane = plane == 1 ? 0 : 1;
            const int64_t src_idx = base_idx + head_dim + half_plane * half_dim + d / 2;
            const uint32_t packed = recv_flat[src_idx];
            const uint16_t bits = (d & 1) == 0 ?
                                      static_cast<uint16_t>(packed & 0xffffu) :
                                      static_cast<uint16_t>(packed >> 16);
            static_cast<half*>(dst)[linear] = __ushort_as_half(bits);
        }
    }
}

static __global__ void wan_fused_roped_kv_recv_unpack_kernel(const uint32_t* __restrict__ recv_flat,
                                                             half* __restrict__ dst,
                                                             int64_t world_size,
                                                             int64_t head_dim,
                                                             int64_t shard_heads,
                                                             int64_t shard_sequence) {
    const int64_t sequence   = shard_sequence * world_size;
    const int64_t half_dim   = head_dim / 2;
    const int64_t packed_dim = head_dim * 2;
    const int64_t total      = head_dim * sequence * shard_heads * 2;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem       = linear;
        const int64_t d   = rem % head_dim;
        rem /= head_dim;
        const int64_t seq = rem % sequence;
        rem /= sequence;
        const int64_t head = rem % shard_heads;
        rem /= shard_heads;
        const int64_t plane = rem;
        const int64_t peer = seq / shard_sequence;
        const int64_t local_seq = seq - peer * shard_sequence;
        const int64_t base_idx = head * packed_dim +
                                 local_seq * packed_dim * shard_heads +
                                 peer * packed_dim * shard_heads * shard_sequence;
        const int64_t src_idx = base_idx + head_dim + plane * half_dim + d / 2;
        const uint32_t packed = recv_flat[src_idx];
        const uint16_t bits = (d & 1) == 0 ?
                                  static_cast<uint16_t>(packed & 0xffffu) :
                                  static_cast<uint16_t>(packed >> 16);
        dst[linear] = __ushort_as_half(bits);
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
                                                                        bool attn_is_f16,
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

        half value = __float2half(0.0f);
        if (stream_tok < stream_real_seq) {
            const int64_t total_tok = is_img ? txt_real_seq + stream_tok : stream_tok;
            const char* src = attn +
                              d * attn_nb0 +
                              head * attn_nb1 +
                              total_tok * attn_nb2 +
                              0 * attn_nb3;
            value = attn_is_f16 ?
                        *reinterpret_cast<const half*>(src) :
                        __float2half(*reinterpret_cast<const float*>(src));
        }
        dst[linear] = value;
    }
}

static __global__ void qwen_fused_attn_head_to_seq_send_pack_f16_h2_kernel(const char* __restrict__ attn,
                                                                           half2* __restrict__ dst,
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
    const int64_t head_dim2 = head_dim / 2;
    const int64_t txt_shard_seq = txt_padded_seq / world_size;
    const int64_t img_shard_seq = img_padded_seq / world_size;
    const int64_t txt_chunk2 = head_dim2 * shard_heads * txt_shard_seq;
    const int64_t img_chunk2 = head_dim2 * shard_heads * img_shard_seq;
    const int64_t count_per_peer2 = txt_chunk2 + img_chunk2;
    const int64_t total = count_per_peer2 * world_size;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int64_t peer = rem / count_per_peer2;
        rem -= peer * count_per_peer2;
        const bool is_img = rem >= txt_chunk2;
        if (is_img) {
            rem -= txt_chunk2;
        }
        const int64_t shard_seq = is_img ? img_shard_seq : txt_shard_seq;
        const int64_t stream_real_seq = is_img ? img_real_seq : txt_real_seq;
        const int64_t d2 = rem % head_dim2;
        rem /= head_dim2;
        const int64_t head = rem % shard_heads;
        rem /= shard_heads;
        const int64_t local_tok = rem;
        const int64_t stream_tok = peer * shard_seq + local_tok;

        half2 value = make_half2(0.0f, 0.0f);
        if (stream_tok < stream_real_seq) {
            const int64_t total_tok = is_img ? txt_real_seq + stream_tok : stream_tok;
            const char* src = attn +
                              (2 * d2) * attn_nb0 +
                              head * attn_nb1 +
                              total_tok * attn_nb2 +
                              0 * attn_nb3;
            value = *reinterpret_cast<const half2*>(src);
        }
        dst[linear] = value;
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

static __global__ void qwen_fused_attn_head_to_seq_recv_unpack_keep_f16_h2_kernel(
    const half* __restrict__ recv_flat,
    half2* __restrict__ dst,
    int64_t stream_index,
    int64_t txt_padded_seq,
    int64_t img_padded_seq,
    int64_t world_size,
    int64_t head_dim,
    int64_t heads) {
    const int64_t shard_heads = heads / world_size;
    const int64_t head_dim2 = head_dim / 2;
    const int64_t txt_shard_seq = txt_padded_seq / world_size;
    const int64_t img_shard_seq = img_padded_seq / world_size;
    const int64_t out_shard_seq = stream_index == 0 ? txt_shard_seq : img_shard_seq;
    const int64_t txt_chunk = head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk = head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    const int64_t stream_offset = stream_index == 0 ? 0 : txt_chunk;
    const int64_t total = head_dim2 * heads * out_shard_seq;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int64_t d2 = rem % head_dim2;
        rem /= head_dim2;
        const int64_t head = rem % heads;
        rem /= heads;
        const int64_t local_tok = rem;
        const int64_t src_peer = head / shard_heads;
        const int64_t local_head = head - src_peer * shard_heads;
        const int64_t src_idx =
            src_peer * count_per_peer +
            stream_offset +
            2 * d2 +
            local_head * head_dim +
            local_tok * head_dim * shard_heads;
        dst[linear] = *reinterpret_cast<const half2*>(recv_flat + src_idx);
    }
}

static __global__ void qwen_fused_attn_head_to_seq_recv_unpack_keep_f16_kernel(
    const half* __restrict__ recv_flat,
    half* __restrict__ dst,
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
        dst[linear] = recv_flat[src_idx];
    }
}

static __global__ void qwen_fused_attn_head_to_seq_recv_unpack_keep_bf16_h2_kernel(
    const half* __restrict__ recv_flat,
    nv_bfloat162* __restrict__ dst,
    int64_t stream_index,
    int64_t txt_padded_seq,
    int64_t img_padded_seq,
    int64_t world_size,
    int64_t head_dim,
    int64_t heads) {
    const int64_t shard_heads = heads / world_size;
    const int64_t head_dim2 = head_dim / 2;
    const int64_t txt_shard_seq = txt_padded_seq / world_size;
    const int64_t img_shard_seq = img_padded_seq / world_size;
    const int64_t out_shard_seq = stream_index == 0 ? txt_shard_seq : img_shard_seq;
    const int64_t txt_chunk = head_dim * shard_heads * txt_shard_seq;
    const int64_t img_chunk = head_dim * shard_heads * img_shard_seq;
    const int64_t count_per_peer = txt_chunk + img_chunk;
    const int64_t stream_offset = stream_index == 0 ? 0 : txt_chunk;
    const int64_t total = head_dim2 * heads * out_shard_seq;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = linear;
        const int64_t d2 = rem % head_dim2;
        rem /= head_dim2;
        const int64_t head = rem % heads;
        rem /= heads;
        const int64_t local_tok = rem;
        const int64_t src_peer = head / shard_heads;
        const int64_t local_head = head - src_peer * shard_heads;
        const int64_t src_idx =
            src_peer * count_per_peer +
            stream_offset +
            2 * d2 +
            local_head * head_dim +
            local_tok * head_dim * shard_heads;
        const half2 hv = *reinterpret_cast<const half2*>(recv_flat + src_idx);
        dst[linear] = __float22bfloat162_rn(__half22float2(hv));
    }
}

static __global__ void qwen_fused_attn_head_to_seq_recv_unpack_keep_bf16_kernel(
    const half* __restrict__ recv_flat,
    nv_bfloat16* __restrict__ dst,
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
        dst[linear] = __float2bfloat16(__half2float(recv_flat[src_idx]));
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

    if (qwen_params->mode == 42) {
        GGML_ASSERT(dst->src[0] != nullptr);
        return;
    }

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
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq >= 0);
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
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_stop = nullptr;
        const bool profile = ggml_cuda_qwen_fused_profile_begin(stream, &profile_start, &profile_stop);
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
        if (profile) {
            ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
        }
        return;
    }

    if (qwen_params->mode == 8) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->stream_index == 0 || qwen_params->stream_index == 1);
        GGML_ASSERT(qwen_params->txt_padded_seq > 0 && qwen_params->img_padded_seq >= 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] % qwen_params->world_size == 0);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_stop = nullptr;
        const bool profile = ggml_cuda_qwen_fused_profile_begin(stream, &profile_start, &profile_stop);
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
        if (profile) {
            ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
        }
        return;
    }

    if (qwen_params->mode == 13) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* attn = dst->src[0];
        GGML_ASSERT(attn);
        GGML_ASSERT(attn->type == GGML_TYPE_F32 || attn->type == GGML_TYPE_F16);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->txt_real_seq > 0 && qwen_params->img_real_seq >= 0);
        GGML_ASSERT(qwen_params->txt_padded_seq >= qwen_params->txt_real_seq);
        GGML_ASSERT(qwen_params->img_padded_seq >= qwen_params->img_real_seq);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(attn->ne[2] == qwen_params->txt_real_seq + qwen_params->img_real_seq);
        GGML_ASSERT(attn->ne[3] == 1);

        const bool use_h2 = attn->type == GGML_TYPE_F16 &&
                            attn->nb[0] == sizeof(half) &&
                            (attn->ne[0] % 2) == 0;
        const int64_t total = use_h2 ? ggml_nelements(dst) / 2 : ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_stop = nullptr;
        const bool profile = ggml_cuda_qwen_fused_profile_begin(stream, &profile_start, &profile_stop);
        if (use_h2) {
            qwen_fused_attn_head_to_seq_send_pack_f16_h2_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const char*>(attn->data),
                reinterpret_cast<half2*>(dst->data),
                qwen_params->txt_real_seq,
                qwen_params->img_real_seq,
                qwen_params->txt_padded_seq,
                qwen_params->img_padded_seq,
                qwen_params->world_size,
                attn->ne[0],
                attn->ne[1],
                attn->nb[0], attn->nb[1], attn->nb[2], attn->nb[3]);
        } else {
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
                attn->type == GGML_TYPE_F16,
                attn->nb[0], attn->nb[1], attn->nb[2], attn->nb[3]);
        }
        CUDA_CHECK(cudaGetLastError());
        if (profile) {
            ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
        }
        return;
    }

    if (qwen_params->mode == 14) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F16);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->stream_index == 0 || qwen_params->stream_index == 1);
        GGML_ASSERT(qwen_params->txt_padded_seq > 0 && qwen_params->img_padded_seq >= 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] % qwen_params->world_size == 0);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_stop = nullptr;
        const bool profile = ggml_cuda_qwen_fused_profile_begin(stream, &profile_start, &profile_stop);
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
        if (profile) {
            ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
        }
        return;
    }

    if (qwen_params->mode == 45) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F16);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->stream_index == 0 || qwen_params->stream_index == 1);
        GGML_ASSERT(qwen_params->txt_padded_seq > 0 && qwen_params->img_padded_seq >= 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] % qwen_params->world_size == 0);

        const bool use_h2 = (dst->ne[0] % 2) == 0;
        const int64_t total = use_h2 ? ggml_nelements(dst) / 2 : ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_stop = nullptr;
        const bool profile = ggml_cuda_qwen_fused_profile_begin(stream, &profile_start, &profile_stop);
        if (use_h2) {
            qwen_fused_attn_head_to_seq_recv_unpack_keep_f16_h2_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const half*>(recv_flat->data),
                reinterpret_cast<half2*>(dst->data),
                qwen_params->stream_index,
                qwen_params->txt_padded_seq,
                qwen_params->img_padded_seq,
                qwen_params->world_size,
                dst->ne[0],
                dst->ne[1]);
        } else {
            qwen_fused_attn_head_to_seq_recv_unpack_keep_f16_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const half*>(recv_flat->data),
                static_cast<half*>(dst->data),
                qwen_params->stream_index,
                qwen_params->txt_padded_seq,
                qwen_params->img_padded_seq,
                qwen_params->world_size,
                dst->ne[0],
                dst->ne[1]);
        }
        CUDA_CHECK(cudaGetLastError());
        if (profile) {
            ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
        }
        return;
    }

    if (qwen_params->mode == 48) {
        GGML_ASSERT(dst->type == GGML_TYPE_BF16);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F16);
        GGML_ASSERT(qwen_params->world_size > 0);
        GGML_ASSERT(qwen_params->stream_index == 0 || qwen_params->stream_index == 1);
        GGML_ASSERT(qwen_params->txt_padded_seq > 0 && qwen_params->img_padded_seq >= 0);
        GGML_ASSERT(qwen_params->txt_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(qwen_params->img_padded_seq % qwen_params->world_size == 0);
        GGML_ASSERT(dst->ne[1] % qwen_params->world_size == 0);

        const bool use_h2 = (dst->ne[0] % 2) == 0;
        const int64_t total = use_h2 ? ggml_nelements(dst) / 2 : ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_stop = nullptr;
        const bool profile = ggml_cuda_qwen_fused_profile_begin(stream, &profile_start, &profile_stop);
        if (use_h2) {
            qwen_fused_attn_head_to_seq_recv_unpack_keep_bf16_h2_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const half*>(recv_flat->data),
                reinterpret_cast<nv_bfloat162*>(dst->data),
                qwen_params->stream_index,
                qwen_params->txt_padded_seq,
                qwen_params->img_padded_seq,
                qwen_params->world_size,
                dst->ne[0],
                dst->ne[1]);
        } else {
            qwen_fused_attn_head_to_seq_recv_unpack_keep_bf16_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const half*>(recv_flat->data),
                static_cast<nv_bfloat16*>(dst->data),
                qwen_params->stream_index,
                qwen_params->txt_padded_seq,
                qwen_params->img_padded_seq,
                qwen_params->world_size,
                dst->ne[0],
                dst->ne[1]);
        }
        CUDA_CHECK(cudaGetLastError());
        if (profile) {
            ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
        }
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
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_stop = nullptr;
        const bool profile = ggml_cuda_qwen_fused_profile_begin(stream, &profile_start, &profile_stop);
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
        if (profile) {
            ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
        }
        return;
    }

    if (qwen_params->mode == 33) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k = dst->src[1];
        const ggml_tensor* v = dst->src[2];
        GGML_ASSERT(q && k && v);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(q->ne[0] > 0 && q->ne[0] % 2 == 0);
        GGML_ASSERT(q->ne[1] % world_size == 0);
        GGML_ASSERT(k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0]);
        GGML_ASSERT(k->ne[1] == q->ne[1] && v->ne[1] == q->ne[1]);
        GGML_ASSERT(k->ne[2] == q->ne[2] && v->ne[2] == q->ne[2]);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);
        const int64_t packed_dim = q->ne[0] * 2 + q->ne[0] / 2;
        const int64_t shard_heads = q->ne[1] / world_size;
        GGML_ASSERT(dst->ne[0] == packed_dim * shard_heads * q->ne[2] * world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_fused_qkv_vhalf_send_pack_kernel<<<blocks, threads, 0, stream>>>(
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

    if (qwen_params->mode == 37) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k = dst->src[1];
        const ggml_tensor* v = dst->src[2];
        const ggml_tensor* pe = dst->src[3];
        GGML_ASSERT(q && k && v && pe);
        GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
        GGML_ASSERT(pe->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        const int64_t rank = qwen_params->img_real_seq;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(rank >= 0 && rank < world_size);
        GGML_ASSERT(q->ne[0] > 0 && q->ne[0] % 2 == 0);
        GGML_ASSERT(q->ne[1] % world_size == 0);
        GGML_ASSERT(k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0]);
        GGML_ASSERT(k->ne[1] == q->ne[1] && v->ne[1] == q->ne[1]);
        GGML_ASSERT(k->ne[2] == q->ne[2] && v->ne[2] == q->ne[2]);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);
        GGML_ASSERT(pe->ne[0] == 2);
        GGML_ASSERT(pe->ne[1] == q->ne[0] / 2);
        GGML_ASSERT(pe->ne[2] >= q->ne[2] * world_size);
        GGML_ASSERT(pe->ne[3] == 2);
        const int64_t packed_dim = q->ne[0] * 2;
        const int64_t shard_heads = q->ne[1] / world_size;
        GGML_ASSERT(dst->ne[0] == packed_dim * shard_heads * q->ne[2] * world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_fused_qkv_roped_half_send_pack_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(q->data),
            static_cast<const char*>(k->data),
            static_cast<const char*>(v->data),
            static_cast<const char*>(pe->data),
            static_cast<uint32_t*>(dst->data),
            world_size,
            rank,
            q->ne[0],
            q->ne[1],
            q->ne[2],
            q->nb[0], q->nb[1], q->nb[2], q->nb[3],
            k->nb[0], k->nb[1], k->nb[2], k->nb[3],
            v->nb[0], v->nb[1], v->nb[2], v->nb[3],
            pe->nb[0], pe->nb[1], pe->nb[2], pe->nb[3]);
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
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_stop = nullptr;
        const bool profile = ggml_cuda_qwen_fused_profile_begin(stream, &profile_start, &profile_stop);
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
        if (profile) {
            ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
        }
        return;
    }

    if (qwen_params->mode == 46) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k = dst->src[1];
        const ggml_tensor* v = dst->src[2];
        GGML_ASSERT(q && k && v);
        GGML_ASSERT((q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16) &&
                    (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16) &&
                    (v->type == GGML_TYPE_F32 || v->type == GGML_TYPE_F16));
        const int64_t world_size = qwen_params->txt_real_seq;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(q->ne[1] % world_size == 0);
        GGML_ASSERT(k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0]);
        GGML_ASSERT(k->ne[1] == q->ne[1] && v->ne[1] == q->ne[1]);
        GGML_ASSERT(k->ne[2] == q->ne[2] && v->ne[2] == q->ne[2]);
        GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1);

        const int64_t total = ggml_nelements(dst);
        GGML_ASSERT((total % 2) == 0);
        GGML_ASSERT((q->ne[0] % 2) == 0);
        const int64_t total_h2 = total / 2;
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total_h2 + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_stop = nullptr;
        const bool profile = ggml_cuda_qwen_fused_profile_begin(stream, &profile_start, &profile_stop);
        const bool use_triplet =
            ggml_cuda_flux_sp_fast_qkv_send_pack_f16_enabled() &&
            q->ne[0] <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
            q->ne[1] <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
            q->ne[2] <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
            world_size <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
            q->nb[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            k->nb[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            v->nb[0] <= static_cast<size_t>(std::numeric_limits<int>::max());
        if (use_triplet) {
            const int64_t triplet_total = (q->ne[0] / 2) * (q->ne[1] / world_size) * q->ne[2] * world_size;
            const int triplet_blocks = static_cast<int>(std::min<int64_t>((triplet_total + threads - 1) / threads, 65535));
            flux_sp_qkv_send_pack_f16_triplet_kernel<<<triplet_blocks, threads, 0, stream>>>(
                static_cast<const char*>(q->data),
                static_cast<const char*>(k->data),
                static_cast<const char*>(v->data),
                static_cast<half2*>(dst->data),
                static_cast<int>(world_size),
                static_cast<int>(q->ne[0]),
                static_cast<int>(q->ne[1]),
                static_cast<int>(q->ne[2]),
                static_cast<int>(q->type),
                static_cast<int>(k->type),
                static_cast<int>(v->type),
                q->nb[0], q->nb[1], q->nb[2],
                k->nb[0], k->nb[1], k->nb[2],
                v->nb[0], v->nb[1], v->nb[2]);
            CUDA_CHECK(cudaGetLastError());
            if (profile) {
                ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
            }
            return;
        }
        const int64_t max_i32 = static_cast<int64_t>(std::numeric_limits<int>::max());
        const bool use_i32 =
            q->ne[0] <= max_i32 / 3 &&
            q->ne[1] <= max_i32 &&
            q->ne[2] <= max_i32 &&
            world_size <= max_i32 &&
            total_h2 <= max_i32;
        if (use_i32) {
            qwen_fused_qkv_send_pack_f16_i32_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const char*>(q->data),
                static_cast<const char*>(k->data),
                static_cast<const char*>(v->data),
                static_cast<half2*>(dst->data),
                static_cast<int>(world_size),
                static_cast<int>(q->ne[0]),
                static_cast<int>(q->ne[1]),
                static_cast<int>(q->ne[2]),
                static_cast<int>(q->type),
                static_cast<int>(k->type),
                static_cast<int>(v->type),
                q->nb[0], q->nb[1], q->nb[2], q->nb[3],
                k->nb[0], k->nb[1], k->nb[2], k->nb[3],
                v->nb[0], v->nb[1], v->nb[2], v->nb[3]);
        } else {
            qwen_fused_qkv_send_pack_f16_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const char*>(q->data),
                static_cast<const char*>(k->data),
                static_cast<const char*>(v->data),
                static_cast<half*>(dst->data),
                world_size,
                q->ne[0],
                q->ne[1],
                q->ne[2],
                static_cast<int>(q->type),
                static_cast<int>(k->type),
                static_cast<int>(v->type),
                q->nb[0], q->nb[1], q->nb[2], q->nb[3],
                k->nb[0], k->nb[1], k->nb[2], k->nb[3],
                v->nb[0], v->nb[1], v->nb[2], v->nb[3]);
        }
        CUDA_CHECK(cudaGetLastError());
        if (profile) {
            ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
        }
        return;
    }

    if (qwen_params->mode == 47) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* first_q = dst->src[0];
        const ggml_tensor* first_k = dst->src[1];
        const ggml_tensor* first_v = dst->src[2];
        const ggml_tensor* second_q = dst->src[3];
        const ggml_tensor* second_k = dst->src[4];
        const ggml_tensor* second_v = dst->src[5];
        GGML_ASSERT(first_q && first_k && first_v && second_q && second_k && second_v);
        GGML_ASSERT((first_q->type == GGML_TYPE_F32 || first_q->type == GGML_TYPE_F16) &&
                    (first_k->type == GGML_TYPE_F32 || first_k->type == GGML_TYPE_F16) &&
                    (first_v->type == GGML_TYPE_F32 || first_v->type == GGML_TYPE_F16) &&
                    (second_q->type == GGML_TYPE_F32 || second_q->type == GGML_TYPE_F16) &&
                    (second_k->type == GGML_TYPE_F32 || second_k->type == GGML_TYPE_F16) &&
                    (second_v->type == GGML_TYPE_F32 || second_v->type == GGML_TYPE_F16));
        const int64_t world_size = qwen_params->world_size;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(first_q->ne[1] % world_size == 0);
        GGML_ASSERT(first_k->ne[0] == first_q->ne[0] && first_v->ne[0] == first_q->ne[0] &&
                    second_q->ne[0] == first_q->ne[0] && second_k->ne[0] == first_q->ne[0] && second_v->ne[0] == first_q->ne[0]);
        GGML_ASSERT(first_k->ne[1] == first_q->ne[1] && first_v->ne[1] == first_q->ne[1] &&
                    second_q->ne[1] == first_q->ne[1] && second_k->ne[1] == first_q->ne[1] && second_v->ne[1] == first_q->ne[1]);
        GGML_ASSERT(first_k->ne[2] == first_q->ne[2] && first_v->ne[2] == first_q->ne[2]);
        GGML_ASSERT(second_k->ne[2] == second_q->ne[2] && second_v->ne[2] == second_q->ne[2]);
        GGML_ASSERT(first_q->ne[3] == 1 && first_k->ne[3] == 1 && first_v->ne[3] == 1 &&
                    second_q->ne[3] == 1 && second_k->ne[3] == 1 && second_v->ne[3] == 1);
        GGML_ASSERT((first_q->ne[0] % 2) == 0);

        const int64_t total = ggml_nelements(dst);
        GGML_ASSERT((total % 2) == 0);
        const int64_t total_h2 = total / 2;
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total_h2 + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_stop = nullptr;
        const bool profile = ggml_cuda_qwen_fused_profile_begin(stream, &profile_start, &profile_stop);
        const bool use_triplet =
            ggml_cuda_flux_sp_fast_qkv_send_pack_f16_enabled() &&
            first_q->ne[0] <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
            first_q->ne[1] <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
            first_q->ne[2] <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
            second_q->ne[2] <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
            world_size <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
            first_q->nb[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            first_k->nb[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            first_v->nb[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            second_q->nb[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            second_k->nb[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            second_v->nb[0] <= static_cast<size_t>(std::numeric_limits<int>::max());
        if (use_triplet) {
            const int64_t triplet_total =
                (first_q->ne[0] / 2) *
                (first_q->ne[1] / world_size) *
                (first_q->ne[2] + second_q->ne[2]) *
                world_size;
            const int triplet_blocks = static_cast<int>(std::min<int64_t>((triplet_total + threads - 1) / threads, 65535));
            flux_sp_double_qkv_send_pack_f16_triplet_kernel<<<triplet_blocks, threads, 0, stream>>>(
                static_cast<const char*>(first_q->data),
                static_cast<const char*>(first_k->data),
                static_cast<const char*>(first_v->data),
                static_cast<const char*>(second_q->data),
                static_cast<const char*>(second_k->data),
                static_cast<const char*>(second_v->data),
                static_cast<half2*>(dst->data),
                static_cast<int>(world_size),
                static_cast<int>(first_q->ne[0]),
                static_cast<int>(first_q->ne[1]),
                static_cast<int>(first_q->ne[2]),
                static_cast<int>(second_q->ne[2]),
                static_cast<int>(first_q->type),
                static_cast<int>(first_k->type),
                static_cast<int>(first_v->type),
                static_cast<int>(second_q->type),
                static_cast<int>(second_k->type),
                static_cast<int>(second_v->type),
                first_q->nb[0], first_q->nb[1], first_q->nb[2],
                first_k->nb[0], first_k->nb[1], first_k->nb[2],
                first_v->nb[0], first_v->nb[1], first_v->nb[2],
                second_q->nb[0], second_q->nb[1], second_q->nb[2],
                second_k->nb[0], second_k->nb[1], second_k->nb[2],
                second_v->nb[0], second_v->nb[1], second_v->nb[2]);
            CUDA_CHECK(cudaGetLastError());
            if (profile) {
                ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
            }
            return;
        }
        flux_sp_double_qkv_send_pack_f16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(first_q->data),
            static_cast<const char*>(first_k->data),
            static_cast<const char*>(first_v->data),
            static_cast<const char*>(second_q->data),
            static_cast<const char*>(second_k->data),
            static_cast<const char*>(second_v->data),
            static_cast<half2*>(dst->data),
            static_cast<int>(world_size),
            static_cast<int>(first_q->ne[0]),
            static_cast<int>(first_q->ne[1]),
            static_cast<int>(first_q->ne[2]),
            static_cast<int>(second_q->ne[2]),
            static_cast<int>(first_q->type),
            static_cast<int>(first_k->type),
            static_cast<int>(first_v->type),
            static_cast<int>(second_q->type),
            static_cast<int>(second_k->type),
            static_cast<int>(second_v->type),
            first_q->nb[0], first_q->nb[1], first_q->nb[2],
            first_k->nb[0], first_k->nb[1], first_k->nb[2],
            first_v->nb[0], first_v->nb[1], first_v->nb[2],
            second_q->nb[0], second_q->nb[1], second_q->nb[2],
            second_k->nb[0], second_k->nb[1], second_k->nb[2],
            second_v->nb[0], second_v->nb[1], second_v->nb[2]);
        CUDA_CHECK(cudaGetLastError());
        if (profile) {
            ggml_cuda_qwen_fused_profile_end(stream, profile_start, profile_stop, qwen_params->mode, dst);
        }
        return;
    }

    if (qwen_params->mode == 22) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        const int64_t plane = qwen_params->stream_index;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(plane == 0 || plane == 1);
        GGML_ASSERT(dst->ne[3] == 2);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        const int64_t head_dim = dst->ne[0] * 2;
        const int64_t shard_sequence = dst->ne[1] / world_size;
        const int64_t shard_heads = dst->ne[2];
        GGML_ASSERT(recv_flat->ne[0] == head_dim * 3 * shard_heads * shard_sequence * world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_fused_qk_recv_unpack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(recv_flat->data),
            static_cast<float*>(dst->data),
            world_size,
            plane,
            head_dim,
            shard_heads,
            shard_sequence);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 23) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        const int64_t plane = qwen_params->stream_index;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(plane == 2);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        const int64_t head_dim = dst->ne[0];
        const int64_t shard_sequence = dst->ne[1] / world_size;
        const int64_t shard_heads = dst->ne[2];
        GGML_ASSERT(recv_flat->ne[0] == head_dim * 3 * shard_heads * shard_sequence * world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_fused_v_recv_unpack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(recv_flat->data),
            static_cast<float*>(dst->data),
            world_size,
            head_dim,
            shard_heads,
            shard_sequence);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 26) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        const int64_t plane = qwen_params->stream_index;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(plane == 2);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        const int64_t head_dim = dst->ne[0];
        const int64_t shard_sequence = dst->ne[1] / world_size;
        const int64_t shard_heads = dst->ne[2];
        GGML_ASSERT(recv_flat->ne[0] == head_dim * 3 * shard_heads * shard_sequence * world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_fused_v_recv_unpack_f16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(recv_flat->data),
            static_cast<half*>(dst->data),
            world_size,
            head_dim,
            shard_heads,
            shard_sequence);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 29) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* x = dst->src[0];
        const ggml_tensor* pe = dst->src[1];
        GGML_ASSERT(x && pe);
        GGML_ASSERT(x->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32);
        GGML_ASSERT(x->ne[3] == 2);
        GGML_ASSERT(pe->ne[0] == 2);
        GGML_ASSERT(pe->ne[1] == x->ne[0]);
        GGML_ASSERT(pe->ne[2] >= x->ne[1]);
        GGML_ASSERT(pe->ne[3] == 2);
        GGML_ASSERT(dst->ne[0] == x->ne[0] * 2);
        GGML_ASSERT(dst->ne[1] == x->ne[1]);
        GGML_ASSERT(dst->ne[2] == x->ne[2]);
        GGML_ASSERT(dst->ne[3] == 1);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_rope_seq_major_f16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(x->data),
            static_cast<const char*>(pe->data),
            static_cast<half*>(dst->data),
            x->ne[0],
            x->ne[1],
            x->ne[2],
            x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            pe->nb[0], pe->nb[1], pe->nb[2], pe->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 30) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* x = dst->src[0];
        const ggml_tensor* pe = dst->src[1];
        GGML_ASSERT(x && pe);
        GGML_ASSERT(x->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32);
        GGML_ASSERT(x->ne[3] == 2);
        GGML_ASSERT(pe->ne[0] == 2);
        GGML_ASSERT(pe->ne[1] == x->ne[0]);
        GGML_ASSERT(pe->ne[2] >= x->ne[1]);
        GGML_ASSERT(pe->ne[3] == 2);
        GGML_ASSERT(dst->ne[0] == x->ne[0] * 2);
        GGML_ASSERT(dst->ne[1] == x->ne[1]);
        GGML_ASSERT(dst->ne[2] == x->ne[2]);
        GGML_ASSERT(dst->ne[3] == 1);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_rope_seq_major_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(x->data),
            static_cast<const char*>(pe->data),
            static_cast<float*>(dst->data),
            x->ne[0],
            x->ne[1],
            x->ne[2],
            x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            pe->nb[0], pe->nb[1], pe->nb[2], pe->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 31 || qwen_params->mode == 32) {
        GGML_ASSERT((qwen_params->mode == 31 && dst->type == GGML_TYPE_F32) ||
                    (qwen_params->mode == 32 && dst->type == GGML_TYPE_F16));
        const ggml_tensor* recv_flat = dst->src[0];
        const ggml_tensor* pe = dst->src[1];
        GGML_ASSERT(recv_flat && pe);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        const int64_t plane = qwen_params->stream_index;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(plane == 0 || plane == 1);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        const int64_t head_dim = dst->ne[0];
        const int64_t shard_sequence = dst->ne[1] / world_size;
        const int64_t shard_heads = dst->ne[2];
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(recv_flat->ne[0] == head_dim * 3 * shard_heads * shard_sequence * world_size);
        GGML_ASSERT(pe->ne[0] == 2);
        GGML_ASSERT(pe->ne[1] == head_dim / 2);
        GGML_ASSERT(pe->ne[2] >= dst->ne[1]);
        GGML_ASSERT(pe->ne[3] == 2);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        if (qwen_params->mode == 32) {
            wan_fused_qk_recv_rope_kernel<true><<<blocks, threads, 0, stream>>>(
                static_cast<const char*>(recv_flat->data),
                static_cast<const char*>(pe->data),
                dst->data,
                world_size,
                plane,
                head_dim,
                shard_heads,
                shard_sequence,
                pe->nb[0], pe->nb[1], pe->nb[2], pe->nb[3]);
        } else {
            wan_fused_qk_recv_rope_kernel<false><<<blocks, threads, 0, stream>>>(
                static_cast<const char*>(recv_flat->data),
                static_cast<const char*>(pe->data),
                dst->data,
                world_size,
                plane,
                head_dim,
                shard_heads,
                shard_sequence,
                pe->nb[0], pe->nb[1], pe->nb[2], pe->nb[3]);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 34 || qwen_params->mode == 35) {
        GGML_ASSERT((qwen_params->mode == 34 && dst->type == GGML_TYPE_F32) ||
                    (qwen_params->mode == 35 && dst->type == GGML_TYPE_F16));
        const ggml_tensor* recv_flat = dst->src[0];
        const ggml_tensor* pe = dst->src[1];
        GGML_ASSERT(recv_flat && pe);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        const int64_t plane = qwen_params->stream_index;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(plane == 0 || plane == 1);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        const int64_t head_dim = dst->ne[0];
        const int64_t shard_sequence = dst->ne[1] / world_size;
        const int64_t shard_heads = dst->ne[2];
        const int64_t packed_dim = head_dim * 2 + head_dim / 2;
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(recv_flat->ne[0] == packed_dim * shard_heads * shard_sequence * world_size);
        GGML_ASSERT(pe->ne[0] == 2);
        GGML_ASSERT(pe->ne[1] == head_dim / 2);
        GGML_ASSERT(pe->ne[2] >= dst->ne[1]);
        GGML_ASSERT(pe->ne[3] == 2);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        if (qwen_params->mode == 35) {
            wan_fused_qk_recv_rope_vhalf_kernel<true><<<blocks, threads, 0, stream>>>(
                static_cast<const uint32_t*>(recv_flat->data),
                static_cast<const char*>(pe->data),
                dst->data,
                world_size,
                plane,
                head_dim,
                shard_heads,
                shard_sequence,
                pe->nb[0], pe->nb[1], pe->nb[2], pe->nb[3]);
        } else {
            wan_fused_qk_recv_rope_vhalf_kernel<false><<<blocks, threads, 0, stream>>>(
                static_cast<const uint32_t*>(recv_flat->data),
                static_cast<const char*>(pe->data),
                dst->data,
                world_size,
                plane,
                head_dim,
                shard_heads,
                shard_sequence,
                pe->nb[0], pe->nb[1], pe->nb[2], pe->nb[3]);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 36) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        const int64_t head_dim = dst->ne[0];
        const int64_t shard_sequence = dst->ne[1] / world_size;
        const int64_t shard_heads = dst->ne[2];
        const int64_t packed_dim = head_dim * 2 + head_dim / 2;
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(recv_flat->ne[0] == packed_dim * shard_heads * shard_sequence * world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_fused_vhalf_recv_unpack_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint32_t*>(recv_flat->data),
            static_cast<half*>(dst->data),
            world_size,
            head_dim,
            shard_heads,
            shard_sequence);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 38 || qwen_params->mode == 39 || qwen_params->mode == 40) {
        GGML_ASSERT((qwen_params->mode == 38 && dst->type == GGML_TYPE_F32) ||
                    (qwen_params->mode != 38 && dst->type == GGML_TYPE_F16));
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        const int64_t plane = qwen_params->stream_index;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(plane >= 0 && plane < 3);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        GGML_ASSERT((qwen_params->mode == 38 && plane == 0) ||
                    (qwen_params->mode == 39 && plane == 1) ||
                    (qwen_params->mode == 40 && plane == 2));
        const int64_t head_dim = dst->ne[0];
        const int64_t shard_sequence = dst->ne[1] / world_size;
        const int64_t shard_heads = dst->ne[2];
        const int64_t packed_dim = head_dim * 2;
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(recv_flat->ne[0] == packed_dim * shard_heads * shard_sequence * world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_fused_roped_qkv_recv_unpack_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint32_t*>(recv_flat->data),
            dst->data,
            world_size,
            plane,
            head_dim,
            shard_heads,
            shard_sequence);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 41) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(dst->ne[3] == 2);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        const int64_t head_dim = dst->ne[0];
        const int64_t shard_sequence = dst->ne[1] / world_size;
        const int64_t shard_heads = dst->ne[2];
        const int64_t packed_dim = head_dim * 2;
        GGML_ASSERT(head_dim > 0 && head_dim % 2 == 0);
        GGML_ASSERT(recv_flat->ne[0] == packed_dim * shard_heads * shard_sequence * world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_fused_roped_kv_recv_unpack_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint32_t*>(recv_flat->data),
            static_cast<half*>(dst->data),
            world_size,
            head_dim,
            shard_heads,
            shard_sequence);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 24) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* attn = dst->src[0];
        GGML_ASSERT(attn);
        GGML_ASSERT(attn->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(attn->ne[3] == 1);
        GGML_ASSERT(attn->ne[2] % world_size == 0);
        const int64_t head_dim = attn->ne[0];
        const int64_t shard_heads = attn->ne[1];
        const int64_t sequence = attn->ne[2];
        GGML_ASSERT(dst->ne[0] == head_dim * shard_heads * sequence);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_fused_attn_head_to_seq_send_pack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(attn->data),
            static_cast<float*>(dst->data),
            world_size,
            head_dim,
            shard_heads,
            sequence,
            attn->nb[0], attn->nb[1], attn->nb[2], attn->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (qwen_params->mode == 25) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const ggml_tensor* recv_flat = dst->src[0];
        GGML_ASSERT(recv_flat);
        GGML_ASSERT(recv_flat->type == GGML_TYPE_F32);
        const int64_t world_size = qwen_params->txt_real_seq;
        GGML_ASSERT(world_size > 0);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(dst->ne[1] % world_size == 0);
        const int64_t head_dim = dst->ne[0];
        const int64_t heads = dst->ne[1];
        const int64_t shard_sequence = dst->ne[2];
        GGML_ASSERT(recv_flat->ne[0] == head_dim * (heads / world_size) * shard_sequence * world_size);

        const int64_t total = ggml_nelements(dst);
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
        cudaStream_t stream = ctx.stream();
        wan_fused_attn_head_to_seq_recv_unpack_f32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(recv_flat->data),
            static_cast<float*>(dst->data),
            world_size,
            head_dim,
            heads,
            shard_sequence);
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
