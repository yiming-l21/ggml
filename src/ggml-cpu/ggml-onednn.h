#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// bf16 weight x bf16 activation matmul via oneDNN (AMX). Returns false on failure
// (caller must fall back to ggml gemm). Activations must already be bf16 (ggml's
// wdata after from_float conversion). See ggml-onednn.cpp for the layout contract.
bool ed_onednn_sgemm_bf16(int64_t n, int64_t m, int64_t k,
                          const void* src0_bf16, int64_t lda_elems,
                          const void* act_bf16,  int64_t ldb_elems,
                          void* dst_f32,          int64_t ldc_elems);

// bf16 2D convolution via oneDNN's native AMX conv primitive (brg_conv_fwd),
// which fuses im2col into the kernel — no giant im2col buffer, unlike ggml's
// im2col+GEMM path. src f32 (NCHW logical), weights bf16 (OIHW logical), dst f32.
// Pointers are to plain contiguous ggml tensor data; oneDNN reorders internally
// (weights packed once and cached). Returns false on failure (caller falls back).
//   N,IC,IH,IW  = input dims; OC,KH,KW = weight dims; OH,OW = output dims
//   sh,sw = stride; ph,pw = pad; dh,dw = dilation (0-based, ggml convention)
bool ed_onednn_conv2d_bf16(int64_t N, int64_t IC, int64_t IH, int64_t IW,
                           int64_t OC, int64_t KH, int64_t KW,
                           int64_t OH, int64_t OW,
                           int64_t sh, int64_t sw, int64_t ph, int64_t pw,
                           int64_t dh, int64_t dw,
                           const void* src_f32,
                           const void* wgt_bf16,
                           void* dst_f32);

// bf16 3D convolution via oneDNN's native AMX conv primitive. Layout contract is
// the 5D extension of ed_onednn_conv2d_bf16: src/dst are plain NCDHW f32, weights
// are plain OIDHW bf16. Returns false on failure so callers can fall back to the
// existing ggml im2col+GEMM implementation.
bool ed_onednn_conv3d_bf16(int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW,
                           int64_t OC, int64_t KD, int64_t KH, int64_t KW,
                           int64_t OD, int64_t OH, int64_t OW,
                           int64_t sd, int64_t sh, int64_t sw,
                           int64_t pd, int64_t ph, int64_t pw,
                           int64_t dd, int64_t dh, int64_t dw,
                           const void* src_f32,
                           const void* wgt_bf16,
                           void* dst_f32);

// Fused flash attention via oneDNN brgemm ukernel (AMX bf16), tiled with online
// softmax so the [n_q, n_kv] score matrix is never materialized. Mirrors ATen's
// cpu_flash_attention. Called per graph node; parallelizes internally over
// (head, q-block) using ggml's thread ids (ith/nth). Returns false if the shape
// is unsupported (caller falls back to ggml flash).
//   q: [d_head, n_q,  n_head,    batch] strides qb1/qb2/qb3 (elements, d contiguous)
//   k: [d_head, n_kv, n_head_kv, batch] strides kb1/kb2/kb3
//   v: [d_head, n_kv, n_head_kv, batch] strides vb1/vb2/vb3  (d contiguous)
//   dst: [d_head, n_head, n_q, batch]  strides ob1/ob2/ob3  (ggml flash output layout)
//   q/k/v are f32 or f16 (per src_type flags); scale applied to QK^T; no mask (mask==null).
bool ed_onednn_flash_attn_bf16(
    int ith, int nth,
    int64_t d_head, int64_t n_q, int64_t n_kv, int64_t n_head, int64_t n_head_kv, int64_t batch,
    float scale,
    const void* q, int q_type, int64_t qb1, int64_t qb2, int64_t qb3,
    const void* k, int k_type, int64_t kb1, int64_t kb2, int64_t kb3,
    const void* v, int v_type, int64_t vb1, int64_t vb2, int64_t vb3,
    void* dst_f32, int64_t ob1, int64_t ob2, int64_t ob3);

#ifdef __cplusplus
}
#endif
