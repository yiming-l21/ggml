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

#ifdef __cplusplus
}
#endif
