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

#ifdef __cplusplus
}
#endif
