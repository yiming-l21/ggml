#pragma once

#include "common.cuh"

bool ggml_cuda_is_qwen_fused_qkv_pack(const ggml_tensor* dst);
void ggml_cuda_op_qwen_fused_qkv_pack(ggml_backend_cuda_context& ctx, ggml_tensor* dst);
