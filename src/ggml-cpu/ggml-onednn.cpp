// edge-dit: oneDNN bf16 matmul fast-path for CPU backend.
// Routes large bf16-weight mul_mat through oneDNN (Intel AMX bf16 tile matmul on
// Emerald Rapids). Same math as ggml (bf16 in, f32 accumulate).
//
// Design follows the PyTorch/ideep + ggml PR#855 pattern to avoid the reorder
// overhead that kills naive integrations:
//   - weights: format_tag::any, packed ONCE per weight pointer and cached
//   - src/dst: plain layouts (explicit strides) -> zero per-call reorder
//   - primitive_desc + primitive cached per (weight ptr, M) so no JIT re-cost
// Hooked in AFTER ggml converts src1 activations to bf16 in wdata, so both
// operands arrive bf16. ith==0 enters; oneDNN parallelizes internally.
#include "ggml-onednn.h"

#include <dnnl.hpp>
#include "oneapi/dnnl/dnnl_threadpool.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace dnnl;

namespace {

// Persistent std::thread pool implementing dnnl's threadpool_iface. Using our own
// threads (not OpenMP) is the fix for the nested-parallel bug: ggml calls this from
// inside its own `#pragma omp parallel` region, where an OMP-runtime oneDNN would
// collapse to a single thread (matmul 45ms). Independent threads bypass that.
// Threads are PERSISTENT (spawned once, woken via condvar) — spawning fresh threads
// per matmul cost ~2.7ms each and erased the AMX win across the many matmuls per step.
class GgmlThreadpool : public threadpool_interop::threadpool_iface {
public:
    GgmlThreadpool() {
        int hw = (int) std::thread::hardware_concurrency();
        const char* env = std::getenv("ED_ONEDNN_THREADS");
        n_ = (env && std::atoi(env) > 0) ? std::atoi(env) : (hw > 0 ? hw : 1);
        for (int t = 1; t < n_; t++) {
            workers_.emplace_back([this, t]() { worker_loop(t); });
        }
    }
    ~GgmlThreadpool() override {
        {
            std::unique_lock<std::mutex> lk(m_);
            stop_ = true;
            gen_++;
            cv_.notify_all();
        }
        for (auto& w : workers_) w.join();
    }
    int get_num_threads() const override { return n_; }
    bool get_in_parallel() const override { return false; }
    uint64_t get_flags() const override { return 0; }

    void parallel_for(int n, const std::function<void(int, int)>& fn) override {
        if (n <= 1) { if (n == 1) fn(0, 1); return; }
        const int nt = std::min(n_, n);
        {
            std::unique_lock<std::mutex> lk(m_);
            fn_ = &fn;
            n_work_ = n;
            nt_ = nt;
            remaining_ = nt - 1;   // workers 1..nt-1
            gen_++;
            cv_.notify_all();
        }
        for (int i = 0; i < n; i += nt) fn(i, n);   // thread 0 does its share inline
        if (nt > 1) {
            std::unique_lock<std::mutex> lk(done_m_);
            done_cv_.wait(lk, [this]() { return remaining_ == 0; });
        }
    }

private:
    void worker_loop(int t) {
        uint64_t seen = 0;
        for (;;) {
            std::function<void(int, int)> const* fn;
            int n, nt;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&]() { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
                fn = fn_; n = n_work_; nt = nt_;
            }
            // Only workers assigned to this call (t < nt) do work AND signal done;
            // idle workers (t >= nt) just loop back to wait, so remaining_ (== nt-1)
            // is decremented exactly nt-1 times.
            if (t < nt) {
                for (int i = t; i < n; i += nt) (*fn)(i, n);
                std::unique_lock<std::mutex> lk(done_m_);
                if (--remaining_ == 0) done_cv_.notify_one();
            }
        }
    }

    int n_ = 1;
    std::vector<std::thread> workers_;
    std::mutex m_, done_m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int, int)>* fn_ = nullptr;
    int n_work_ = 0, nt_ = 0;
    std::atomic<int> remaining_{0};
    uint64_t gen_ = 0;
    bool stop_ = false;
};

engine& get_engine() { static engine eng(engine::kind::cpu, 0); return eng; }
GgmlThreadpool& get_tp() { static GgmlThreadpool tp; return tp; }
stream& get_stream() {
    static stream s = [] {
        GgmlThreadpool& tp = get_tp();
        // oneDNN captures matmul thread count (bgmmc.nthr) from get_max_threads() at
        // PRIMITIVE-CREATION time. Under the threadpool runtime that falls back to a
        // single socket's core count (~48 on this 2-socket box) unless we tell it the
        // real concurrency up front — otherwise every matmul runs on only 48 threads.
        dnnl_threadpool_interop_set_max_concurrency(tp.get_num_threads());
        return threadpool_interop::make_stream(get_engine(), &tp);
    }();
    return s;
}

// Two-level cache. The packed weight layout depends only on (weight ptr, n, k) —
// NOT on m (the activation/token count). m varies constantly (per token count, per
// conv patch tile), so keying the weight reorder on m re-packed the same weight on
// every shape change (~1.3s/step of wasted reorders). WeightPack caches the packed
// bf16 weights across all m; Entry caches the m-specific primitive + memory objects.
struct WeightPack {
    int64_t n = 0, k = 0;
    memory b_packed;
    bool valid = false;
};
struct Entry {
    int64_t m = 0, n = 0, k = 0;
    matmul::primitive_desc pd;
    matmul prim;
    memory a_mem;   // src, data handle swapped per call
    memory c_mem;   // dst, data handle swapped per call
    bool valid = false;
};
struct Cache {
    std::mutex mtx;
    std::unordered_map<const void*, WeightPack> wpack;  // weight ptr -> packed weights
    std::unordered_map<const void*, Entry>      map;    // weight ptr -> m-specific primitive
};
Cache& cache() { static Cache c; return c; }

} // namespace

// One 2D slice, bf16 activations. ggml: dst[m,n] = <src0_row[n], act_row[m]>,
// i.e. C[M,N] = A[M,K] @ B[N,K]^T, A = bf16 acts (wdata), B = src0 bf16 weights.
//   n=ne01, m=ne11, k=ne00; lda/ldb/ldc = row strides in elements.
extern "C" bool ed_onednn_sgemm_bf16(int64_t n, int64_t m, int64_t k,
                                     const void* src0_bf16, int64_t lda_elems,
                                     const void* act_bf16,  int64_t ldb_elems,
                                     void* dst_f32,          int64_t ldc_elems) {
    try {
        using dt  = memory::data_type;
        using tag = memory::format_tag;

        engine& eng = get_engine();
        stream& s   = get_stream();

        auto& c = cache();
        Entry* e = nullptr;
        memory* b_packed = nullptr;
        {
            std::lock_guard<std::mutex> lk(c.mtx);

            // Level 2 first: m-specific primitive + src/dst memory objects.
            auto it = c.map.find(src0_bf16);
            if (it != c.map.end() && it->second.valid &&
                it->second.m == m && it->second.n == n && it->second.k == k) {
                e = &it->second;
            } else {
                Entry ent;
                ent.m = m; ent.n = n; ent.k = k;

                // src/dst plain (explicit strides) so they never need reorder.
                // weights `any` so oneDNN picks the AMX-optimal packed layout.
                memory::desc a_md({m, k}, dt::bf16, memory::dims{ldb_elems, 1});
                memory::desc b_any({k, n}, dt::bf16, tag::any);
                memory::desc c_md({m, n}, dt::f32,  memory::dims{ldc_elems, 1});
                ent.pd = matmul::primitive_desc(eng, a_md, b_any, c_md);
                ent.prim = matmul(ent.pd);

                // Build src/dst memory objects ONCE; the hot path only swaps handles.
                ent.a_mem = memory(ent.pd.src_desc(), eng, const_cast<void*>(act_bf16));
                ent.c_mem = memory(ent.pd.dst_desc(), eng, dst_f32);

                ent.valid = true;
                auto res = c.map.insert_or_assign(src0_bf16, std::move(ent));
                e = &res.first->second;
            }

            // Level 1: packed weights, independent of m. Reorder ONCE per (ptr,n,k),
            // reusing the packed layout the primitive wants. m varies constantly
            // (token counts, conv patch tiles); keying the reorder on m re-packed the
            // same weight every shape change (~1.3s/step wasted). Pack keyed on (ptr,n,k).
            auto wit = c.wpack.find(src0_bf16);
            if (wit != c.wpack.end() && wit->second.valid &&
                wit->second.n == n && wit->second.k == k) {
                b_packed = &wit->second.b_packed;
            } else {
                memory::desc b_user({k, n}, dt::bf16, memory::dims{1, lda_elems});
                memory b_user_mem(b_user, eng, const_cast<void*>(src0_bf16));
                WeightPack wp;
                wp.n = n; wp.k = k;
                wp.b_packed = memory(e->pd.weights_desc(), eng);
                reorder(b_user_mem, wp.b_packed).execute(s, b_user_mem, wp.b_packed);
                s.wait();
                wp.valid = true;
                auto wres = c.wpack.insert_or_assign(src0_bf16, std::move(wp));
                b_packed = &wres.first->second.b_packed;
            }
        }

        // Hot path: reuse cached memory objects, only swap the data handles.
        e->a_mem.set_data_handle(const_cast<void*>(act_bf16));
        e->c_mem.set_data_handle(dst_f32);

        e->prim.execute(s, {{DNNL_ARG_SRC, e->a_mem},
                            {DNNL_ARG_WEIGHTS, *b_packed},
                            {DNNL_ARG_DST, e->c_mem}});
        s.wait();
        return true;
    } catch (const dnnl::error&) {
        return false;
    }
}

namespace {
// Conv primitive cache, keyed by weight pointer. Holds the primitive_desc (with
// `any` formats so oneDNN picks AMX-optimal blocking), the JIT'd primitive, and the
// packed weights (reordered once). src/dst are reordered per call when oneDNN's
// preferred layout differs from ggml's plain nchw — cheap vs the conv itself, and
// still far below the old im2col+GEMM materialization.
struct ConvEntry {
    int64_t N=0, IC=0, IH=0, IW=0, OC=0, KH=0, KW=0, OH=0, OW=0;
    int64_t sh=0, sw=0, ph=0, pw=0, dh=0, dw=0;
    convolution_forward::primitive_desc pd;
    convolution_forward prim;
    memory w_packed;   // weights, reordered + cached (data is stable per ptr)
    // Cached src/dst reorder scaffolding so the hot path only swaps data handles
    // (allocating scratch memory + building reorder primitives every call was pure
    // per-call overhead). need_src/need_dst record whether a reorder is required.
    bool   need_src = false, need_dst = false;
    memory src_scratch, dst_scratch;   // blocked-layout buffers (allocated once)
    memory src_user_mem, dst_user_mem; // plain-nchw views; data handle swapped per call
    reorder src_reorder, dst_reorder;
    bool valid = false;
};
struct ConvCache {
    std::mutex mtx;
    std::unordered_map<const void*, ConvEntry> map;
};
ConvCache& conv_cache() { static ConvCache c; return c; }
}  // namespace

extern "C" bool ed_onednn_conv2d_bf16(int64_t N, int64_t IC, int64_t IH, int64_t IW,
                                      int64_t OC, int64_t KH, int64_t KW,
                                      int64_t OH, int64_t OW,
                                      int64_t sh, int64_t sw, int64_t ph, int64_t pw,
                                      int64_t dh, int64_t dw,
                                      const void* src_f32,
                                      const void* wgt_bf16,
                                      void* dst_f32) {
    try {
        using dt  = memory::data_type;
        using tag = memory::format_tag;

        engine& eng = get_engine();
        stream& s   = get_stream();

        auto& c = conv_cache();
        ConvEntry* e = nullptr;
        {
            std::lock_guard<std::mutex> lk(c.mtx);
            auto it = c.map.find(wgt_bf16);
            if (it != c.map.end() && it->second.valid &&
                it->second.N==N && it->second.IC==IC && it->second.IH==IH && it->second.IW==IW &&
                it->second.OC==OC && it->second.KH==KH && it->second.KW==KW &&
                it->second.sh==sh && it->second.sw==sw && it->second.ph==ph && it->second.pw==pw &&
                it->second.dh==dh && it->second.dw==dw) {
                e = &it->second;
            } else {
                ConvEntry ent;
                ent.N=N; ent.IC=IC; ent.IH=IH; ent.IW=IW; ent.OC=OC; ent.KH=KH; ent.KW=KW;
                ent.OH=OH; ent.OW=OW; ent.sh=sh; ent.sw=sw; ent.ph=ph; ent.pw=pw; ent.dh=dh; ent.dw=dw;

                // All `any`: oneDNN picks the AMX-optimal blocked layout for src,
                // weights, and dst. Pinning src or dst to plain nchw forced a
                // pathologically slow conv kernel (100x). The per-call src/dst
                // reorders this incurs are the tax for ggml's plain-layout tensors;
                // still a net win over the im2col+GEMM path.
                memory::desc src_any({N, IC, IH, IW}, dt::bf16, tag::any);
                memory::desc wgt_any({OC, IC, KH, KW}, dt::bf16, tag::any);
                memory::desc dst_any({N, OC, OH, OW}, dt::f32,  tag::any);
                // oneDNN dilation is 0-based (0 == no dilation), same as ggml's d-1.
                ent.pd = convolution_forward::primitive_desc(
                    eng, prop_kind::forward_inference, algorithm::convolution_direct,
                    src_any, wgt_any, dst_any,
                    memory::dims{sh, sw}, memory::dims{dh, dw},
                    memory::dims{ph, pw}, memory::dims{ph, pw});
                ent.prim = convolution_forward(ent.pd);

                // ggml kernel is [KW,KH,IC,OC] contiguous == oihw dims {OC,IC,KH,KW}
                // with plain strides. Reorder ONCE into the packed layout.
                memory::desc w_user({OC, IC, KH, KW}, dt::bf16, tag::oihw);
                memory w_user_mem(w_user, eng, const_cast<void*>(wgt_bf16));
                ent.w_packed = memory(ent.pd.weights_desc(), eng);
                reorder(w_user_mem, ent.w_packed).execute(s, w_user_mem, ent.w_packed);
                s.wait();

                // Build src/dst reorder scaffolding once. src is ggml f32 nchw
                // {N,IC,IH,IW}; dst is ggml f32 nchw {N,OC,OH,OW}. If the primitive
                // wants a different (blocked/bf16) layout, pre-build the scratch
                // buffer + reorder primitive so the hot path only swaps data handles.
                memory::desc src_plain({N, IC, IH, IW}, dt::f32, tag::nchw);
                memory::desc dst_plain({N, OC, OH, OW}, dt::f32, tag::nchw);
                ent.src_user_mem = memory(src_plain, eng, nullptr);
                ent.dst_user_mem = memory(dst_plain, eng, nullptr);
                ent.need_src = (ent.pd.src_desc() != src_plain);
                ent.need_dst = (ent.pd.dst_desc() != dst_plain);
                if (ent.need_src) {
                    ent.src_scratch = memory(ent.pd.src_desc(), eng);
                    ent.src_reorder = reorder(ent.src_user_mem, ent.src_scratch);
                }
                if (ent.need_dst) {
                    ent.dst_scratch = memory(ent.pd.dst_desc(), eng);
                    ent.dst_reorder = reorder(ent.dst_scratch, ent.dst_user_mem);
                }

                ent.valid = true;
                auto res = c.map.insert_or_assign(wgt_bf16, std::move(ent));
                e = &res.first->second;
            }
        }

        // Hot path: swap data handles into the cached memory objects; run cached
        // reorder primitives (built once above). src f32->bf16/blocked in, blocked
        // conv, blocked->f32 nchw out.
        e->src_user_mem.set_data_handle(const_cast<void*>(src_f32));
        e->dst_user_mem.set_data_handle(dst_f32);

        memory& src_mem = e->need_src ? e->src_scratch : e->src_user_mem;
        memory& dst_mem = e->need_dst ? e->dst_scratch : e->dst_user_mem;

        if (e->need_src) {
            e->src_reorder.execute(s, e->src_user_mem, e->src_scratch);
        }

        e->prim.execute(s, {{DNNL_ARG_SRC, src_mem},
                            {DNNL_ARG_WEIGHTS, e->w_packed},
                            {DNNL_ARG_DST, dst_mem}});

        if (e->need_dst) {
            e->dst_reorder.execute(s, e->dst_scratch, e->dst_user_mem);
        }
        s.wait();
        return true;
    } catch (const dnnl::error&) {
        return false;
    }
}
