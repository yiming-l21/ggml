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
#include "oneapi/dnnl/dnnl_ukernel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
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

// Per-M concurrency. oneDNN bakes bgmmc.nthr from get_max_threads() at PRIMITIVE
// CREATION time and parallelizes over output (M,N) blocks. When BOTH M and N are
// small there aren't enough output tiles to feed 192 threads, so each thread's tile
// is tiny and sync dominates: M=256,N<=4096 runs ~1000-1300 GFLOP/s at 192 threads
// but 3300-3600 at 24 threads (same N/K). A large N alone provides enough N-blocks
// (M=256,N=12288 is already fast at 192), so only cap when N is also small.
// A large K is also enough reduction work to feed many threads: M=256,N=3072,K=12288
// (mlp_out) capped to 24 runs 5204 GFLOP/s but its mirror N=12288,K=3072 (same FLOP)
// at 192 hits 9238 — so DON'T cap when K is large either.
// Measured per-shape (real Flux flow): cap helps every M=256,N<=8192,K<=8192 matmul
// (T5 proj + DiT txt qkv/proj) 1.7-3.8x; capping large-N or large-K M=256 REGRESSES.
// ED_MM_NTHR_SMALL overrides the capped thread count (0 = disable, use full).
int ed_pick_nthr(int64_t m, int64_t n, int64_t k, int full) {
    static int small = -2;
    if (small == -2) {
        const char* env = getenv("ED_MM_NTHR_SMALL");
        small = env ? atoi(env) : 24;   // default: cap small-tile matmuls at 24
    }
    if (small <= 0) return full;               // disabled
    if (m <= 512 && n <= 8192 && k <= 8192) return std::min(full, small);
    return full;
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
    int nthr = 0;   // concurrency baked at creation (per-M tuned)
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

                // src/dst plain (explicit strides). Weights default to plain `ba`
                // (column-major, matching ggml's stored [n,k]) fed DIRECTLY with no
                // reorder — exactly what PyTorch/diffusers does. Diffusion/T5 weights are
                // used once per forward, so oneDNN's AMX weight-pack (format_tag::any)
                // never amortizes: the per-call reorder was pure overhead (T5 text-encode
                // 2.28s->1.38s, DiT matmul 7459->9999 GFLOP/s once removed). Set
                // ED_ONEDNN_PACK to restore the packed path for weights reused many times.
                const bool nopack = getenv("ED_ONEDNN_PACK") == nullptr;
                memory::desc a_md({m, k}, dt::bf16, memory::dims{ldb_elems, 1});
                memory::desc b_desc = nopack
                    ? memory::desc({k, n}, dt::bf16, memory::dims{1, lda_elems})  // plain ba, no reorder
                    : memory::desc({k, n}, dt::bf16, tag::any);
                memory::desc c_md({m, n}, dt::f32,  memory::dims{ldc_elems, 1});
                // Bake per-M concurrency into the primitive (oneDNN reads
                // get_max_threads() here). Small M -> fewer threads to avoid
                // over-splitting; restore full afterward so other prims are unaffected.
                ent.nthr = ed_pick_nthr(m, n, k, get_tp().get_num_threads());
                dnnl_threadpool_interop_set_max_concurrency(ent.nthr);
                ent.pd = matmul::primitive_desc(eng, a_md, b_desc, c_md);
                ent.prim = matmul(ent.pd);
                dnnl_threadpool_interop_set_max_concurrency(get_tp().get_num_threads());

                // Build src/dst memory objects ONCE; the hot path only swaps handles.
                ent.a_mem = memory(ent.pd.src_desc(), eng, const_cast<void*>(act_bf16));
                ent.c_mem = memory(ent.pd.dst_desc(), eng, dst_f32);

                ent.valid = true;
                auto res = c.map.insert_or_assign(src0_bf16, std::move(ent));
                e = &res.first->second;
            }

            // Level 1: packed weights. Skipped by default (weight fed directly in plain
            // ba layout, like diffusers — no reorder). ED_ONEDNN_PACK restores packing.
            if (getenv("ED_ONEDNN_PACK") == nullptr) {
                b_packed = nullptr;  // sentinel: use src0 directly below
            } else {
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
        }

        // Hot path: reuse cached memory objects, only swap the data handles.
        e->a_mem.set_data_handle(const_cast<void*>(act_bf16));
        e->c_mem.set_data_handle(dst_f32);

        if (b_packed == nullptr) {
            // NOPACK: weight fed directly in plain ba layout (no reorder), like diffusers.
            memory w_direct(e->pd.weights_desc(), eng, const_cast<void*>(src0_bf16));
            e->prim.execute(s, {{DNNL_ARG_SRC, e->a_mem},
                                {DNNL_ARG_WEIGHTS, w_direct},
                                {DNNL_ARG_DST, e->c_mem}});
        } else {
            e->prim.execute(s, {{DNNL_ARG_SRC, e->a_mem},
                                {DNNL_ARG_WEIGHTS, *b_packed},
                                {DNNL_ARG_DST, e->c_mem}});
        }
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

struct Conv3dEntry {
    int64_t N=0, IC=0, ID=0, IH=0, IW=0, OC=0, KD=0, KH=0, KW=0, OD=0, OH=0, OW=0;
    int64_t sd=0, sh=0, sw=0, pd=0, ph=0, pw=0, dd=0, dh=0, dw=0;
    convolution_forward::primitive_desc pd_desc;
    convolution_forward prim;
    memory w_packed;
    bool need_src = false, need_dst = false;
    memory src_scratch, dst_scratch;
    memory src_user_mem, dst_user_mem;
    reorder src_reorder, dst_reorder;
    bool valid = false;
};
struct Conv3dCache {
    std::mutex mtx;
    std::unordered_map<const void*, Conv3dEntry> map;
};
Conv3dCache& conv3d_cache() { static Conv3dCache c; return c; }
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

extern "C" bool ed_onednn_conv3d_bf16(int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW,
                                      int64_t OC, int64_t KD, int64_t KH, int64_t KW,
                                      int64_t OD, int64_t OH, int64_t OW,
                                      int64_t sd, int64_t sh, int64_t sw,
                                      int64_t pd, int64_t ph, int64_t pw,
                                      int64_t dd, int64_t dh, int64_t dw,
                                      const void* src_f32,
                                      const void* wgt_bf16,
                                      void* dst_f32) {
    try {
        using dt  = memory::data_type;
        using tag = memory::format_tag;

        engine& eng = get_engine();
        stream& s   = get_stream();

        auto& c = conv3d_cache();
        Conv3dEntry* e = nullptr;
        {
            std::lock_guard<std::mutex> lk(c.mtx);
            auto it = c.map.find(wgt_bf16);
            if (it != c.map.end() && it->second.valid &&
                it->second.N==N && it->second.IC==IC && it->second.ID==ID &&
                it->second.IH==IH && it->second.IW==IW && it->second.OC==OC &&
                it->second.KD==KD && it->second.KH==KH && it->second.KW==KW &&
                it->second.OD==OD && it->second.OH==OH && it->second.OW==OW &&
                it->second.sd==sd && it->second.sh==sh && it->second.sw==sw &&
                it->second.pd==pd && it->second.ph==ph && it->second.pw==pw &&
                it->second.dd==dd && it->second.dh==dh && it->second.dw==dw) {
                e = &it->second;
            } else {
                Conv3dEntry ent;
                ent.N=N; ent.IC=IC; ent.ID=ID; ent.IH=IH; ent.IW=IW;
                ent.OC=OC; ent.KD=KD; ent.KH=KH; ent.KW=KW; ent.OD=OD; ent.OH=OH; ent.OW=OW;
                ent.sd=sd; ent.sh=sh; ent.sw=sw; ent.pd=pd; ent.ph=ph; ent.pw=pw; ent.dd=dd; ent.dh=dh; ent.dw=dw;

                memory::desc src_any({N, IC, ID, IH, IW}, dt::bf16, tag::any);
                memory::desc wgt_any({OC, IC, KD, KH, KW}, dt::bf16, tag::any);
                memory::desc dst_any({N, OC, OD, OH, OW}, dt::f32,  tag::any);
                ent.pd_desc = convolution_forward::primitive_desc(
                    eng, prop_kind::forward_inference, algorithm::convolution_direct,
                    src_any, wgt_any, dst_any,
                    memory::dims{sd, sh, sw}, memory::dims{dd, dh, dw},
                    memory::dims{pd, ph, pw}, memory::dims{pd, ph, pw});
                ent.prim = convolution_forward(ent.pd_desc);

                memory::desc w_user({OC, IC, KD, KH, KW}, dt::bf16, tag::oidhw);
                memory w_user_mem(w_user, eng, const_cast<void*>(wgt_bf16));
                ent.w_packed = memory(ent.pd_desc.weights_desc(), eng);
                reorder(w_user_mem, ent.w_packed).execute(s, w_user_mem, ent.w_packed);
                s.wait();

                memory::desc src_plain({N, IC, ID, IH, IW}, dt::f32, tag::ncdhw);
                memory::desc dst_plain({N, OC, OD, OH, OW}, dt::f32, tag::ncdhw);
                ent.src_user_mem = memory(src_plain, eng, nullptr);
                ent.dst_user_mem = memory(dst_plain, eng, nullptr);
                ent.need_src = (ent.pd_desc.src_desc() != src_plain);
                ent.need_dst = (ent.pd_desc.dst_desc() != dst_plain);
                if (ent.need_src) {
                    ent.src_scratch = memory(ent.pd_desc.src_desc(), eng);
                    ent.src_reorder = reorder(ent.src_user_mem, ent.src_scratch);
                }
                if (ent.need_dst) {
                    ent.dst_scratch = memory(ent.pd_desc.dst_desc(), eng);
                    ent.dst_reorder = reorder(ent.dst_scratch, ent.dst_user_mem);
                }

                ent.valid = true;
                auto res = c.map.insert_or_assign(wgt_bf16, std::move(ent));
                e = &res.first->second;
            }
        }

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

// ---------------------------------------------------------------------------
// Fused flash attention via brgemm ukernel (AMX bf16). Online softmax; the
// [n_q,n_kv] scores never fully materialize (only a qBlk x kvBlk tile at a time).
// Correctness-first: scalar exp/softmax (SIMD optimization is a follow-up).
// Requires an oneDNN built with DNNL_EXPERIMENTAL_UKERNEL; otherwise the entry
// point below returns false and the caller falls back to ggml's flash attention.
// ---------------------------------------------------------------------------
#include <cmath>
#include <cstring>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef DNNL_EXPERIMENTAL_UKERNEL
namespace {
using namespace dnnl::ukernel;

inline float ed_ld(const void* base, int type, int64_t idx){
    // type: 0=f32, 1=f16
    if (type == 0) return ((const float*)base)[idx];
    // f16 -> f32
    uint16_t h = ((const uint16_t*)base)[idx];
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  =  h & 0x3ff;
    uint32_t f;
    if (exp == 0){ if(man==0){ f=sign; } else { exp=127-15+1; while(!(man&0x400)){man<<=1;exp--;} man&=0x3ff; f=sign|(exp<<23)|(man<<13);} }
    else if (exp == 0x1f){ f = sign | 0x7f800000u | (man<<13); }
    else { f = sign | ((exp + (127-15))<<23) | (man<<13); }
    float r; std::memcpy(&r,&f,4); return r;
}
inline uint16_t ed_f2bf16(float f){ uint32_t u; std::memcpy(&u,&f,4); return (uint16_t)((u + 0x7fff + ((u>>16)&1)) >> 16); }

#if defined(__AVX512F__)
// AVX-512 exp — copied verbatim from ggml's ggml_v_expf (vec.h). The previous
// hand-rolled version had a broken manual 2^n reconstruction (returned ~2x the
// true exp everywhere, garbage for large-negative inputs), which corrupted the
// flash softmax denominator whenever attention scores were large (peaked softmax)
// — the SD3 d=64 "garbage image" bug. This proven version uses _mm512_scalef_ps
// for the 2^n scaling and the full 5-term minimax poly.
inline __m512 ed_v_expf(__m512 x) {
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b =
        _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                         _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __mmask16 d =
        _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                        _mm512_set1_ps(0x1.573e2ep-5f)),
                        u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                        _mm512_set1_ps(0x1.fffdb6p-2f))),
        u,
        _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
    const __m512 res = _mm512_scalef_ps(j, n);
    if (_mm512_kortestz(d, d))
        return res;
    const __m512 zero = _mm512_setzero_ps();
    const __m512 alt = _mm512_mask_blend_ps(
        _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
    return _mm512_mask_blend_ps(d, res, alt);
}
// Fused: for row srow[0..kn), compute p=exp((srow*scale)-mcur), store bf16 into
// pbf[0..kvBlk), return sum(p). Scalar tail for kn%16.
inline float ed_softmax_pack_row(const float* srow, float scale, float mcur,
                                 uint16_t* pbf, int64_t kn, int64_t kvBlk) {
    int64_t j = 0;
    __m512 vsum = _mm512_setzero_ps();
    const __m512 vscale = _mm512_set1_ps(scale), vmcur = _mm512_set1_ps(mcur);
    for (; j + 16 <= kn; j += 16) {
        __m512 s = _mm512_mul_ps(_mm512_loadu_ps(srow + j), vscale);
        __m512 p = ed_v_expf(_mm512_sub_ps(s, vmcur));
        vsum = _mm512_add_ps(vsum, p);
        // f32 -> bf16 (round-to-nearest-even), store 16 x uint16
        __m512i u = _mm512_castps_si512(p);
        __m512i r = _mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7fff)),
                                     _mm512_and_epi32(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1)));
        _mm256_storeu_si256((__m256i*)(pbf + j), _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16)));
    }
    float sum = _mm512_reduce_add_ps(vsum);
    for (; j < kn; ++j) { float p = std::exp(srow[j]*scale - mcur); pbf[j] = ed_f2bf16(p); sum += p; }
    for (int64_t t = kn; t < kvBlk; ++t) pbf[t] = 0;
    return sum;
}
// Vectorized row convert: d contiguous elements (f32 or f16) -> bf16 out[0..d).
// f16 uses the F16C hardware cvt; f32 packs directly. Falls back to scalar tail.
inline void ed_row_to_bf16(const void* src, int type, uint16_t* out, int64_t d) {
    int64_t dd = 0;
    if (type == 0) {  // f32
        const float* s = (const float*)src;
        for (; dd + 16 <= d; dd += 16) {
            __m512i u = _mm512_castps_si512(_mm512_loadu_ps(s + dd));
            __m512i r = _mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7fff)),
                                         _mm512_and_epi32(_mm512_srli_epi32(u,16), _mm512_set1_epi32(1)));
            _mm256_storeu_si256((__m256i*)(out+dd), _mm512_cvtepi32_epi16(_mm512_srli_epi32(r,16)));
        }
    } else {  // f16
        const uint16_t* s = (const uint16_t*)src;
        for (; dd + 16 <= d; dd += 16) {
            __m512 f = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(s+dd)));  // F16C
            __m512i u = _mm512_castps_si512(f);
            __m512i r = _mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7fff)),
                                         _mm512_and_epi32(_mm512_srli_epi32(u,16), _mm512_set1_epi32(1)));
            _mm256_storeu_si256((__m256i*)(out+dd), _mm512_cvtepi32_epi16(_mm512_srli_epi32(r,16)));
        }
    }
    for (; dd < d; ++dd) out[dd] = ed_f2bf16(ed_ld(src, type, dd));
}
#endif


// Per-thread cached brgemm kernels + packing (shapes fixed per call site; qBlk/kvBlk const).
struct FlashKernels {
    int64_t d=0, qB=0, kB=0;
    brgemm qk, pv;
    transform packKT, packV;
    size_t qk_sp=0, pv_sp=0;
    bool ok=false;
    void build(int64_t d_, int64_t qB_, int64_t kB_){
        if (ok && d==d_ && qB==qB_ && kB==kB_) return;
        d=d_; qB=qB_; kB=kB_;
        using dt = memory::data_type;
        qk = brgemm(qB,kB,d,1,d,kB,kB, dt::bf16,dt::bf16,dt::f32); qk.finalize(); qk.generate(); qk_sp=qk.get_scratchpad_size();
        pv = brgemm(qB,d,kB,1,kB,d,d,  dt::bf16,dt::bf16,dt::f32); pv.finalize(); pv.generate(); pv_sp=pv.get_scratchpad_size();
        packKT = transform(d,kB,pack_type::trans,   d,kB, dt::bf16,dt::bf16); packKT.generate();
        packV  = transform(kB,d,pack_type::no_trans,d,d,  dt::bf16,dt::bf16); packV.generate();
        ok=true;
    }
};
thread_local FlashKernels g_fk;

} // namespace

extern "C" bool ed_onednn_flash_attn_bf16(
    int ith, int nth,
    int64_t d_head, int64_t n_q, int64_t n_kv, int64_t n_head, int64_t n_head_kv, int64_t batch,
    float scale,
    const void* q, int q_type, int64_t qb1, int64_t qb2, int64_t qb3,
    const void* k, int k_type, int64_t kb1, int64_t kb2, int64_t kb3,
    const void* v, int v_type, int64_t vb1, int64_t vb2, int64_t vb3,
    void* dst_f32, int64_t ob1, int64_t ob2, int64_t ob3)
{
    try {
        const int64_t qBlk = 256, kvBlk = 512;
        if (d_head > 256) return false;               // tile buffers sized for <=256
        const int64_t nqb = (n_q + qBlk - 1) / qBlk;
        const int64_t heads_ratio = n_head / (n_head_kv > 0 ? n_head_kv : 1);

        g_fk.build(d_head, qBlk, kvBlk);
        g_fk.qk.set_hw_context();

        // thread-local scratch (grown as needed)
        thread_local std::vector<uint8_t> sp_qk, sp_pv;
        thread_local std::vector<uint16_t> Qb, KTp, Vp, Pb;
        thread_local std::vector<float> S, acc, m_i, l_i;
        sp_qk.resize(g_fk.qk_sp); sp_pv.resize(g_fk.pv_sp);
        Qb.resize((size_t)qBlk*d_head);
        KTp.resize((size_t)d_head*kvBlk); Vp.resize((size_t)kvBlk*d_head);
        S.resize((size_t)qBlk*kvBlk); Pb.resize((size_t)qBlk*kvBlk);
        acc.resize((size_t)qBlk*d_head); m_i.resize(qBlk); l_i.resize(qBlk);
        std::vector<std::pair<memory::dim,memory::dim>> off = {{0,0}};

        const int64_t total = batch * n_head * nqb;   // work items
        for (int64_t it = ith; it < total; it += nth) {
            int64_t qb = it % nqb;
            int64_t h  = (it / nqb) % n_head;
            int64_t b  = it / (nqb * n_head);
            int64_t hk = h / heads_ratio;             // kv head (GQA); =h if MHA
            int64_t q0 = qb * qBlk, qn = std::min(qBlk, n_q - q0);

            // ---- load & bf16-pack Q block: Qb[qn, d] ----
            for (int64_t i=0;i<qn;++i){
                const void* qrow_base = (const char*)q + (size_t)(b*qb3 + h*qb2 + (q0+i)*qb1)*(q_type==0?4:2);
#if defined(__AVX512F__)
                ed_row_to_bf16(qrow_base, q_type, &Qb[(size_t)i*d_head], d_head);
#else
                for (int64_t dd=0; dd<d_head; ++dd) Qb[(size_t)i*d_head+dd] = ed_f2bf16(ed_ld(qrow_base,q_type,dd));
#endif
            }
            for (int64_t i=0;i<qn;++i){ m_i[i]=-1e30f; l_i[i]=0; }
            std::fill(acc.begin(), acc.begin()+(size_t)qn*d_head, 0.0f);

            for (int64_t k0=0;k0<n_kv;k0+=kvBlk){
                int64_t kn = std::min(kvBlk, n_kv-k0);
                // Build K block in NATURAL [kv, d] layout (bf16); packKT (pack_type::trans,
                // in_ld=d) transposes+packs it into the [d, kvBlk] B operand for QK^T.
                // Build V block [kv, d] (bf16); packV (no_trans) packs as-is.
                thread_local std::vector<uint16_t> Kraw, Vraw;
                Kraw.assign((size_t)kvBlk*d_head, 0); Vraw.assign((size_t)kvBlk*d_head, 0);
                for (int64_t j=0;j<kn;++j){
                    const void* krow = (const char*)k + (size_t)(b*kb3 + hk*kb2 + (k0+j)*kb1)*(k_type==0?4:2);
                    const void* vrow = (const char*)v + (size_t)(b*vb3 + hk*vb2 + (k0+j)*vb1)*(v_type==0?4:2);
#if defined(__AVX512F__)
                    ed_row_to_bf16(krow, k_type, &Kraw[(size_t)j*d_head], d_head);
                    ed_row_to_bf16(vrow, v_type, &Vraw[(size_t)j*d_head], d_head);
#else
                    for (int64_t dd=0; dd<d_head; ++dd){
                        Kraw[(size_t)j*d_head + dd] = ed_f2bf16(ed_ld(krow,k_type,dd)); // [kv,d]
                        Vraw[(size_t)j*d_head + dd] = ed_f2bf16(ed_ld(vrow,v_type,dd)); // [kv,d]
                    }
#endif
                }
                g_fk.packKT.execute(Kraw.data(), KTp.data());  // trans: [kv,d] -> packed [d,kvBlk]
                g_fk.packV.execute(Vraw.data(),  Vp.data());   // no_trans: [kv,d] packed
                // S[qn,kvBlk] = Qb * KT
                g_fk.qk.execute(Qb.data(), KTp.data(), off, S.data(), sp_qk.data());
                // online softmax over [0,kn)
                for (int64_t i=0;i<qn;++i){
                    float* srow = &S[(size_t)i*kvBlk];
                    float mprev=m_i[i], mcur=mprev;
#if defined(__AVX512F__)
                    // vectorized max of (srow*scale) over [0,kn)
                    {
                        int64_t j=0; __m512 vmax=_mm512_set1_ps(mcur); const __m512 vs=_mm512_set1_ps(scale);
                        for (; j+16<=kn; j+=16) vmax=_mm512_max_ps(vmax,_mm512_mul_ps(_mm512_loadu_ps(srow+j),vs));
                        mcur=_mm512_reduce_max_ps(vmax);
                        for (; j<kn; ++j){ float v=srow[j]*scale; if(v>mcur)mcur=v; }
                    }
                    float alpha = std::exp(mprev-mcur);
                    float lsum = ed_softmax_pack_row(srow, scale, mcur, &Pb[(size_t)i*kvBlk], kn, kvBlk);
#else
                    for (int64_t j=0;j<kn;++j){ srow[j]*=scale; if(srow[j]>mcur)mcur=srow[j]; }
                    float alpha = std::exp(mprev-mcur), lsum=0;
                    for (int64_t j=0;j<kn;++j){ float p=std::exp(srow[j]-mcur); Pb[(size_t)i*kvBlk+j]=ed_f2bf16(p); lsum+=p; }
                    for (int64_t j=kn;j<kvBlk;++j) Pb[(size_t)i*kvBlk+j]=0;  // zero pad
#endif
                    float* arow=&acc[(size_t)i*d_head]; for(int64_t dd=0;dd<d_head;++dd) arow[dd]*=alpha;
                    l_i[i]=l_i[i]*alpha+lsum; m_i[i]=mcur;
                }
                // acc[qn,d] += P[qn,kvBlk] * Vp[kvBlk,d]  (tail rows of V are zero -> no effect)
                thread_local std::vector<float> pvout; pvout.resize((size_t)qBlk*d_head);
                g_fk.pv.execute(Pb.data(), Vp.data(), off, pvout.data(), sp_pv.data());
                for (int64_t i=0;i<qn;++i){ float*a=&acc[(size_t)i*d_head]; float*p=&pvout[(size_t)i*d_head]; for(int64_t dd=0;dd<d_head;++dd) a[dd]+=p[dd]; }
            }
            // write dst[d_head, n_head, n_q, batch] layout: dst[:, h, q0+i, b]
            for (int64_t i=0;i<qn;++i){
                float inv = 1.0f / l_i[i]; float* a=&acc[(size_t)i*d_head];
                float* orow = (float*)dst_f32 + (size_t)(b*ob3 + (q0+i)*ob2 + h*ob1);
                for (int64_t dd=0; dd<d_head; ++dd) orow[dd] = a[dd]*inv;
            }
        }
        return true;
    } catch (const dnnl::error&) {
        return false;
    }
}

#else  // !DNNL_EXPERIMENTAL_UKERNEL — oneDNN lacks the brgemm ukernel; no fused flash.
extern "C" bool ed_onednn_flash_attn_bf16(
    int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float,
    const void*, int, int64_t, int64_t, int64_t,
    const void*, int, int64_t, int64_t, int64_t,
    const void*, int, int64_t, int64_t, int64_t,
    void*, int64_t, int64_t, int64_t) {
    return false;  // caller falls back to ggml flash attention
}
#endif // DNNL_EXPERIMENTAL_UKERNEL
