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

// Cache keyed by weight pointer. Holds the packed weights, the primitive_desc,
// the JIT'd primitive, and REUSED src/dst memory objects. Constructing a fresh
// dnnl::memory per call for these shapes cost ~8ms (5ms->13ms) and was the real
// bottleneck; we build them once and only swap the data handle on the hot path.
struct Entry {
    int64_t m = 0, n = 0, k = 0;
    matmul::primitive_desc pd;
    matmul prim;
    memory b_packed;
    memory a_mem;   // src, data handle swapped per call
    memory c_mem;   // dst, data handle swapped per call
    bool valid = false;
};
struct Cache {
    std::mutex mtx;
    std::unordered_map<const void*, Entry> map;
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
        {
            std::lock_guard<std::mutex> lk(c.mtx);
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

                // src0 stored [n,k] row-major (stride lda); view as Bt[k,n] {1,lda},
                // reorder ONCE into the packed layout the primitive wants.
                memory::desc b_user({k, n}, dt::bf16, memory::dims{1, lda_elems});
                memory b_user_mem(b_user, eng, const_cast<void*>(src0_bf16));
                ent.b_packed = memory(ent.pd.weights_desc(), eng);
                reorder(b_user_mem, ent.b_packed).execute(s, b_user_mem, ent.b_packed);
                s.wait();

                // Build src/dst memory objects ONCE; the hot path only swaps handles.
                ent.a_mem = memory(ent.pd.src_desc(), eng, const_cast<void*>(act_bf16));
                ent.c_mem = memory(ent.pd.dst_desc(), eng, dst_f32);

                ent.valid = true;
                auto res = c.map.insert_or_assign(src0_bf16, std::move(ent));
                e = &res.first->second;
            }
        }

        // Hot path: reuse cached memory objects, only swap the data handles.
        e->a_mem.set_data_handle(const_cast<void*>(act_bf16));
        e->c_mem.set_data_handle(dst_f32);

        e->prim.execute(s, {{DNNL_ARG_SRC, e->a_mem},
                            {DNNL_ARG_WEIGHTS, e->b_packed},
                            {DNNL_ARG_DST, e->c_mem}});
        s.wait();
        return true;
    } catch (const dnnl::error&) {
        return false;
    }
}
