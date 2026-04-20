#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/SplitKvCacheCopy.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <chrono>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace rtp_llm {

namespace {

at::cuda::CUDAStream& getNoBlockCopyStream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    return stream;
}

inline int64_t elapsed_us(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

void execNoBlockCopy(const MultiCopyParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    int copy_device = -1;
    if (!params.multi_dst.empty()) {
        if (params.multi_dst[0].is_cuda()) {
            copy_device = static_cast<int>(params.multi_dst[0].get_device());
        } else if (params.multi_src[0].is_cuda()) {
            copy_device = static_cast<int>(params.multi_src[0].get_device());
        }
        if (copy_device >= 0) {
            check_cuda_value(cudaSetDevice(copy_device));
        }
    }

    auto stream = getNoBlockCopyStream().stream();

    size_t total_bytes = 0;
    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        total_bytes += params.multi_src[i].nbytes();
    }

    auto t_begin = std::chrono::steady_clock::now();
    RTP_LLM_LOG_INFO("[NoBlockCopy] BEGIN device=%d num=%zu total_bytes=%zu split_kv=%d",
                     copy_device, params.multi_src.size(), total_bytes, params.split_kv_layer_num);

    if (params.split_kv_layer_num > 0 && copy_device >= 0) {
        auto t0 = std::chrono::steady_clock::now();
        if (splitKvMultiCopy(params.multi_src,
                             params.multi_dst,
                             params.split_kv_layer_num,
                             static_cast<int64_t>(params.split_kv_cache_stride_bytes),
                             static_cast<int64_t>(params.split_kv_scale_stride_bytes),
                             stream)) {
            auto t1 = std::chrono::steady_clock::now();
            check_cuda_value(cudaStreamSynchronize(stream));
            auto t2 = std::chrono::steady_clock::now();
            check_cuda_error();
            RTP_LLM_LOG_INFO("[NoBlockCopy] END(split_kv) launch_us=%lld sync_us=%lld total_us=%lld",
                             (long long)elapsed_us(t0),
                             (long long)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(),
                             (long long)elapsed_us(t_begin));
            return;
        }
    }

    // Sync path: use cudaMemcpy (synchronous, default stream) instead of
    // cudaMemcpyAsync + cudaStreamSynchronize. Diagnostic: per-copy elapsed.
    int64_t slow_threshold_us = 100000;  // log if any single copy >100ms
    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto ret = cudaMemcpy(params.multi_dst[i].data_ptr(),
                              params.multi_src[i].data_ptr(),
                              params.multi_src[i].nbytes(),
                              cudaMemcpyDefault);
        auto cost_us = elapsed_us(t0);
        if (ret != cudaSuccess) {
            RTP_LLM_LOG_WARNING("[NoBlockCopy] cudaMemcpy[%zu/%zu] FAILED: %s bytes=%zu cost_us=%lld",
                                i, params.multi_src.size(), cudaGetErrorString(ret),
                                params.multi_src[i].nbytes(), (long long)cost_us);
            check_cuda_value(ret);
        }
        if (cost_us > slow_threshold_us) {
            RTP_LLM_LOG_WARNING("[NoBlockCopy] SLOW cudaMemcpy[%zu/%zu] bytes=%zu cost_us=%lld",
                                i, params.multi_src.size(),
                                params.multi_src[i].nbytes(), (long long)cost_us);
        }
    }
    check_cuda_error();
    RTP_LLM_LOG_INFO("[NoBlockCopy] END(sync_memcpy) num=%zu total_bytes=%zu total_us=%lld",
                     params.multi_src.size(), total_bytes, (long long)elapsed_us(t_begin));
}

void warmupNoBlockCopy() {
    if (!warmupSplitKvCopyKernels(at::cuda::getCurrentCUDAStream().stream())) {
        RTP_LLM_LOG_WARNING("warmupSplitKvCopyKernels failed; split-KV copy may JIT on first use");
    }
}

}  // namespace rtp_llm
