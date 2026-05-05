#pragma once

#include <future>
#include <list>
#include <memory>

#include <pybind11/pybind11.h>
#include <torch/all.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/normal_engine/NormalSamplerInputGatherer.h"

namespace py = pybind11;

namespace rtp_llm {

class NormalBatchStreamProcessor {
public:
    NormalBatchStreamProcessor(const ModelConfig&                 model_config,
                               const PDSepConfig&                 pd_sep_config,
                               const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                               const CacheConfig&                 cache_config,
                               bool                               warm_up);

    virtual absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const;
    virtual absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const;
    virtual absl::StatusOr<SamplerInputs>  gatherSamplerInput(const StreamGroups&    stream_groups,
                                                              const GptModelInputs&  model_inputs,
                                                              const GptModelOutputs& model_output) const;

    void applyGrammarConstraints(SamplerInputs& inputs) const;

protected:
    SamplerInputs allocateSamplerInputs(const StreamGroups& stream_groups,
                                        size_t              total_batch_size_in,
                                        size_t              total_batch_size_out,
                                        size_t              propose_step = 0) const;

    void setCommonSamplerInputs(SamplerInputs&                sampler_inputs,
                                std::list<GenerateStreamPtr>& all_streams,
                                bool                          score_batch  = false,
                                size_t                        propose_step = 0) const {
        fillSamplerCommonInputs(sampler_inputs, all_streams, score_batch, propose_step);
    }

    void fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                 std::list<GenerateStreamPtr>& all_streams,
                                 bool                          score_batch  = false,
                                 size_t                        propose_step = 0) const;

    void setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                  std::list<GenerateStreamPtr>& all_streams,
                                  bool                          score_batch = false) const;

protected:
    NormalModelInputGathererConfig              model_input_gatherer_config_;
    std::unique_ptr<NormalModelInputGatherer>   model_input_gatherer_;
    std::unique_ptr<NormalSamplerInputGatherer> sampler_input_gatherer_;
    std::unique_ptr<NormalOutputDispatcher>     output_dispatcher_;
    py::module_                                 grammar_batch_ops_;
    // Thread-safety: written in dispatch() and consumed (get()) in
    // applyGrammarConstraints(). Both run on the single executor worker
    // thread, serialized per forward iteration (dispatch -> next forward ->
    // applyGrammarConstraints). `mutable` is only for the const-qualified
    // applyGrammarConstraints accessor; no external synchronization is
    // needed as long as that contract holds.
    mutable std::future<void> grammar_accept_future_;
};

}  // namespace rtp_llm
