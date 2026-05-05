#pragma once

#include <future>

#include <pybind11/pybind11.h>
#include <torch/all.h>
#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace py = pybind11;

namespace rtp_llm {

class NormalOutputDispatcher {
public:
    NormalOutputDispatcher() = default;
    explicit NormalOutputDispatcher(py::module_ grammar_batch_ops): grammar_batch_ops_(std::move(grammar_batch_ops)) {}

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const;

    std::future<void> batchAcceptGrammarTokensAsync(const StreamGroups&  stream_groups,
                                                    const torch::Tensor& token_ids_cpu) const;

private:
    void dispatchSingleStream(GenerateStreamPtr    stream,
                              const MergedOutput&  merge_outputs,
                              int                  batch_idx_in,
                              int                  batch_idx_out,
                              int                  token_offset,
                              bool                 return_all_probs,
                              const torch::Tensor& new_tokens_all,
                              const torch::Tensor& token_ids_cpu,
                              const torch::Tensor& success_cpu) const;

    py::module_ grammar_batch_ops_;
};

}  // namespace rtp_llm
