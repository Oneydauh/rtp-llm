#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace rtp_llm {

NormalBatchStreamProcessor::NormalBatchStreamProcessor(
    const ModelConfig&                 model_config,
    const PDSepConfig&                 pd_sep_config,
    const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
    const CacheConfig&                 cache_config,
    bool                               warm_up) {
    model_input_gatherer_config_.num_layers              = model_config.num_layers;
    model_input_gatherer_config_.vocab_size              = model_config.vocab_size;
    model_input_gatherer_config_.input_vocab_size        = model_config.input_vocab_size;
    model_input_gatherer_config_.has_positional_encoding = model_config.has_positional_encoding;
    model_input_gatherer_config_.is_multimodal           = model_config.mm_model_config.is_multimodal;
    model_input_gatherer_config_.mm_position_ids_style =
        static_cast<PositionIdsStyle>(model_config.mm_model_config.mm_position_ids_style);
    model_input_gatherer_config_.position_id_len_factor     = model_config.attn_config.rope_config.index_factor;
    model_input_gatherer_config_.role_type                  = pd_sep_config.role_type;
    model_input_gatherer_config_.decode_entrance            = pd_sep_config.decode_entrance;
    model_input_gatherer_config_.block_stride_bytes         = cache_config.kv_block_stride_bytes;
    model_input_gatherer_config_.scale_stride_bytes         = cache_config.kv_scale_stride_bytes;
    model_input_gatherer_config_.seq_size_per_block         = cache_config.seq_size_per_block;
    model_input_gatherer_config_.kernel_seq_size_per_block  = cache_config.kernel_seq_size_per_block;
    model_input_gatherer_config_.kernel_blocks_per_kv_block = cache_config.kernelBlocksPerKvBlock();
    model_input_gatherer_config_.kv_cache_group_nums        = cache_config.groupNums();
    model_input_gatherer_config_.layer_to_kv_cache_group_id = cache_config.layer_to_group_id;
    model_input_gatherer_config_.kv_cache_group_types       = cache_config.group_types;
    model_input_gatherer_config_.warm_up                    = warm_up;
    model_input_gatherer_config_.enable_detail_log          = profiling_debug_logging_config.enable_detail_log;

    model_input_gatherer_   = std::make_unique<NormalModelInputGatherer>(model_input_gatherer_config_);
    sampler_input_gatherer_ = std::make_unique<NormalSamplerInputGatherer>();
    // Skip the grammar_batch_ops import when Python isn't initialized — cc_test
    // binaries built without an embedded interpreter never run grammar streams,
    // so leaving grammar_batch_ops_ as a default-constructed empty py::module_
    // is safe. The downstream guards in applyGrammarConstraints() and
    // batchAcceptGrammarTokensAsync() short-circuit before touching it.
    if (Py_IsInitialized()) {
        py::gil_scoped_acquire acquire;
        grammar_batch_ops_ = py::module_::import("rtp_llm.async_decoder_engine.grammar_batch_ops");
    }
    output_dispatcher_ = std::make_unique<NormalOutputDispatcher>(grammar_batch_ops_);
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups& stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    auto status = output_dispatcher_->dispatch(stream_groups, merge_outputs);
    if (status.ok()) {
        // No-GIL pre-check: only invoke the grammar-accept path when at least one
        // stream actually carries a grammar object. hasGrammarObject() short-circuits
        // on m_ptr==nullptr without touching Python, so cc_test runs (no embedded
        // interpreter, no grammars) take this fast path and never call into
        // batchAcceptGrammarTokensAsync (which assumes Python is ready).
        bool any_grammar = false;
        for (auto& stream : stream_groups.allStreams()) {
            if (stream->hasGrammarObject()) {
                any_grammar = true;
                break;
            }
        }
        if (any_grammar) {
            const auto&         token_ids     = merge_outputs.sampler_output.token_ids;
            const torch::Tensor token_ids_cpu = token_ids.defined() ? token_ids.cpu() : torch::Tensor();
            grammar_accept_future_ = output_dispatcher_->batchAcceptGrammarTokensAsync(stream_groups, token_ids_cpu);
        }
    }
    return status;
}

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    return model_input_gatherer_->gather(stream_groups);
}

absl::StatusOr<SamplerInputs> NormalBatchStreamProcessor::gatherSamplerInput(
    const StreamGroups& stream_groups, const GptModelInputs& model_inputs, const GptModelOutputs& model_output) const {
    return sampler_input_gatherer_->gather(stream_groups, model_inputs, model_output);
}

SamplerInputs NormalBatchStreamProcessor::allocateSamplerInputs(const StreamGroups& stream_groups,
                                                                size_t              total_batch_size_in,
                                                                size_t              total_batch_size_out,
                                                                size_t              propose_step) const {
    return sampler_input_gatherer_->allocateSamplerInputs(
        stream_groups, total_batch_size_in, total_batch_size_out, propose_step);
}

void NormalBatchStreamProcessor::fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                                         std::list<GenerateStreamPtr>& all_streams,
                                                         bool                          score_batch,
                                                         size_t                        propose_step) const {
    sampler_input_gatherer_->fillSamplerCommonInputs(sampler_inputs, all_streams, score_batch, propose_step);
}

void NormalBatchStreamProcessor::setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                                          std::list<GenerateStreamPtr>& all_streams,
                                                          bool                          score_batch) const {
    sampler_input_gatherer_->setLogitsProcessorInputs(sampler_inputs, all_streams, score_batch);
}

void NormalBatchStreamProcessor::applyGrammarConstraints(SamplerInputs& inputs) const {
    // Fast-path probe without GIL. Both `static_cast<bool>(obj)` (m_ptr != nullptr)
    // and `is_none()` (m_ptr == Py_None) are pure C++ pointer compares — no refcount
    // / heap / interpreter interaction, so they're safe without the GIL. Safety also
    // relies on two invariants:
    //   (a) `inputs.grammar_objs` was populated upstream on this same executor thread
    //       by `gatherSamplerInput` and is not concurrently mutated during this
    //       forward, so there is no data race on the stored PyObject* members.
    //   (b) `gatherSamplerInput` fills each slot with one of:
    //         - default-empty py::object() (m_ptr=nullptr) → "no grammar"
    //         - py::none() → "no grammar" (legacy, kept for resilience)
    //         - real grammar object → "has grammar"
    //       Both empty and None mean "no grammar" for the purpose of this probe.
    //       The empty case is what cc_test sees (no embedded interpreter, no stream
    //       ever sets a grammar); production sees real grammars or None.
    auto& grammar_objs = inputs.grammar_objs;
    bool  has_grammar  = false;
    for (auto& obj : grammar_objs) {
        if (static_cast<bool>(obj) && !obj.is_none()) {
            has_grammar = true;
            break;
        }
    }
    if (!has_grammar && !grammar_accept_future_.valid()) {
        return;
    }

    if (grammar_accept_future_.valid()) {
        grammar_accept_future_.get();
    }
    if (!has_grammar) {
        return;
    }

    py::gil_scoped_acquire acquire;
    py::object             logits_py = convertTensorToObject(inputs.logits);
    grammar_batch_ops_.attr("batch_apply_grammar_constraints")(grammar_objs, logits_py);
}

}  // namespace rtp_llm
