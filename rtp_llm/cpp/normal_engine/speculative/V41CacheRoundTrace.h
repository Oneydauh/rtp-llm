#pragma once

#include <torch/all.h>
#include <openssl/sha.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <optional>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm::v41_cache_round {
inline bool fixedTarget() {
    const char* t       = std::getenv("TEST_TARGET");
    const char* enabled = std::getenv("RTP_V41_FIXED_TARGET");
    return t && enabled && std::strcmp(enabled, "1") == 0
           && std::strcmp(t, "//rtp_llm/test/smoke:deepseek_v41_fixed_target_sm100_arm") == 0;
}
inline bool enabled() {
    static const bool on = [] {
        const char* mode = std::getenv("RTP_V41_CACHE_TRACE");
        const char* test = std::getenv("TEST_TARGET");
        return mode && std::strcmp(mode, "1") == 0 && test
               && (fixedTarget() || std::strcmp(test, "//rtp_llm/test/smoke:deepseek_v41_cache_rounds_sm100_arm") == 0);
    }();
    return on;
}
inline std::string hashIds(const int32_t* values, size_t count) {
    std::vector<unsigned char> bytes(count * 4);
    for (size_t i = 0; i < count; ++i)
        for (int j = 0; j < 4; ++j)
            bytes[4 * i + j] = static_cast<uint32_t>(values[i]) >> (j * 8);
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(bytes.data(), bytes.size(), digest);
    std::ostringstream out;
    for (auto byte : digest)
        out << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}
inline std::string tokenArray(const torch::Tensor& values) {
    auto               t = values.to(torch::kCPU).to(torch::kInt64).contiguous().view({-1});
    std::ostringstream out;
    out << '[';
    for (int64_t i = 0; i < t.numel(); ++i) {
        if (i)
            out << ',';
        out << t.data_ptr<int64_t>()[i];
    }
    out << ']';
    return out.str();
}
struct Round {
    std::string   dir, run, phase, history_sha, suffix_sha;
    int           offset;
    torch::Tensor first_scores;
};
struct Stage {
    std::string dir, run, phase;
};
inline std::optional<Stage> stage() {
    if (!enabled())
        return std::nullopt;
    const char* dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (!dir || !*dir)
        return std::nullopt;
    std::string   run, phase, extra;
    std::ifstream marker(std::string(dir) + "/v41_cache_trace_stage");
    if (!(marker >> run >> phase) || (marker >> extra) || run.size() != 32
        || run.find_first_not_of("0123456789abcdef") != std::string::npos
        || (phase != "cold" && phase != "fill" && phase != "hit"))
        return std::nullopt;
    return Stage{dir, run, phase};
}
inline bool matchesDecodePrompt(const torch::Tensor& prompt) {
    if (!stage() || !prompt.defined() || prompt.dim() != 1 || prompt.numel() != 783 || prompt.scalar_type() != at::kInt)
        return false;
    auto snapshot = prompt.cpu().contiguous();
    return hashIds(snapshot.data_ptr<int32_t>(), 783)
           == "ba6a5e4d28d7dc1a743354919c00be1d8fae404ef6e45500363093d6824829d6";
}
// Only the marked hit, on the sampling root, before verify/Engram rows are built.
// Later rounds and all regular smoke requests retain the native draft proposals.
inline bool replaceFirstVerifyRow(torch::Tensor& proposals, const torch::Tensor& ends, const torch::Tensor& anchors) {
    if (!ends.defined() || ends.numel() != 1 || ends.item<int64_t>() != 783)
        return false;
    TORCH_CHECK(anchors.defined() && anchors.numel() == 1 && anchors.item<int64_t>() == 1606,
                "unexpected anchor for the V4.1 fixed verify fixture");
    TORCH_CHECK(proposals.defined() && proposals.dim() == 2 && proposals.size(0) == 1 && proposals.size(1) == 3,
                "unexpected candidate geometry");
    proposals = torch::tensor({11556, 2521, 304}, proposals.options()).reshape({1, 3});
    return true;
}
inline bool fixFirstCandidates(torch::Tensor&       proposals,
                               const torch::Tensor& ends,
                               const torch::Tensor& anchors,
                               const torch::Tensor& prompt) {
    const auto s = stage();
    return fixedTarget() && s && s->phase == "hit" && matchesDecodePrompt(prompt)
           && replaceFirstVerifyRow(proposals, ends, anchors);
}
// Prepare on the D-side TP sampling root before the actual sampler, finish
// with its outputs after rejection and processor caps. Never access live
// stream history while an async bookkeeping update may be in flight.
inline std::optional<Round> prepare(const SamplerInputs& in) {
    const auto current = stage();
    if (!current)
        return std::nullopt;
    const auto i32rows = [](const torch::Tensor& t) {
        return t.defined() && t.device().is_cpu() && t.dim() == 1 && t.numel() == 4 && t.scalar_type() == at::kInt
               && t.is_contiguous();
    };
    if (!i32rows(in.sequence_lengths) || !i32rows(in.input_lengths) || in.input_lengths[0].item<int>() != 783
        || !in.token_ids.defined() || in.token_ids.dim() != 2 || in.token_ids.size(0) != 4
        || in.token_ids.scalar_type() != at::kInt)
        return std::nullopt;
    const auto length = in.sequence_lengths[0].item<int>();
    if (length < 784 || length >= 807 || length > in.token_ids.size(1))
        return std::nullopt;
    const auto history = in.token_ids[0].narrow(0, 0, length).cpu().contiguous();
    if (hashIds(history.data_ptr<int32_t>(), 783) != "ba6a5e4d28d7dc1a743354919c00be1d8fae404ef6e45500363093d6824829d6")
        return std::nullopt;
    Round ctx{current->dir,
              current->run,
              current->phase,
              hashIds(history.data_ptr<int32_t>(), length),
              hashIds(history.data_ptr<int32_t>() + 783, length - 783),
              length - 783,
              {}};
    if (ctx.offset == 1) {
        TORCH_CHECK(in.logits.defined() && in.logits.dim() == 2 && in.logits.size(0) == 4,
                    "invalid target sampler logits");
        // Sampler can change its inputs in place; retain row 0 on device until
        // its sampled token is known. Only the two selected scalars are read.
        ctx.first_scores = in.logits[0].detach().clone();
    }
    return ctx;
}
inline void finish(const Round&         ctx,
                   const torch::Tensor& draft,
                   const torch::Tensor& target,
                   const torch::Tensor& accepted_count,
                   const torch::Tensor& accepted_ids) {
    const auto integral = [](const torch::Tensor& t) {
        return t.defined() && (t.scalar_type() == at::kInt || t.scalar_type() == at::kLong);
    };
    TORCH_CHECK(integral(draft) && draft.dim() == 2 && draft.size(0) == 1 && draft.size(1) == 3 && integral(target)
                    && target.dim() == 2 && target.size(0) == 4 && target.size(1) > 0 && integral(accepted_ids)
                    && accepted_ids.dim() == 2 && accepted_ids.size(0) == 1 && accepted_ids.size(1) == 4
                    && integral(accepted_count) && accepted_count.numel() == 1,
                "V4.1 diagnostic sampler shape mismatch");
    const int n = accepted_count.item<int>();
    TORCH_CHECK(n >= 1 && n <= 4, "V4.1 diagnostic invalid acceptance length");
    std::ostringstream row;
    row << "{\"run_id\":\"" << ctx.run << "\",\"phase\":\"" << ctx.phase << "\",\"offset\":" << ctx.offset
        << ",\"history_sha256_le_i32\":\"" << ctx.history_sha << "\""
        << ",\"committed_sha256_le_i32\":\"" << ctx.suffix_sha << "\"";
    if (ctx.first_scores.defined()) {
        const auto d = draft[0][0].item<int>(), t = target[0][target.size(1) - 1].item<int>();
        TORCH_CHECK(d >= 0 && t >= 0 && d < ctx.first_scores.numel() && t < ctx.first_scores.numel(),
                    "invalid fixture token id");
        const auto sd = ctx.first_scores[d].item<double>(), st = ctx.first_scores[t].item<double>();
        TORCH_CHECK(std::isfinite(sd) && std::isfinite(st), "invalid fixture logits");
        row << std::setprecision(17) << ",\"first_verify_logits\":{\"draft_id\":" << d
            << ",\"draft_id_target_logit\":" << sd << ",\"sampled_target_id\":" << t
            << ",\"sampled_id_target_logit\":" << st << "}";
        if (fixedTarget() && ctx.phase == "hit") {
            TORCH_CHECK(tokenArray(draft[0]) == "[11556,2521,304]", "fixed candidate proof missing");
            const double a = ctx.first_scores[305].item<double>(), b = ctx.first_scores[16].item<double>();
            TORCH_CHECK(std::isfinite(a) && std::isfinite(b), "nonfinite fixed candidate logits");
            row << ",\"fixed_scores\":{\"305\":" << a << ",\"16\":" << b << "}";
        }
    }
    row << ",\"draft\":" << tokenArray(draft[0]) << ",\"target\":" << tokenArray(target.select(1, target.size(1) - 1))
        << ",\"accept_len\":" << n << ",\"result\":" << tokenArray(accepted_ids[0].narrow(0, 0, n)) << "}\n";
    std::ofstream out(ctx.dir + "/v41_cache_rounds_" + ctx.run + "_" + ctx.phase + ".jsonl", std::ios::app);
    out << row.str();
    out.close();
    TORCH_CHECK(out.good(), "V4.1 diagnostic could not save the sampler record");
}
}  // namespace rtp_llm::v41_cache_round
