#include "rtp_llm/cpp/engine_base/schedulers/GrammarManager.h"

#include <chrono>
#include <future>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace {

std::string pyObjTypeName(const py::object& obj) {
    try {
        if (obj.is_none()) {
            return "None";
        }
        py::object  cls   = obj.attr("__class__");
        std::string mod   = py::str(cls.attr("__module__")).cast<std::string>();
        std::string cname = py::str(cls.attr("__name__")).cast<std::string>();
        return mod + "." + cname;
    } catch (...) {
        return "<unknown_py_type>";
    }
}

bool isLikelyXGrammarObject(const py::object& obj) {
    std::string type_name = pyObjTypeName(obj);
    return type_name.find("xgram") != std::string::npos || type_name.find("XGram") != std::string::npos
           || type_name.find("XGrammar") != std::string::npos;
}

std::string keyBrief(const GrammarKey& key) {
    return key.key_type + "(len=" + std::to_string(key.key_string.size()) + ")";
}

}  // namespace

GrammarManager::GrammarManager(py::object grammar_backend, int num_workers, int64_t compile_timeout_ms):
    grammar_backend_(std::move(grammar_backend)) {
    if (compile_timeout_ms > 0) {
        grammar_compile_timeout_ms_ = compile_timeout_ms;
    }

    // No backend → "disabled" mode used by cc_test ctors. Skip every
    // Python touch (GIL acquire, hasattr, attr lookup, worker spawn). The
    // manager then short-circuits in process_req_with_grammar and friends
    // via hasBackend() guards.
    if (!hasBackend()) {
        RTP_LLM_LOG_INFO("GrammarManager init: backend=disabled, compile_timeout_ms=%lld",
                         static_cast<long long>(grammar_compile_timeout_ms_));
        return;
    }

    {
        py::gil_scoped_acquire acquire;
        if (py::hasattr(grammar_backend_, "_invalid_grammar_cls")) {
            invalid_grammar_cls_ = grammar_backend_.attr("_invalid_grammar_cls");
        }
        RTP_LLM_LOG_INFO("GrammarManager init: backend_type=%s, compile_timeout_ms=%lld, num_workers=%d",
                         pyObjTypeName(grammar_backend_).c_str(),
                         static_cast<long long>(grammar_compile_timeout_ms_),
                         num_workers);
    }

    if (num_workers < 1) {
        num_workers = 1;
    }
    workers_.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

GrammarManager::~GrammarManager() {
    // Signal and wake all workers. Any outstanding compile will run to
    // completion; its payload will be discarded since nothing reads the
    // corresponding future after we clear grammar_queue_.
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    worker_cv_.notify_all();

    // Disabled mode: no workers were spawned and members are empty
    // py::object() (m_ptr=nullptr). Their dtors are no-ops, no GIL needed.
    if (!hasBackend()) {
        return;
    }

    {
        // Workers may be inside compile_now waiting to re-acquire GIL.
        // If the caller holds GIL, t.join() deadlocks — release for the
        // join. PyGILState_Check guards against double-release when the
        // caller already released GIL.
        std::optional<py::gil_scoped_release> release;
        if (PyGILState_Check()) {
            release.emplace();
        }
        for (auto& t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    // Drop python handles and queue entries under GIL for clean refcount.
    if (Py_IsInitialized()) {
        try {
            py::gil_scoped_acquire acquire;
            grammar_queue_.clear();
            compile_tasks_.clear();
            grammar_backend_     = py::object();
            invalid_grammar_cls_ = py::object();
        } catch (...) {}
    }
}

size_t GrammarManager::size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return grammar_queue_.size();
}

bool GrammarManager::has_waiting_grammars() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return !grammar_queue_.empty();
}

void GrammarManager::clear() {
    // Swap out the entries under lock, then process them without the lock
    // (set stop on each stream, drop grammar state). Also ask the Python
    // backend to reset its cache.
    std::list<GrammarEntry> drained;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        drained.swap(grammar_queue_);
        compile_tasks_.clear();
    }

    RTP_LLM_LOG_INFO("GrammarManager clear: drained=%zu", drained.size());

    if (!hasBackend()) {
        // Disabled mode: drained is empty (no entries are ever queued
        // when there's no backend); just clear it and return.
        drained.clear();
        return;
    }

    py::gil_scoped_acquire acquire;
    try {
        grammar_backend_.attr("reset")();
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("grammar backend reset failed: %s", e.what());
    }
    for (auto& entry : drained) {
        if (entry.stream) {
            entry.stream->clearGrammarObject();
        }
    }
    // Destroy drained while the GIL is still held — each entry's shared
    // future may own a GrammarReadyPayload::grammar_obj (py::object).
    drained.clear();
}

bool GrammarManager::isGrammarRequested(const GenerateStreamPtr& stream) const {
    auto& config = stream->generateConfig();
    return config->json_schema.has_value() || config->regex.has_value() || config->ebnf.has_value()
           || config->structural_tag.has_value();
}

GrammarKey GrammarManager::extractGrammarKey(const GenerateStreamPtr& stream) const {
    auto& config = stream->generateConfig();
    if (config->json_schema.has_value()) {
        return {"json", config->json_schema.value()};
    } else if (config->regex.has_value()) {
        return {"regex", config->regex.value()};
    } else if (config->ebnf.has_value()) {
        return {"ebnf", config->ebnf.value()};
    } else if (config->structural_tag.has_value()) {
        return {"structural_tag", config->structural_tag.value()};
    }
    return {};
}

py::tuple GrammarManager::grammarKeyToPyTuple(const GrammarKey& key) const {
    return py::make_tuple(key.key_type, key.key_string);
}

bool GrammarManager::isInvalidGrammar(const py::object& obj) const {
    if (!static_cast<bool>(invalid_grammar_cls_) || invalid_grammar_cls_.is_none()) {
        return false;
    }
    return py::isinstance(obj, invalid_grammar_cls_);
}

std::string GrammarManager::extractInvalidGrammarError(const py::object& obj) const {
    try {
        return py::str(obj.attr("error_message")).cast<std::string>();
    } catch (...) {
        return "<no error_message attr>";
    }
}

void GrammarManager::replayPrefillTokensToGrammar(const GenerateStreamPtr& stream, py::object& grammar_obj) {
    size_t output_len = stream->outputTokenLen();
    if (output_len == 0) {
        return;
    }
    auto all_tokens = stream->completeTokenIdsVec(0);
    int  input_len  = stream->inputLength();
    RTP_LLM_LOG_INFO("stream [%ld] grammar replay prefill tokens: output_len=%zu, input_len=%d",
                     stream->streamId(),
                     output_len,
                     input_len);
    try {
        for (size_t i = 0; i < output_len; ++i) {
            int token_id = all_tokens[input_len + i];
            grammar_obj.attr("accept_token")(token_id);
            RTP_LLM_LOG_INFO("stream [%ld] grammar replay accept token_id=%d (%zu/%zu)",
                             stream->streamId(),
                             token_id,
                             i + 1,
                             output_len);
        }
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("stream [%ld] grammar replay failed: %s", stream->streamId(), e.what());
        stream->reportError(ErrorCode::INVALID_PARAMS, std::string("grammar replay prefill tokens error: ") + e.what());
    }
}

bool GrammarManager::process_req_with_grammar(const GenerateStreamPtr& stream) {
    // ------------------------------------------------------------------
    // Fast path: under GIL, check request type, short-circuit on cache hit.
    // If we end up needing async compile we *release* GIL before taking
    // queue_mutex_ to enforce the lock order (queue_mutex_ before GIL).
    // ------------------------------------------------------------------
    RTP_LLM_LOG_INFO("stream [%ld] process_req_with_grammar ENTER", stream ? stream->streamId() : -1);

    // Disabled mode (no backend, no Python): short-circuit before any GIL
    // acquire. isGrammarRequested only reads the C++ generate config — safe
    // to call without GIL.
    if (!hasBackend()) {
        if (isGrammarRequested(stream)) {
            stream->reportError(ErrorCode::INVALID_PARAMS,
                                "Grammar-based generation requested but grammar backend is disabled");
        }
        return false;
    }

    bool       require_reasoning = false;
    GrammarKey key;
    {
        py::gil_scoped_acquire acquire;

        if (!isGrammarRequested(stream)) {
            stream->clearGrammarObject();
            RTP_LLM_LOG_INFO("stream [%ld] no grammar constraints, bypass grammar queue", stream->streamId());
            return false;
        }

        key               = extractGrammarKey(stream);
        require_reasoning = stream->generateConfig()->in_think_mode;

        RTP_LLM_LOG_INFO("stream [%ld] grammar preprocess begin: key=%s, require_reasoning=%d",
                         stream->streamId(),
                         keyBrief(key).c_str(),
                         static_cast<int>(require_reasoning));

        // Try synchronous cache hit first.
        try {
            py::object cached = grammar_backend_.attr("get_cached")(grammarKeyToPyTuple(key), require_reasoning);
            if (!cached.is_none()) {
                if (isInvalidGrammar(cached)) {
                    std::string err = extractInvalidGrammarError(cached);
                    stream->reportError(ErrorCode::INVALID_PARAMS,
                                    "Failed to compile " + key.key_type + " grammar: " + err);
                    return false;
                }
                stream->setGrammarObject(cached);
                replayPrefillTokensToGrammar(stream, cached);
                RTP_LLM_LOG_INFO("stream [%ld] grammar cache hit accepted: key=%s, grammar_type=%s, likely_xgrammar=%d",
                                 stream->streamId(),
                                 keyBrief(key).c_str(),
                                 pyObjTypeName(cached).c_str(),
                                 static_cast<int>(isLikelyXGrammarObject(cached)));
                return false;
            }
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING("stream [%ld] grammar backend get_cached exception: %s", stream->streamId(), e.what());
            stream->reportError(ErrorCode::INVALID_PARAMS, std::string("grammar backend error: ") + e.what());
            return false;
        }
    }
    // GIL released here.

    // ------------------------------------------------------------------
    // Slow path: queue an async compile. If another in-flight entry is
    // already compiling the same key, subscribe to its shared_future
    // instead of submitting a duplicate task.
    // ------------------------------------------------------------------
    GrammarEntry entry;
    entry.stream            = stream;
    entry.key               = key;
    entry.require_reasoning = require_reasoning;
    entry.deadline          = std::chrono::steady_clock::now() + std::chrono::milliseconds(grammar_compile_timeout_ms_);

    bool   submitted_new_task  = false;
    size_t queue_size_after    = 0;
    size_t compile_tasks_after = 0;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (const auto& existing : grammar_queue_) {
            if (existing.key.key_type == key.key_type && existing.key.key_string == key.key_string
                && existing.future.valid()) {
                entry.future = existing.future;
                break;
            }
        }
        if (!entry.future.valid()) {
            auto promise = std::make_shared<std::promise<GrammarReadyPayload>>();
            entry.future = promise->get_future().share();

            CompileTask task;
            task.key               = key;
            task.require_reasoning = require_reasoning;
            task.promise           = std::move(promise);
            compile_tasks_.emplace_back(std::move(task));
            submitted_new_task = true;
        }
        grammar_queue_.emplace_back(std::move(entry));
        queue_size_after    = grammar_queue_.size();
        compile_tasks_after = compile_tasks_.size();
    }
    if (submitted_new_task) {
        worker_cv_.notify_one();
    }

    RTP_LLM_LOG_INFO("stream [%ld] grammar async compile %s: key=%s, queue_size=%zu, pending_tasks=%zu",
                     stream->streamId(),
                     submitted_new_task ? "queued" : "subscribed to in-flight",
                     keyBrief(key).c_str(),
                     queue_size_after,
                     compile_tasks_after);
    return true;
}

std::list<GenerateStreamPtr> GrammarManager::get_ready_grammar_requests() {
    // Split into two phases so we hold at most one lock at a time:
    //   Phase A (queue_mutex_, no GIL): determine ready / failed / pending
    //     entries, move ready entries into a local list, remove from queue.
    //   Phase B (GIL, no queue_mutex_): apply payloads — setGrammarObject,
    //     replay prefill tokens, update cache, setStop on failures.
    std::list<GrammarEntry> ready_entries;
    std::list<GrammarEntry> failed_entries;
    size_t                  queue_size_before = 0;
    size_t                  queue_size_after  = 0;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_size_before = grammar_queue_.size();
        if (grammar_queue_.empty()) {
            return {};
        }

        const auto now = std::chrono::steady_clock::now();
        for (auto it = grammar_queue_.begin(); it != grammar_queue_.end();) {
            auto& entry = *it;
            // Skip / drop entries whose streams are already dead.
            if (!entry.stream || !entry.stream->isActive()) {
                ready_entries.splice(ready_entries.end(), grammar_queue_, it++);
                continue;
            }
            // C++ future: non-blocking check.
            if (entry.future.valid() && entry.future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                ready_entries.splice(ready_entries.end(), grammar_queue_, it++);
                continue;
            }
            // Still pending — check wall-clock deadline.
            if (now >= entry.deadline) {
                auto waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     now - (entry.deadline - std::chrono::milliseconds(grammar_compile_timeout_ms_)))
                                     .count();
                RTP_LLM_LOG_WARNING("stream [%ld] grammar wait timeout: waited_ms=%lld, limit_ms=%lld",
                                    entry.stream->streamId(),
                                    static_cast<long long>(waited_ms),
                                    static_cast<long long>(grammar_compile_timeout_ms_));
                failed_entries.splice(failed_entries.end(), grammar_queue_, it++);
                continue;
            }
            ++it;
        }
        queue_size_after = grammar_queue_.size();
    }

    if (ready_entries.empty() && failed_entries.empty()) {
        return {};
    }

    RTP_LLM_LOG_INFO("grammar poll: ready=%zu, failed=%zu, queue=%zu->%zu",
                     ready_entries.size(),
                     failed_entries.size(),
                     queue_size_before,
                     queue_size_after);

    std::list<GenerateStreamPtr> return_reqs;

    // ---- Phase B: GIL-held apply loop. ----
    RTP_LLM_LOG_INFO(
        "grammar poll phase B: acquiring GIL (ready=%zu, failed=%zu)", ready_entries.size(), failed_entries.size());
    py::gil_scoped_acquire acquire;

    // With in-flight dedup, multiple entries in ready_entries may share the
    // same key (and the same underlying grammar_obj). Write each key's cache
    // at most once, and give every stream its own fresh copy of the grammar
    // so per-stream accept_token mutations don't bleed across streams.
    auto keyId = [](const GrammarKey& k) { return k.key_type + std::string("\x1f") + k.key_string; };
    std::unordered_set<std::string> cache_written;

    for (auto& entry : ready_entries) {
        if (!entry.stream) {
            continue;
        }
        return_reqs.emplace_back(entry.stream);

        // Stream already done before compile landed — no Python work needed.
        if (!entry.stream->isActive()) {
            continue;
        }

        // Pull payload from the future (should not block — we only put it
        // here because wait_for returned ready). If the future is invalid
        // (shouldn't happen) fall back to an error.
        GrammarReadyPayload payload;
        try {
            payload = entry.future.get();
        } catch (const std::exception& e) {
            entry.stream->reportError(ErrorCode::INVALID_PARAMS, std::string("grammar compile error: ") + e.what());
            continue;
        }

        const std::string kid = keyId(entry.key);

        if (payload.is_invalid || payload.grammar_obj.is_none()) {
            // Cache the invalid marker so repeat requests fail fast.
            if (!payload.grammar_obj.is_none() && cache_written.insert(kid).second) {
                try {
                    grammar_backend_.attr("set_cache")(grammarKeyToPyTuple(entry.key), payload.grammar_obj);
                } catch (const py::error_already_set& e) {
                    RTP_LLM_LOG_WARNING(
                        "stream [%ld] set_cache(invalid) failed: %s", entry.stream->streamId(), e.what());
                }
            }
            std::string err = payload.error_msg.empty() ? "unknown compile error" : payload.error_msg;
            entry.stream->reportError(ErrorCode::INVALID_PARAMS,
                                  "Failed to compile " + entry.key.key_type + " grammar: " + err);
            continue;
        }

        // Valid grammar — cache a *copy* (once per key) and give this stream
        // its own fresh copy.
        if (cache_written.insert(kid).second) {
            try {
                grammar_backend_.attr("set_cache")(grammarKeyToPyTuple(entry.key), payload.grammar_obj.attr("copy")());
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_WARNING("stream [%ld] set_cache failed: %s", entry.stream->streamId(), e.what());
            }
        }

        py::object stream_grammar;
        try {
            stream_grammar = payload.grammar_obj.attr("copy")();
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING("stream [%ld] grammar copy failed: %s", entry.stream->streamId(), e.what());
            entry.stream->reportError(ErrorCode::INVALID_PARAMS, std::string("grammar copy failed: ") + e.what());
            continue;
        }

        entry.stream->setGrammarObject(stream_grammar);
        replayPrefillTokensToGrammar(entry.stream, stream_grammar);
        RTP_LLM_LOG_INFO("stream [%ld] grammar ready -> waiting candidate: key=%s, grammar_type=%s",
                         entry.stream->streamId(),
                         keyBrief(entry.key).c_str(),
                         pyObjTypeName(stream_grammar).c_str());
    }

    for (auto& entry : failed_entries) {
        if (!entry.stream) {
            continue;
        }
        return_reqs.emplace_back(entry.stream);

        // Timeouts are back-pressure signals (queue was too long / worker too
        // busy), not a property of the schema itself. We deliberately do NOT
        // poison the memory cache with an InvalidGrammarObject here — doing
        // so would permanently lock out every subsequent request for the
        // same schema until process restart. Genuine compile failures are
        // cached via the valid-path writer above (payload.is_invalid with a
        // real InvalidGrammarObject from compile_now).
        //
        // A still-pending sibling entry in grammar_queue_ sharing the same
        // shared_future may still complete later; new requests for this key
        // will miss the cache, go through the slow path, and either
        // subscribe to that in-flight future or submit a fresh compile.
        entry.stream->reportError(ErrorCode::GENERATE_TIMEOUT, "Grammar preprocessing timed out");
        RTP_LLM_LOG_WARNING("stream [%ld] grammar timeout: key=%s (cache NOT poisoned; future requests may retry)",
                            entry.stream->streamId(),
                            keyBrief(entry.key).c_str());
    }

    // Destroy the two local lists while we still hold the GIL. Each entry
    // owns a shared_future whose shared state may be the last reference to
    // a GrammarReadyPayload containing a py::object; that py::object's
    // destructor needs GIL. Without this explicit clear, the lists would
    // be destroyed after `acquire` has released the GIL (reverse
    // declaration order), which aborts the interpreter.
    RTP_LLM_LOG_INFO("grammar poll phase B: clearing entry lists under GIL");
    ready_entries.clear();
    failed_entries.clear();
    RTP_LLM_LOG_INFO("grammar poll done: returning %zu streams", return_reqs.size());

    return return_reqs;
}

void GrammarManager::abort_requests(const GenerateStreamPtr& stream) {
    if (!stream) {
        return;
    }
    // Splice the matched entry (if any) into a local list so its shared_future
    // — which may own a GrammarReadyPayload with a py::object — is destroyed
    // under GIL, never under queue_mutex_.
    std::list<GrammarEntry> dropped;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto it = grammar_queue_.begin(); it != grammar_queue_.end(); ++it) {
            if (it->stream == stream) {
                dropped.splice(dropped.end(), grammar_queue_, it);
                break;
            }
        }
    }
    if (!dropped.empty()) {
        // reportError may reach into pybind and acquire GIL internally — do it
        // outside queue_mutex_ to keep the lock order clean.
        stream->reportError(ErrorCode::CANCELLED, "Aborted");
        RTP_LLM_LOG_INFO("abort_requests: stream [%ld] removed from grammar queue", stream->streamId());
        py::gil_scoped_acquire acquire;
        dropped.clear();
    } else {
        RTP_LLM_LOG_INFO("abort_requests: stream [%ld] not in grammar queue (no-op)", stream->streamId());
    }
}

void GrammarManager::cleanupStream(const GenerateStreamPtr& stream) {
    if (!stream) {
        return;
    }
    std::list<GrammarEntry> dropped;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto it = grammar_queue_.begin(); it != grammar_queue_.end(); ++it) {
            if (it->stream == stream) {
                dropped.splice(dropped.end(), grammar_queue_, it);
                break;
            }
        }
    }
    // grammar_obj_ is owned by the stream; clear it (acquires GIL inside).
    stream->clearGrammarObject();
    if (!dropped.empty()) {
        // GrammarEntry destruction may drop the last ref to the shared future's
        // payload (a py::object) — must happen under GIL.
        RTP_LLM_LOG_INFO("cleanupStream: stream [%ld] removed from grammar queue", stream->streamId());
        py::gil_scoped_acquire acquire;
        dropped.clear();
    }
}

void GrammarManager::workerLoop() {
    for (;;) {
        CompileTask task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            worker_cv_.wait(lock, [this] { return stop_ || !compile_tasks_.empty(); });
            if (stop_ && compile_tasks_.empty()) {
                return;
            }
            task = std::move(compile_tasks_.front());
            compile_tasks_.pop_front();
        }

        RTP_LLM_LOG_INFO("grammar worker picked up task: key=%s, require_reasoning=%d",
                         keyBrief(task.key).c_str(),
                         static_cast<int>(task.require_reasoning));

        const auto t_start = std::chrono::steady_clock::now();

        // Hold the GIL for the entire task lifecycle: compile, set_value,
        // and the final drop of `task.promise`. The caller may have already
        // dropped its shared_future (timeout path), in which case releasing
        // the last promise reference here destroys the shared state
        // synchronously on this worker thread. Without GIL the contained
        // py::object's destructor calls Py_XDECREF and aborts with
        // "PyThreadState_Get: GIL released".
        try {
            py::gil_scoped_acquire acquire;

            GrammarReadyPayload payload;
            try {
                py::object grammar = grammar_backend_.attr("compile_now")(
                    py::make_tuple(task.key.key_type, task.key.key_string), task.require_reasoning);
                payload.grammar_obj = grammar;
                payload.is_invalid  = isInvalidGrammar(grammar);
                if (payload.is_invalid) {
                    payload.error_msg = extractInvalidGrammarError(grammar);
                }
            } catch (const py::error_already_set& e) {
                payload.grammar_obj = py::none();
                payload.is_invalid  = true;
                payload.error_msg   = e.what();
            }

            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start)
                    .count();
            RTP_LLM_LOG_INFO("grammar worker compile_now done: key=%s, invalid=%d, elapsed_ms=%lld, err=%s",
                             keyBrief(task.key).c_str(),
                             static_cast<int>(payload.is_invalid),
                             static_cast<long long>(elapsed_ms),
                             payload.error_msg.empty() ? "" : payload.error_msg.c_str());

            try {
                task.promise->set_value(std::move(payload));
                RTP_LLM_LOG_INFO("grammar worker promise set_value: key=%s", keyBrief(task.key).c_str());
            } catch (const std::future_error& fe) {
                // Promise already satisfied or moved-from — ignore.
                RTP_LLM_LOG_WARNING(
                    "grammar worker set_value future_error: key=%s, what=%s", keyBrief(task.key).c_str(), fe.what());
            }

            // Drop the promise under GIL. If the caller already dropped the
            // future on timeout, the shared state — and the py::object it
            // owns — is freed synchronously here, safely under GIL.
            task.promise.reset();
        } catch (const std::exception& e) {
            // Should only fire if GIL acquire itself throws during shutdown.
            RTP_LLM_LOG_WARNING("grammar worker setup error: key=%s, what=%s",
                                keyBrief(task.key).c_str(),
                                e.what());
        }
    }
}

}  // namespace rtp_llm
