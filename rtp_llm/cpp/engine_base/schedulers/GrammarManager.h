#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

// Plain C++ representation of a grammar compile request key. Kept purely C++
// so it can be stored / compared / logged without holding GIL.
struct GrammarKey {
    std::string key_type;    // one of: json, regex, ebnf, structural_tag
    std::string key_string;  // schema / pattern / ebnf / structural_tag JSON
    bool        empty() const {
        return key_type.empty();
    }
};

// Result produced by a C++ worker after invoking the Python grammar backend.
// `grammar_obj` is a freshly compiled BaseGrammarObject (may be an
// InvalidGrammarObject — in which case `is_invalid` is true). `error_msg`
// carries the Python exception message when the worker itself threw.
struct GrammarReadyPayload {
    py::object  grammar_obj;         // py::none() if compile threw
    bool        is_invalid = false;  // true for InvalidGrammarObject or exception
    std::string error_msg;           // for logging / setStop reason
};

// GrammarManager schedules grammar compilation requests on a pool of C++
// worker threads. Each worker synchronously calls the Python backend under
// GIL. Public methods below are thread-safe (internally guarded by
// `queue_mutex_`); the contract is that the Python GIL is NEVER held by the
// caller while any public method is waiting on `queue_mutex_`.
class GrammarManager {
public:
    // Default-construct grammar_backend = empty py::object() so callers
    // (mainly cc_test ctors) that have no Python interpreter can omit it
    // entirely. py::object() with m_ptr=nullptr does NOT touch Python; only
    // a real backend (or py::none()) would. hasBackend() guards the rest.
    explicit GrammarManager(py::object grammar_backend  = py::object(),
                            int        num_workers      = 2,
                            int64_t    compile_timeout_ms = 60000);
    ~GrammarManager();

    size_t size() const;
    void   clear();
    bool   has_waiting_grammars() const;

    // Entry path from the scheduler. Returns true iff the stream was queued
    // for async compile (caller should NOT enqueue into waiting_streams_ yet).
    bool process_req_with_grammar(const GenerateStreamPtr& stream);

    // Called from the scheduler thread each tick. Reaps any finished compile
    // futures, writes `grammar_obj_` onto the corresponding stream, and
    // returns the streams that can now proceed to waiting_streams_.
    std::list<GenerateStreamPtr> get_ready_grammar_requests();

    // Mark an outstanding grammar request as aborted. Removes it from the
    // queue and calls setStop on the stream. Any in-flight compile result is
    // discarded when it eventually lands.
    void abort_requests(const GenerateStreamPtr& stream);

    // Remove a stream from the queue (if present) and drop grammar state on
    // the stream. Called when a stream finishes / is stopped.
    void cleanupStream(const GenerateStreamPtr& stream);

private:
    struct GrammarEntry {
        GenerateStreamPtr                       stream;
        GrammarKey                              key;
        bool                                    require_reasoning = false;
        std::shared_future<GrammarReadyPayload> future;
        std::chrono::steady_clock::time_point   deadline;
    };

    struct CompileTask {
        GrammarKey                                         key;
        bool                                               require_reasoning;
        std::shared_ptr<std::promise<GrammarReadyPayload>> promise;
    };

    // Worker thread body.
    void workerLoop();

    // Helpers. `finalizeReadyStream` / `replayPrefillTokensToGrammar` /
    // `isInvalidGrammar` require the caller to hold GIL.
    bool        isGrammarRequested(const GenerateStreamPtr& stream) const;
    GrammarKey  extractGrammarKey(const GenerateStreamPtr& stream) const;
    py::tuple   grammarKeyToPyTuple(const GrammarKey& key) const;         // requires GIL
    bool        isInvalidGrammar(const py::object& obj) const;            // requires GIL
    std::string extractInvalidGrammarError(const py::object& obj) const;  // requires GIL
    void        replayPrefillTokensToGrammar(const GenerateStreamPtr& stream,
                                             py::object&              grammar_obj);  // requires GIL

    // True iff grammar_backend_ holds a real (non-null, non-None) Python
    // object. Tests construct with the default empty py::object() and never
    // need a Python interpreter.
    bool hasBackend() const {
        return static_cast<bool>(grammar_backend_) && !grammar_backend_.is_none();
    }

    // Python-side backend. `grammar_backend_` is accessed only under GIL.
    // Default-constructed = empty (m_ptr == nullptr), no GIL needed for ctor
    // or dtor. invalid_grammar_cls_ likewise — drop the `= py::none()` form
    // because that initializer would call into Python at member init time.
    py::object grammar_backend_;
    py::object invalid_grammar_cls_;

    // Queue of in-flight grammar entries. Protected by queue_mutex_ for all
    // read and write access. Entries are ordered by arrival (FIFO-ish).
    mutable std::mutex      queue_mutex_;
    std::condition_variable worker_cv_;
    std::list<GrammarEntry> grammar_queue_;
    std::deque<CompileTask> compile_tasks_;

    // Worker pool. `workers_` is immutable after ctor; `stop_` is signaled in
    // the dtor, which then joins all workers.
    std::vector<std::thread> workers_;
    std::atomic<bool>        stop_{false};

    // Config (read-only after ctor). Per-entry wall-clock deadline for an
    // outstanding compile, measured from when the entry was enqueued.
    int64_t grammar_compile_timeout_ms_ = 60000;
};

}  // namespace rtp_llm
