// Unit tests for GrammarManager. The manager owns the C++ worker pool that
// drives the Python grammar backend (xgrammar / reasoner / …) under GIL.
// These tests use a pure-Python fake backend defined inline via py::exec so
// we can dictate:
//   * cache hit vs miss
//   * synchronous compile failure (InvalidGrammarObject)
//   * slow compile → timeout path
// without pulling in xgrammar or a real tokenizer.
//
// GenerateStream instances are real NormalGenerateStreams since the manager
// reaches into several stream APIs (setGrammarObject / reportError /
// generateConfig / ...). We fake-init KV blocks just enough for the stream
// to report `isActive() == true`.

#include <chrono>
#include <memory>
#include <thread>

#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/engine_base/schedulers/GrammarManager.h"
#undef private
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace py = pybind11;
using namespace std;

namespace rtp_llm {

class GrammarManagerTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        // This test exercises Python paths (py::exec / py::module_::import for
        // the fake backend), so we explicitly opt in to the embedded
        // interpreter. Most other cc_tests don't touch Python.
        rtp_llm::test_helpers::ensurePythonInterpreterStarted();
    }

    // Build a fresh stream with the given grammar request. `json_schema` is
    // set to drive isGrammarRequested=true; that's all the manager inspects.
    GenerateStreamPtr createGrammarStream(const ModelConfig&     model_config,
                                          const RuntimeConfig&   runtime_config,
                                          const ResourceContext& resource_context,
                                          const std::string&     json_schema,
                                          int                    block_id = 1) {
        std::shared_ptr<GenerateInput> query = std::make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(std::vector<int32_t>{1, 2, 3}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        query->generate_config->json_schema = json_schema;

        GenerateStreamPtr stream =
            std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

        // Minimal KV cache so the stream looks active.
        BatchKVCacheResource addr;
        addr.resetBatchSize(1);
        addr.initGroups(1, 1, {0});
        addr.setBatchBlocks(0, 0, {block_id});
        stream->setKVCache(addr);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setNeedReleaseResource(false);
        return stream;
    }

    // Load the fake backend into a py::module_ that looks-and-smells like
    // BaseGrammarBackend. Each test passes a different mode via knobs set on
    // the module: _mode in {"cache_hit", "cache_miss", "slow_compile",
    // "compile_fail"}, _delay_s (float), _compile_calls (int counter).
    py::object makeFakeBackend(const std::string& mode, double delay_s = 0.0) {
        py::object builtins = py::module_::import("builtins");
        py::dict   ns;
        py::exec(R"py(
import time

class _InvalidGrammarObject:
    def __init__(self, msg="fake_invalid"):
        self.error_message = msg

class _FakeGrammar:
    """Minimal grammar object the manager can accept/copy/cache."""
    def __init__(self, key):
        self.key = key
    def copy(self):
        return _FakeGrammar(self.key)
    def accept_token(self, token_id):
        pass
    def maybe_init_reasoning(self, reasoning):
        pass

class _FakeBackend:
    _invalid_grammar_cls = _InvalidGrammarObject
    def __init__(self, mode, delay_s):
        self.mode = mode
        self.delay_s = float(delay_s)
        self.cache = {}
        self.compile_calls = 0
        self.reset_calls = 0
    def get_cached(self, key, require_reasoning):
        return self.cache.get(tuple(key))
    def compile_now(self, key, require_reasoning):
        self.compile_calls += 1
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.mode == "compile_fail":
            return _InvalidGrammarObject("schema broken")
        return _FakeGrammar(tuple(key))
    def set_cache(self, key, value):
        self.cache[tuple(key)] = value
    def reset(self):
        self.reset_calls += 1
        self.cache.clear()
)py", ns, ns);

        py::object cls     = ns["_FakeBackend"];
        py::object backend = cls(mode, delay_s);
        return backend;
    }

    // Minimal ModelConfig / RuntimeConfig / ResourceContext needed to stamp
    // a NormalGenerateStream. Shared across tests for brevity.
    void buildContext(ModelConfig& model_config, RuntimeConfig& runtime_config, ResourceContext& rc) {
        model_config.max_seq_len = 128;
        // Leave others at defaults; the manager does not read them.
        (void)rc;
        (void)runtime_config;
    }
};

TEST_F(GrammarManagerTest, BypassWhenNoGrammar) {
    py::object backend = makeFakeBackend("cache_miss");
    GrammarManager mgr(backend, /*num_workers=*/1, /*compile_timeout_ms=*/5000);

    ModelConfig model_config;
    RuntimeConfig runtime_config;
    ResourceContext rc;
    buildContext(model_config, runtime_config, rc);

    // Stream WITHOUT json_schema / regex / etc. — should take the bypass
    // branch, return false, and leave the queue untouched.
    std::shared_ptr<GenerateInput> query = std::make_shared<GenerateInput>();
    query->input_ids       = torch::tensor(std::vector<int32_t>{1, 2}, torch::kInt32);
    query->generate_config = std::make_shared<GenerateConfig>();
    GenerateStreamPtr stream =
        std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, rc, nullptr);

    bool queued = mgr.process_req_with_grammar(stream);
    EXPECT_FALSE(queued);
    EXPECT_EQ(mgr.size(), 0u);
}

TEST_F(GrammarManagerTest, CacheHitSyncPath) {
    py::object backend = makeFakeBackend("cache_miss");
    // Pre-populate the fake cache so the first call is a hit.
    {
        py::gil_scoped_acquire acquire;
        py::object fake_grammar = backend.attr("compile_now")(
            py::make_tuple("json", "{}"), false);
        backend.attr("cache")[py::make_tuple("json", "{}")] = fake_grammar;
    }

    GrammarManager mgr(backend, 1, 5000);

    ModelConfig mc; RuntimeConfig rtc; ResourceContext rc;
    buildContext(mc, rtc, rc);
    auto stream = createGrammarStream(mc, rtc, rc, "{}");

    bool queued = mgr.process_req_with_grammar(stream);
    EXPECT_FALSE(queued) << "cache hit must take the sync path (no queue)";
    EXPECT_EQ(mgr.size(), 0u);

    py::gil_scoped_acquire acquire;
    py::object g = stream->tryGetGrammarObject();
    EXPECT_FALSE(g.is_none()) << "cache hit must place grammar on the stream";
}

TEST_F(GrammarManagerTest, CacheMissQueuesAndWorkerCompletes) {
    py::object backend = makeFakeBackend("cache_miss");
    GrammarManager mgr(backend, 1, 5000);

    ModelConfig mc; RuntimeConfig rtc; ResourceContext rc;
    buildContext(mc, rtc, rc);
    auto stream = createGrammarStream(mc, rtc, rc, "{\"type\":\"object\"}");

    bool queued = mgr.process_req_with_grammar(stream);
    EXPECT_TRUE(queued);
    EXPECT_EQ(mgr.size(), 1u);

    // Spin-wait on get_ready_grammar_requests for up to 2s.
    // Release GIL during the wait so the GrammarManager worker can acquire
    // it for compile_now; otherwise main holds GIL forever and worker hangs.
    std::list<GenerateStreamPtr> ready;
    {
        py::gil_scoped_release release;
        for (int i = 0; i < 200 && ready.empty(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ready = mgr.get_ready_grammar_requests();
        }
    }
    ASSERT_EQ(ready.size(), 1u) << "worker failed to complete compile within 2s";
    EXPECT_EQ(ready.front(), stream);
    EXPECT_EQ(mgr.size(), 0u);

    py::gil_scoped_acquire acquire;
    EXPECT_FALSE(stream->tryGetGrammarObject().is_none());
}

TEST_F(GrammarManagerTest, InFlightDedupShareCompile) {
    py::object backend = makeFakeBackend("cache_miss", /*delay_s=*/0.3);
    GrammarManager mgr(backend, 1, 5000);

    ModelConfig mc; RuntimeConfig rtc; ResourceContext rc;
    buildContext(mc, rtc, rc);
    auto s1 = createGrammarStream(mc, rtc, rc, "{\"k\":1}", /*block_id=*/1);
    auto s2 = createGrammarStream(mc, rtc, rc, "{\"k\":1}", /*block_id=*/2);

    EXPECT_TRUE(mgr.process_req_with_grammar(s1));
    EXPECT_TRUE(mgr.process_req_with_grammar(s2));
    EXPECT_EQ(mgr.size(), 2u);

    // Wait for both to go ready. Release GIL so the worker can compile.
    size_t total_ready = 0;
    {
        py::gil_scoped_release release;
        for (int i = 0; i < 300 && total_ready < 2; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto ready = mgr.get_ready_grammar_requests();
            total_ready += ready.size();
        }
    }
    EXPECT_EQ(total_ready, 2u);

    // compile_now should have fired ONCE — the second stream subscribes to
    // the in-flight future.
    py::gil_scoped_acquire acquire;
    int calls = backend.attr("compile_calls").cast<int>();
    EXPECT_EQ(calls, 1) << "in-flight dedup failed: compile ran " << calls << " times";
}

TEST_F(GrammarManagerTest, InvalidGrammarReportsError) {
    py::object backend = makeFakeBackend("compile_fail");
    GrammarManager mgr(backend, 1, 5000);

    ModelConfig mc; RuntimeConfig rtc; ResourceContext rc;
    buildContext(mc, rtc, rc);
    auto stream = createGrammarStream(mc, rtc, rc, "{\"bad\":\"schema\"}");

    EXPECT_TRUE(mgr.process_req_with_grammar(stream));

    std::list<GenerateStreamPtr> ready;
    {
        py::gil_scoped_release release;
        for (int i = 0; i < 200 && ready.empty(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ready = mgr.get_ready_grammar_requests();
        }
    }
    ASSERT_EQ(ready.size(), 1u);
    // Stream must have an error status; isActive() returns false.
    EXPECT_FALSE(stream->isActive()) << "stream should be in error state after invalid grammar";
}

TEST_F(GrammarManagerTest, TimeoutFailsStream) {
    // Compile takes 500ms; give the queue a 50ms timeout.
    py::object backend = makeFakeBackend("cache_miss", /*delay_s=*/0.5);
    GrammarManager mgr(backend, 1, /*compile_timeout_ms=*/50);

    ModelConfig mc; RuntimeConfig rtc; ResourceContext rc;
    buildContext(mc, rtc, rc);
    auto stream = createGrammarStream(mc, rtc, rc, "{\"slow\":true}");

    EXPECT_TRUE(mgr.process_req_with_grammar(stream));

    // Poll until the manager reports the stream as failed. Must observe
    // failure within a few hundred ms — not wait for full compile.
    std::list<GenerateStreamPtr> ready;
    {
        py::gil_scoped_release release;
        for (int i = 0; i < 30; ++i) {  // ~300ms
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto cur = mgr.get_ready_grammar_requests();
            ready.splice(ready.end(), cur);
            if (!ready.empty()) break;
        }
    }
    ASSERT_EQ(ready.size(), 1u);
    EXPECT_FALSE(stream->isActive());

    // The cache MUST NOT have been poisoned: timeout != invalid-schema.
    py::gil_scoped_acquire acquire;
    size_t cache_len = py::len(backend.attr("cache"));
    EXPECT_EQ(cache_len, 0u) << "timeout poisoned the cache";
}

TEST_F(GrammarManagerTest, AbortRemovesFromQueue) {
    py::object backend = makeFakeBackend("cache_miss", /*delay_s=*/1.0);
    GrammarManager mgr(backend, 1, 10000);

    ModelConfig mc; RuntimeConfig rtc; ResourceContext rc;
    buildContext(mc, rtc, rc);
    auto stream = createGrammarStream(mc, rtc, rc, "{\"x\":1}");

    EXPECT_TRUE(mgr.process_req_with_grammar(stream));
    EXPECT_EQ(mgr.size(), 1u);

    mgr.abort_requests(stream);
    EXPECT_EQ(mgr.size(), 0u);
    EXPECT_FALSE(stream->isActive());  // Aborted sets error state
}

TEST_F(GrammarManagerTest, CleanupStreamRemovesFromQueue) {
    py::object backend = makeFakeBackend("cache_miss", /*delay_s=*/1.0);
    GrammarManager mgr(backend, 1, 10000);

    ModelConfig mc; RuntimeConfig rtc; ResourceContext rc;
    buildContext(mc, rtc, rc);
    auto stream = createGrammarStream(mc, rtc, rc, "{\"y\":2}");

    EXPECT_TRUE(mgr.process_req_with_grammar(stream));
    EXPECT_EQ(mgr.size(), 1u);

    mgr.cleanupStream(stream);
    EXPECT_EQ(mgr.size(), 0u);

    py::gil_scoped_acquire acquire;
    // After cleanup, tryGetGrammarObject() may return either an empty py::object()
    // (m_ptr=nullptr) or py::none() — both mean "no grammar" for our purposes.
    py::object g = stream->tryGetGrammarObject();
    EXPECT_TRUE(!static_cast<bool>(g) || g.is_none())
        << "cleanupStream should clear the stream's grammar object";
}

TEST_F(GrammarManagerTest, ClearDrainsAndResetsBackend) {
    py::object backend = makeFakeBackend("cache_miss", /*delay_s=*/1.0);
    GrammarManager mgr(backend, 1, 10000);

    ModelConfig mc; RuntimeConfig rtc; ResourceContext rc;
    buildContext(mc, rtc, rc);
    auto s1 = createGrammarStream(mc, rtc, rc, "{\"a\":1}", 1);
    auto s2 = createGrammarStream(mc, rtc, rc, "{\"b\":2}", 2);
    EXPECT_TRUE(mgr.process_req_with_grammar(s1));
    EXPECT_TRUE(mgr.process_req_with_grammar(s2));
    EXPECT_EQ(mgr.size(), 2u);

    mgr.clear();
    EXPECT_EQ(mgr.size(), 0u);

    py::gil_scoped_acquire acquire;
    int reset_calls = backend.attr("reset_calls").cast<int>();
    EXPECT_EQ(reset_calls, 1) << "clear() must call backend.reset()";
}

}  // namespace rtp_llm
