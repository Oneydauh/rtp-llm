# 投机解码 + xgrammar 约束解码

## 概述

RTP-LLM 支持语法约束生成（json_schema、regex、ebnf、structural_tag）与 MTP（Multi-Token Prediction）投机解码的组合使用。本文档描述了在投机验证阶段如何应用语法约束。

设计方案参考 sglang 的成熟做法：
- 草稿模型（draft model）**不受约束**运行 —— 不对 draft token 做语法掩码
- 通过 **DFS accept/rollback** 在目标模型 logits 上生成语法 bitmask
- 只有**验证通过的 token** 才会永久性地 accept 到语法状态机中

## 架构

### 普通解码（每步单 token）

```
gatherSamplerInput → applyGrammarConstraints → sampler.forward → dispatch → batchAcceptGrammarTokensAsync
```

语法状态流程：`fill_bitmask` → 应用到 logits → 采样 → `accept_token`。

### 投机解码（每步 K+1 个 token）

```
Prefill:
  gatherSamplerInput → applyGrammarConstraints → sampler.forward → dispatch

Decode:
  draftModelDecode（不受约束）→ targetVerifyForward →
  gatherSpecSamplerInput → applySpecGrammarConstraints → sampler.forward →
  rejectionSampling → dispatch → batchAcceptSpecGrammarTokensAsync
```

## DFS Bitmask 生成算法

给定一个流的语法状态 S（包含所有已 accept 的 token）和 K 个 draft token `[d1, d2, ..., dK]`，目标模型产出 K+1 行 logits。语法 bitmask 通过 DFS 方式生成：

```
state = S（当前语法状态）

位置 0: fill_bitmask(state) → 应用到 logits[0]    # 约束上一个已 accept token 之后的预测
accept(d1) → 状态前进

位置 1: fill_bitmask(state) → 应用到 logits[1]    # 约束 d1 之后的预测
accept(d2) → 状态前进

...

位置 K-1: fill_bitmask(state) → 应用到 logits[K-1]  # 约束 d(K-1) 之后的预测
accept(dK) → 状态前进

位置 K: fill_bitmask(state) → 应用到 logits[K]      # 约束 bonus token 的预测

rollback(K) → 状态恢复到 S
```

关键特性：
- 语法状态在 DFS 过程中**临时前进**，最后完全回滚
- 每行 logit 获得其在序列中对应位置的正确 bitmask
- `rollback(K)` 使用 xgrammar 的 `GrammarMatcher.rollback()`，时间复杂度 O(K)

## 拒绝采样后的永久 Accept

拒绝采样确定每个流 accept 的 token 数量（`accept_len`）后：
1. 对每个有语法约束的流，逐个调用 `accept_token()` accept 验证通过的 token
2. 永久性地推进语法状态
3. 如果流已结束，调用 `recycle()` 将 matcher 归还到对象池
4. Accept 操作**异步执行**（与普通解码模式相同），因为结果在下一步之前不需要

## 配置

### MAX_ROLLBACK_TOKENS

xgrammar 的 `GrammarMatcher` 创建时设置 `max_rollback_tokens=200`（定义在 `xgrammer_backend.py` 中）。该值必须 >= `propose_step`（通常为 1-8）。默认值 200 足够使用。

### 环境变量

- `SGLANG_GRAMMAR_POLL_INTERVAL`：语法编译轮询间隔（默认：0.005s）
- `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS`：语法编译最大轮询次数（默认：10000）
- `GRAMMAR_CACHE_DIR`：跨 DP 共享的语法文件缓存目录

## 文件布局

| 文件 | 职责 |
|------|------|
| `grammar_batch_ops.py` | `batch_apply_spec_grammar_constraints()` —— DFS bitmask 生成；`batch_accept_spec_tokens()` —— 多 token accept |
| `MtpBatchStreamProcessor.h/cc` | `applySpecGrammarConstraints()` —— DFS 的 C++ 入口；`batchAcceptSpecGrammarTokensAsync()` —— 异步永久 accept |
| `MtpExecutor.cc` | `prefillStep()` 和 `decodeStep()` 中的集成点 |
| `xgrammer_backend.py` | `XGrammarGrammar.accept_token()`、`rollback()`、`fill_vocab_mask()` —— 单语法状态机操作 |

## Draft Token ID 布局

`propose_step == 1` 时：
- `sp_output_buffer->tokens = [上一个已 accept 的目标 token, draft token]`
- DFS 使用 `draft token`（第 1 列）

`propose_step > 1` 时：
- `draft_token_ids_t` 形状：`[batch_size, propose_step+1]`
- 第 0 列：上一个已 accept 的目标 token（跳过）
- 第 1..propose_step 列：用于 DFS 的 draft token

## 与 PD 分离的交互

启用 PD 分离（prefill 在一个节点，decode 在另一个节点）时：
1. Prefill 节点生成第一个 token 并 accept 到语法中
2. Decode 节点编译新的语法，通过 `replayPrefillTokensToGrammar()` 重放 prefill 生成的 token
3. 重放后，语法状态完成同步
4. 投机解码正常进行 DFS bitmask 生成

## 与 ReasonerGrammarBackend（思考模式）的交互

`ReasonerGrammarBackend` 包装了基础语法，管理 think/answer 模式转换。它支持带有模式感知状态跟踪的 `rollback()` 和 `accept_token()`。DFS 模式可以透明地通过该包装层工作。
