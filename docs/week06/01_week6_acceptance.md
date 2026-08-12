# Week 6：Auditable Agent Runtime 验收报告

## 1. 本周目标

Week 6 的核心目标是建立一套：

- 可执行；
- 可限制；
- 可恢复；
- 可观测；
- 可追踪；
- 可回放；
- 可审计；

的企业可信文档 Agent Runtime。

本周重点不再是继续改进检索模型，而是构建 Agent 的控制平面，使系统能够明确回答：

1. 当前 Agent 正在执行什么？
2. 为什么选择这个 Tool？
3. Tool 接收了什么参数？
4. Tool 执行是否成功？
5. 失败后是否进行了重试？
6. Agent 中断以后从哪里恢复？
7. 数值结论来自哪个结构化 Fact 或 Calculation？
8. 最终答案能否沿着 Trajectory 回放？

---

## 2. Week 6 最终架构

当前 Runtime 主链路：

```text
User Query
    ↓
RuntimeQueryParser
    ↓
RuntimeIntentRouter
    ↓
RuntimePlanner
    ↓
RuntimePlan
    ↓
AgentRuntime
    ↓
RuntimePlanExecutor
    ├── query_financial_data
    ├── retrieve_documents
    └── execute_calculation
    ↓
RuntimeEvidenceVerifier
    ↓
RuntimeAnswerGenerator
    ↓
Final Answer
```

同时维护：

```text
AgentState
Checkpoint
NodeSpan
ToolCallTrace
RetrievalTrace
CalculationTrace
AgentTrajectory
```

LangGraph 作为 AgentRuntime 的 orchestration adapter：

```text
LangGraph
    ↓
AgentRuntime
    ↓
Business Services
```

业务逻辑本身不直接绑定 LangGraph。

---

## 3. Structured AgentState

Week 6 建立统一 `AgentState`。

AgentState 保存完整执行上下文，包括：

### Query Understanding

```text
query
parsed_query
company_ids
report_ids
years
metric_ids
intent
confidence
```

### Planning

```text
runtime_plan
planner_version
current_step
completed_step_ids
```

### Runtime Results

```text
runtime_refs
tool_execution_results
retrieved_documents
resolved_fact_ids
evidence_ids
calculation_ids
```

### Observability

```text
node_spans
tool_call_traces
retrieval_traces
calculation_traces
errors
```

### Control State

```text
status
stop_reason
retry_count
step_count
max_steps
current_node
next_node
checkpoint_revision
```

### Human Review

```text
pending_human_review
human_review_reason
human_decision
```

这使得 Agent 的运行状态可以：

```text
Serialize
↓
Checkpoint
↓
Recover
↓
Replay
```

---

## 4. Query Parser

`RuntimeQueryParser` 负责把自然语言问题解析成结构化查询。

主要识别：

- Company；
- Report；
- Fiscal Year；
- Metric；
- Derived Metric；
- Statement Scope；
- Comparison；
- Ranking；
- Explanation；
- Missing Fields。

例如：

```text
美的集团2024年营业收入是多少？
```

解析为：

```text
company = midea_group
year = 2024
metric = revenue
report = midea_group_2024
```

Parser 不负责：

- 查询 FinancialFact；
- 检索 Chunk；
- 执行计算；
- 生成答案。

---

## 5. Intent Router

`RuntimeIntentRouter` 将 Parsed Query 路由到以下 Intent：

```text
financial_fact
financial_calculation
financial_comparison
document_evidence
unsupported
```

例如：

```text
营业收入是多少？
→ financial_fact
```

```text
毛利率是多少？
→ financial_calculation
```

```text
比较2024年和2025年营业收入
→ financial_comparison
```

```text
为什么营业收入增长？
→ document_evidence
```

Router 不会因为字段缺失就把问题判定为 unsupported。

例如：

```text
营业收入是多少？
```

仍然属于：

```text
financial_fact
```

只是后续进入：

```text
awaiting_human
```

---

## 6. Runtime Planner

Planner 根据：

```text
ParsedFinancialQuery
+
AgentIntent
```

构造确定性的 RuntimePlan。

支持的 Plan Action：

```text
retrieve
calculate
compare
rank
synthesize
```

例如：

```text
美的集团2024年毛利率是多少？
```

会被规划为：

```text
s1 query revenue
s2 query operating_cost
s3 execute gross_profit_margin calculation
```

即：

```text
Revenue
      +
Operating Cost
      ↓
Gross Profit Margin
```

Planner 只规划：

> 需要获取什么数据以及以什么顺序执行。

真正的 Fact ID 必须在 Runtime 执行过程中得到。

---

## 7. Tool Registry

Week 6 建立统一 Tool Registry。

每个 Tool Definition 包含：

```text
tool_name
description
version
input_schema
output_schema
permission
timeout_seconds
max_retries
idempotent
max_result_bytes
```

因此 Agent 不直接调用任意 Python Function，而必须经过统一 Tool Contract。

---

## 8. query_financial_data

作用：

> 查询已经核验的结构化 FinancialFact。

执行路径：

```text
RuntimePlan
↓
query_financial_data
↓
FinancialFactRegistry
↓
Verified FinancialFact
↓
Primary SourceEvidence
```

数值事实问题不允许模型自由生成数值。

例如：

```text
美的集团2024年营业收入是多少？
```

答案中的数值必须来自：

```text
FinancialFact.normalized_value
```

而不是语言模型根据上下文猜测。

---

## 9. retrieve_documents

作用：

> 查询年报中的叙述性证据。

适用于：

- 风险；
- 战略；
- 管理层说明；
- 市场情况；
- 经营情况；
- 业务布局；
- 原因归因；
- 未来展望。

生产检索链路：

```text
Dense Retrieval
      +
BM25
      ↓
RRF
      ↓
Cross-Encoder Reranker
      ↓
RetrievedDocument
```

Tool 只负责提供标准输入输出 Contract，不重新实现底层检索算法。

---

## 10. execute_calculation

作用：

> 使用经过验证的 FinancialFact 执行确定性财务计算。

例如毛利率：

```text
Revenue Fact
+
Operating Cost Fact
        ↓
Deterministic Calculator
        ↓
Gross Profit Margin
```

Calculation Trace 保存：

```text
calculation_id
metric_id
formula_id
input_fact_ids
result_value
result_unit
latency
status
```

当前支持的确定性公式包括：

- Gross Profit Margin；
- Selling + R&D Expense Ratio；
- Operating Cash Flow / Net Profit；
- Current Ratio；
- Debt-to-Equity Ratio；
- Effective Income Tax Rate。

---

## 11. RuntimePlanExecutor

`RuntimePlanExecutor` 根据 RuntimePlan 顺序执行 Step。

例如：

```text
s1 retrieve
↓
s2 retrieve
↓
s3 calculate
↓
s4 compare
```

每执行一步：

1. 解析 Runtime References；
2. 构造 Tool Arguments；
3. 通过 ToolExecutor 调用；
4. 记录 ToolCallTrace；
5. 更新 AgentState；
6. 推进 current_step。

同时支持内部确定性操作：

```text
compare
rank
synthesize
```

这些不需要额外 External Tool。

---

## 12. Tool Permission

Tool Definition 中声明 permission。

当前权限包括：

```text
read_financial_data
read_documents
execute_calculation
```

ToolExecutor 调用前检查：

```text
required permission
⊆
granted permissions
```

权限不足则拒绝 Tool Execution。

---

## 13. Tool Timeout 与 Retry

ToolExecutor 统一处理：

```text
timeout
retry
retryable error
permanent error
```

并生成对应 ToolCallTrace。

Retry 不会修改业务 Tool 的实现。

Tool 只需要明确声明：

```text
max_retries
idempotent
```

其中：

> 只有声明为 idempotent 的 Tool 才允许自动 Retry。

---

## 14. Tool Idempotency

ToolExecutor 使用稳定 Idempotency Key。

其核心输入包括：

```text
run_id
step_id
tool_name
arguments
```

因此同一个 Agent Run 中：

```text
same run
+
same step
+
same tool
+
same arguments
```

会得到相同 Idempotency Key。

如果 Tool Result 已存在：

```text
status = reused
```

而不是重复执行。

---

## 15. CheckpointStore

Week 6 实现：

```text
InMemoryCheckpointStore
SQLiteCheckpointStore
```

Checkpoint 采用：

```text
append-only revisions
```

例如：

```text
revision 1
revision 2
revision 3
...
```

不会直接覆盖旧 State。

同时支持：

```text
expected_revision
```

用于乐观并发控制。

---

## 16. Runtime Recovery

AgentRuntime 提供：

```python
runtime.run(...)
```

用于创建新的 Run。

同时提供：

```python
runtime.resume(
    run_id=...,
    thread_id=...,
)
```

用于从最新 Checkpoint 恢复。

恢复依据：

```text
status
next_node
current_step
completed_step_ids
runtime_refs
```

例如：

```text
s1 completed
↓
checkpoint current_step = 1
↓
process crash
↓
resume
↓
continue from s2
```

已经成功写入 Checkpoint 的步骤不会重新执行。

---

## 17. Checkpoint 与 Idempotency 的区别

Checkpoint 解决：

> Agent 从哪里继续？

Idempotency 解决：

> 已经执行成功但尚未来得及写 Checkpoint 的 Tool，恢复后被再次请求怎么办？

例如：

```text
s1 Tool succeeded
↓
process crash
↓
checkpoint still before s1
↓
resume
↓
Runtime requests s1 again
```

此时：

```text
same run_id
same step_id
same tool
same arguments
```

生成相同 Idempotency Key。

如果缓存结果存在：

```text
status = reused
```

因此二者共同构成可靠恢复机制。

---

## 18. SQLite 跨实例恢复

Recovery Smoke 中验证了：

```text
Runtime Instance A
      ↓
SQLiteCheckpointStore
      ↓
execute part of plan
      ↓
process exits
```

重新创建：

```text
Runtime Instance B
      ↓
same SQLite database
      ↓
resume()
```

可以继续之前尚未完成的 Plan。

这意味着恢复能力不依赖单个 Python Runtime 实例。

---

## 19. LangGraph

Week 6 在稳定的 Framework-independent Runtime 上增加 LangGraph。

LangGraph Graph 包含：

```text
parse_query
route_intent
create_plan
execute_plan
verify_evidence
generate_answer
await_human
finish
```

主要负责：

- StateGraph；
- Edge；
- Conditional Edge；
- Checkpointer；
- thread_id；
- interrupt；
- resume。

核心业务仍由既有 Runtime Service 提供。

---

## 20. Human Interrupt

如果 Query 缺少执行所需关键字段，例如：

```text
营业收入是多少？
```

缺少：

```text
company
year
```

则：

```text
Runtime
↓
awaiting_human
↓
LangGraph interrupt
```

用户提供修正后的 Query：

```text
美的集团2024年营业收入是多少？
```

然后使用相同 `thread_id`：

```text
Command(resume=...)
```

继续执行。

---

## 21. Human Rejection

如果用户拒绝补充信息：

```text
approved = false
```

则 Runtime 终止：

```text
status = refused
stop_reason = human_rejected
```

而不是无限等待。

---

## 22. Max Steps

Runtime 使用：

```text
step_count
max_steps
```

限制 Agent 最大执行步数。

超过限制：

```text
status = failed
stop_reason = max_steps_exceeded
```

从机制上避免无限 Agent Loop。

---

## 23. Failure Classification

当前 Runtime 明确区分：

```text
unsupported
insufficient_evidence
tool_failure
tool_timeout
calculation_failed
max_steps_exceeded
human_review_required
human_rejected
internal_error
```

所有失败都进入统一 Failure Handling。

最终保存：

```text
AgentErrorRecord
NodeSpan
StopReason
```

---

## 24. Evidence Verification

执行完成后进入：

```text
RuntimeEvidenceVerifier
```

其职责是检查：

- 是否真正获得所需 Fact；
- 是否存在对应 Evidence；
- Calculation 是否具有输入 Fact；
- Runtime Result 是否足够支持 Answer。

如果证据不足：

```text
status = refused
stop_reason = insufficient_evidence
```

而不是让模型自行补充解释。

---

## 25. Answer Generation

数值任务的 Answer Generator 使用 Runtime 中的结构化结果。

例如：

```text
FinancialFact
or
ComplexCalculationTrace
```

而不是再次让语言模型决定数值。

因此：

```text
Numeric Answer
      ↑
Structured Result
      ↑
Tool / Calculator
```

---

## 26. TrajectoryStore

终止运行保存为：

```text
AgentTrajectory
```

Trajectory 包含：

```text
query
intent
plan

node_spans
tool_call_traces
retrieval_traces
calculation_traces

fact_ids
evidence_ids
calculation_ids
citations

errors
answer

latency
final_status
stop_reason
```

Trajectory 保存为 immutable record。

同一个 `run_id` 不允许静默覆盖。

---

## 27. Trajectory Replay

TrajectoryStore 支持：

```text
replay
```

Replay 可以恢复：

- Node Sequence；
- Tool Sequence；
- Supporting Fact IDs；
- Evidence IDs；
- Calculation IDs；
- Final Status；
- Stop Reason。

因此可以实现：

```text
Final Answer
    ↑
Evidence / Calculation
    ↑
Tool Execution
    ↑
Plan
    ↑
User Query
```

---

## 28. Recovery Smoke Test

Step 10 完成 5 个恢复级测试。

覆盖：

### Case 1

已成功并 Checkpoint 的 Step 不重复执行。

### Case 2

Tool 成功但 Checkpoint 尚未保存时：

```text
resume
↓
same idempotency key
↓
reused
```

### Case 3

SQLite 跨 Runtime Instance 恢复。

### Case 4

Terminal Checkpoint 已存在，但 Trajectory 尚未保存：

```text
resume
↓
finalize trajectory
```

### Case 5

恢复后的 Calculation Run 可以完整 Replay。

结果：

```text
5 / 5 passed
```

---

## 29. Runtime Control Dev V1

Week 6 构造固定：

```text
50 cases
```

覆盖：

| Category | Count |
|---|---:|
| Financial Fact | 16 |
| Financial Calculation | 8 |
| Financial Comparison | 10 |
| Document Evidence | 8 |
| Clarification | 4 |
| Unsupported | 4 |
| **Total** | **50** |

---

## 30. Runtime Control Metrics

评测指标包括：

```text
Intent Accuracy
Argument Accuracy
Plan Accuracy
Tool Accuracy
Tool Sequence Accuracy
Termination Accuracy
Task Success Rate
Replay Success
```

最终结果：

```text
Cases:                    50
Passed:                   50 / 50

Intent Accuracy:          100.00%
Argument Accuracy:        100.00%
Plan Accuracy:            100.00%
Tool Accuracy:            100.00%
Tool Sequence Accuracy:   100.00%
Termination Accuracy:     100.00%
Task Success Rate:        100.00%
Replay Success:           46 / 46
```

状态分布：

```text
Completed:       42
Refused:          4
Awaiting Human:   4
Failed:           0
```

4 个 awaiting_human Case 尚未进入最终终止状态，因此不要求最终 Trajectory。

---

## 31. Runtime Control Eval 的边界

这次 50/50 证明的是：

```text
Query
↓
Parser
↓
Router
↓
Planner
↓
Tool Selection
↓
Tool Arguments
↓
Tool Sequence
↓
Termination
↓
Replay
```

在固定 Runtime Control Dev Set 上符合预期。

它不代表：

```text
真实 Agent 问答准确率 = 100%
```

也不替代以下指标：

- Retrieval Recall@k；
- MRR；
- NDCG；
- Citation Accuracy；
- Citation Completeness；
- Faithfulness；
- Answer Correctness；
- Prompt Injection Robustness；
- Permission Safety；
- HITL Risk Policy；
- End-to-End Task Success。

---

## 32. Engineering Regression

Week 6 最终工程测试：

```text
841 passed
```

意味着此前：

- Registry；
- Retrieval；
- Complex Plan；
- Calculator；
- Tool Registry；
- Runtime；
- LangGraph；
- Recovery；
- Runtime Eval；

相关测试均未发生回归。

---

## 33. Week 6 Acceptance Checklist

| Acceptance Item | Result |
|---|---|
| Structured AgentState | PASS |
| Runtime Query Parser | PASS |
| Runtime Intent Router | PASS |
| Runtime Planner | PASS |
| Agent Runtime | PASS |
| RuntimePlanExecutor | PASS |
| Tool Registry | PASS |
| Tool Schema Validation | PASS |
| Tool Permission | PASS |
| Tool Timeout | PASS |
| Tool Retry | PASS |
| Tool Idempotency | PASS |
| Financial Data Tool | PASS |
| Document Retrieval Tool | PASS |
| Calculation Tool | PASS |
| CheckpointStore | PASS |
| SQLite Recovery | PASS |
| LangGraph StateGraph | PASS |
| Conditional Routing | PASS |
| Interrupt | PASS |
| Human Resume | PASS |
| Human Rejection | PASS |
| Max Steps | PASS |
| Failure Classification | PASS |
| Evidence Verification | PASS |
| Structured Numeric Answer | PASS |
| Trajectory Persistence | PASS |
| Trajectory Replay | PASS |
| Recovery Smoke | PASS |
| 50-case Runtime Eval | PASS |
| Full Regression | PASS |

---

## 34. Week 6 Final Result

```text
Week 6 Acceptance: PASS
```

当前 Runtime 已具备：

```text
Planning
+
Tool Execution
+
Checkpoint
+
Recovery
+
Idempotency
+
LangGraph
+
Interrupt
+
Failure Handling
+
Observability
+
Trajectory
+
Replay
```

完成了从普通 RAG Pipeline 向可审计 Agent Runtime 的关键升级。

---

## 35. Week 7 Entry Baseline

Week 7 不再重新搭建 Runtime。

下一阶段直接基于当前 Week 6 基线继续增加可信与安全能力：

```text
Evidence Verification
+
Numeric Verification
+
Citation Verification
+
Tool Permission Policy
+
Prompt Injection Defense
+
Risk Classification
+
Human-in-the-loop
```

Week 6 Runtime 从此作为后续功能的稳定底座。