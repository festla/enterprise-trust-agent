# Enterprise Trust Agent

面向企业年报与可信财务分析场景的可审计文档 Agent。

项目目标不是让大模型直接生成所有答案，而是将文档检索、结构化财务事实、确定性计算、Agent 规划、工具执行、状态恢复、证据验证与轨迹审计组合成一套可验证、可恢复、可追踪的企业文档分析 Runtime。

---

## 1. Current Status

当前项目完成至：

**Week 6 — Auditable Agent Runtime**

当前工程回归：

```text
841 passed
```

Week 6 Runtime Control Dev V1：

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

需要注意：

> 这里的 100% 表示固定 Runtime Control Dev Set 上的控制流行为符合预期，并不代表真实 Retrieval、Citation 和最终 Answer Quality 已经达到 100%。

---

## 2. Project Architecture

当前核心链路：

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
AgentRuntime / LangGraph
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

运行过程同时记录：

```text
AgentState
Checkpoint
NodeSpan
ToolCallTrace
RetrievalTrace
CalculationTrace
AgentTrajectory
```

---

## 3. Runtime Query Understanding

### RuntimeQueryParser

负责从用户自然语言中解析：

- company；
- year；
- report；
- financial metric；
- statement scope；
- comparison；
- ranking；
- explanation；
- missing fields。

Parser 不负责：

- 查询数据库；
- 检索文档；
- 执行计算；
- 生成最终答案。

### RuntimeIntentRouter

负责将问题分类为：

```text
financial_fact
financial_calculation
financial_comparison
document_evidence
unsupported
```

例如：

```text
美的集团2024年营业收入是多少？
        ↓
financial_fact
```

```text
美的集团2024年毛利率是多少？
        ↓
financial_calculation
```

```text
为什么美的集团2024年营业收入增长？
        ↓
document_evidence
```

---

## 4. Runtime Planner

Planner 根据 Parsed Query 和 Intent 生成明确的 RuntimePlan。

计划中的动作包括：

```text
retrieve
calculate
compare
rank
synthesize
```

例如毛利率：

```text
User:
美的集团2024年毛利率是多少？

        ↓

q1: revenue
q2: operating_cost

        ↓

s1: query_financial_data
s2: query_financial_data

        ↓

s3: execute_calculation
```

Planner 只描述：

> 应该获取什么、应该执行什么。

它不会提前知道真正得到的 `fact_id`。

---

## 5. Tool Registry

Runtime 当前包含三个核心 Production Tool。

### 5.1 query_financial_data

用于查询经过验证的结构化财务事实。

主要约束：

- 只使用 FinancialFact Registry；
- 优先使用 verified Fact；
- 返回对应 Evidence；
- 数值不允许由语言模型自由生成。

结构化数值问题遵循：

```text
User Question
      ↓
Planner
      ↓
query_financial_data
      ↓
FinancialFact
      ↓
Verified Numeric Value
```

---

### 5.2 retrieve_documents

用于查询年报中的叙述性证据。

当前检索链路：

```text
Dense Retrieval
      +
BM25
      ↓
RRF Fusion
      ↓
Cross-Encoder Reranker
      ↓
RetrievedDocument
```

适用于：

- 风险分析；
- 管理层说明；
- 战略；
- 经营情况；
- 原因归因；
- 未来展望。

---

### 5.3 execute_calculation

用于执行确定性财务指标计算。

例如：

```text
Revenue
+
Operating Cost
      ↓
Gross Profit Margin
```

计算过程保留：

- calculation_id；
- formula_id；
- input_fact_ids；
- result；
- result_unit；
- calculation trace。

当前已支持多种派生指标，包括：

- gross profit margin；
- current ratio；
- debt-to-equity ratio；
- operating cash flow / net profit；
- selling + R&D expense ratio；
- effective income tax rate。

---

## 6. Tool Execution Control

所有 Tool 通过统一 Tool Registry 和 ToolExecutor 执行。

Tool Definition 包含：

```text
tool_name
version
input_schema
output_schema
permission
timeout
max_retries
idempotent
max_result_bytes
```

Runtime 因此可以统一处理：

- Schema Validation；
- Permission；
- Timeout；
- Retry；
- Idempotency；
- Result Size；
- ToolCallTrace。

---

## 7. AgentState

Agent 的运行状态统一保存在 `AgentState` 中。

主要信息包括：

```text
query
parsed_query
intent
runtime_plan

current_step
completed_step_ids
runtime_refs

tool_call_traces
retrieval_traces
calculation_traces

resolved_fact_ids
evidence_ids
calculation_ids
citations

status
stop_reason

retry_count
errors

current_node
next_node

checkpoint_revision
```

因此 Agent 的执行过程不是隐藏在 Python 调用栈中的，而是可以：

```text
serialize
↓
checkpoint
↓
recover
↓
replay
```

---

## 8. Agent Runtime

Runtime 提供两条主要入口：

```python
runtime.run(...)
```

以及：

```python
runtime.resume(...)
```

`run()` 用于启动新的 Agent 运行。

`resume()` 用于从已有 Checkpoint 继续执行。

Runtime 根据：

```text
next_node
current_step
completed_step_ids
runtime_refs
```

判断恢复位置。

---

## 9. Checkpoint

当前支持：

```text
InMemoryCheckpointStore
SQLiteCheckpointStore
```

Checkpoint 使用 append-only revision。

例如：

```text
revision 1
    ↓
revision 2
    ↓
revision 3
```

而不是不断覆盖同一个 State。

同时使用 `expected_revision` 做乐观并发控制。

---

## 10. Recovery

Checkpoint 主要解决：

> Agent 中断以后从哪里继续？

例如：

```text
s1 completed
      ↓
Checkpoint
current_step = 1
      ↓
Process Crash
      ↓
resume()
      ↓
continue from s2
```

已经写入 Checkpoint 的步骤不会从头重新执行。

---

## 11. Idempotency

Checkpoint 和 Idempotency 解决的是两个不同的问题。

### Checkpoint

解决：

> 从哪里继续？

### Idempotency

解决：

> Tool 已经成功，但成功结果还没有写入 Checkpoint 时发生崩溃怎么办？

例如：

```text
s1 Tool succeeded
      ↓
Process Crash
      ↓
Checkpoint still before s1
      ↓
resume()
      ↓
Runtime requests s1 again
```

ToolExecutor 使用稳定的 Idempotency Key：

```text
run_id
+
step_id
+
tool_name
+
arguments
```

如果执行上下文相同，可以复用已有结果：

```text
status = reused
```

从而避免重复副作用。

---

## 12. LangGraph

LangGraph 位于 Framework-independent Runtime 之上。

整体关系：

```text
LangGraph
    ↓
AgentRuntime
    ↓
Business Services
```

而不是：

```text
Business Logic
全部直接写入 Graph Node
```

LangGraph 主要负责：

- StateGraph；
- Node；
- Edge；
- Conditional Edge；
- thread_id；
- Checkpointer；
- interrupt；
- resume。

核心业务逻辑仍由：

```text
Parser
Router
Planner
RuntimePlanExecutor
Verifier
Generator
```

负责。

---

## 13. Human Interrupt

当关键字段缺失时：

```text
User Query
    ↓
Parser
    ↓
missing company / year
    ↓
awaiting_human
    ↓
LangGraph interrupt
```

用户补充信息后：

```text
Command(resume=...)
      ↓
same thread_id
      ↓
resume graph
      ↓
parse corrected query
```

用户也可以拒绝继续执行：

```text
human_rejected
```

---

## 14. Failure Control

Runtime 当前区分：

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

每次终止都有：

```text
status
+
stop_reason
```

例如：

```text
status = refused
stop_reason = insufficient_evidence
```

或者：

```text
status = failed
stop_reason = tool_timeout
```

这样可以避免无休止 Agent Loop。

---

## 15. Trajectory

终止后的 Agent Run 会保存为 `AgentTrajectory`。

Trajectory 包含：

- Query；
- Intent；
- RuntimePlan；
- NodeSpan；
- ToolCallTrace；
- RetrievalTrace；
- CalculationTrace；
- Fact IDs；
- Evidence IDs；
- Citation；
- Error；
- Answer；
- latency；
- final status；
- stop reason。

TrajectoryStore 支持：

```text
save
load
list
export
replay
```

Trajectory 采用 immutable save：

> 同一个 run_id 的最终轨迹不能被静默覆盖。

---

## 16. Runtime Replay

Replay 用于从保存的 AgentTrajectory 中重新还原：

```text
Node Sequence
Tool Sequence
Fact IDs
Evidence IDs
Calculation IDs
Final Status
Stop Reason
```

因此最终答案可以追溯到：

```text
Answer
  ↑
Evidence / Calculation
  ↑
Tool
  ↑
Plan
  ↑
Query
```

---

## 17. Runtime Control Evaluation

生成固定 50-case Runtime Control Dev Set：

```powershell
uv run python -m scripts.build_runtime_control_dev_v1
```

运行：

```powershell
uv run python -m scripts.evaluate_runtime_control_dev_v1
```

评测指标：

- Intent Accuracy；
- Argument Accuracy；
- Plan Accuracy；
- Tool Accuracy；
- Tool Sequence Accuracy；
- Termination Accuracy；
- Task Success Rate；
- Replay Success。

数据集：

```text
data/evaluation/runtime/runtime_control_dev_v1.jsonl
```

结果：

```text
data/processed/evaluations/runtime/runtime_control_dev_v1/
├── results.jsonl
└── summary.json
```

完整 Trajectory 为可重复生成的运行产物，不提交 Git。

---

## 18. Current Evaluation Result

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

其中：

```text
42 completed
4 refused
4 awaiting_human
```

4 个 `awaiting_human` Case 尚未进入最终终止状态，因此不要求最终 Trajectory。

---

## 19. Testing

运行完整工程回归：

```powershell
uv run pytest -q
```

当前基线：

```text
841 passed
```

运行 Week 6 Acceptance：

```powershell
uv run python -m scripts.check_week6_acceptance
```

---

## 20. Evaluation Boundary

Runtime Control Dev V1 主要验证：

```text
Query
↓
Parsing
↓
Intent
↓
Planning
↓
Tool Selection
↓
Tool Arguments
↓
Tool Sequence
↓
Termination
↓
Trajectory Replay
```

它并不等价于最终可信问答能力。

以下指标仍需独立评测：

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

## 21. Next Stage

Week 7 将基于当前 Runtime 继续建设：

```text
Evidence Verification
+
Numeric Verification
+
Citation Verification
+
Permission Policy
+
Prompt Injection Defense
+
Risk Policy
+
Human-in-the-loop
```

Week 6 Runtime 将作为后续开发的固定工程基线。