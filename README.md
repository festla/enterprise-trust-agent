# Enterprise Trust Agent

面向企业年报与财务分析场景的 **可信、可验证、可追溯 Agent Runtime**。

项目不让大模型自由生成财务事实，而是将 **结构化财务数据、文档检索、确定性计算、Agent 规划、证据验证、权限控制与安全审计** 组合成一套可执行、可恢复、可审计的企业文档分析系统。

---

## 1. Current Status

当前完成至：

**Week 7 — Trust, Safety & Responsible AI Controls**

工程回归：

```text
963 passed
```

Runtime Control Dev V1：

```text
Cases:                    50
Passed:                   50 / 50
Task Success Rate:        100%
Replay Success:           46 / 46
```

Week 7 Safety Eval V1：

```text
Cases:                             40
Passed:                            40 / 40

Trust Violation Detection Rate:    100%
Prompt Injection Detection Rate:   100%
Permission Denial Accuracy:        100%
HITL Routing Accuracy:             100%

False Refusal Rate:                  0%
Unsafe Answer Release Rate:          0%
Overall Safety Success Rate:       100%
```

> 以上结果仅代表当前固定评测集上的表现，不表示系统对所有真实场景或攻击达到绝对安全。

---

## 2. Project Architecture

```text
User Query
    ↓
RuntimeQueryParser
    ↓
RuntimeIntentRouter
    ↓
RuntimePlanner
    ↓
RBAC / Permission Gate
    ↓
RuntimePlanExecutor
    ├── query_financial_data
    ├── retrieve_documents
    └── execute_calculation
    ↓
Prompt Injection Defense
    ↓
RuntimeEvidenceVerifier
    ↓
RuntimeAnswerDraftBuilder
    ↓
RuntimeTrustVerifier
    ↓
RuntimeRiskPolicy
  ↙       ↓        ↘
allow    refuse   require_human
                      ↓
                     HITL
                      ↓
RuntimeAnswerGenerator
    ↓
Generator Hard Gate
    ↓
Final Answer
```

运行过程同时保存：

```text
AgentState
Checkpoint
NodeSpan
ToolCallTrace
RetrievalTrace
CalculationTrace
VerificationReport
PolicyDecision
HumanReviewDecision
AgentTrajectory
```

---

## 3. Core Capabilities

### Structured Financial Query

支持：

- 财务事实查询；
- 财务指标计算；
- 跨期比较；
- 文档证据分析；
- 缺失信息澄清；
- 不支持问题拒答。

### Hybrid Document Retrieval

```text
Dense Retrieval
+
BM25
↓
RRF Fusion
↓
Cross-Encoder Reranker
```

用于风险、战略、管理层说明、经营情况和原因归因等叙述性问题。

### Deterministic Calculation

派生指标由确定性 Calculator 计算，而不是由 LLM 自由生成数值。

例如：

```text
Revenue
+
Operating Cost
↓
Gross Profit Margin
```

计算过程保留 `formula_id`、`input_fact_ids`、结果和 Calculation Trace。

### Recoverable Agent Runtime

支持：

```text
run
resume
checkpoint
idempotency
trajectory replay
```

Agent 中断后可以从 Checkpoint 恢复，已成功执行的步骤不会无条件重复执行。

---

## 4. Trust & Safety

Week 7 增加多层独立安全控制。

### Trust Verification

最终答案先生成结构化 `AnswerDraft / Claim`，再验证：

- Evidence 是否存在；
- Claim 是否有事实支持；
- 年份是否一致；
- 单位是否一致；
- 财务口径是否一致；
- Calculation 输入是否一致；
- Citation 是否匹配；
- Evidence 是否冲突。

验证失败时直接拒绝发布答案。

### RBAC

角色：

```text
viewer
reviewer
admin
```

权限：

```text
read_financial_data
read_documents
execute_calculation
```

模型选择 Tool 不等于拥有 Tool 权限，Runtime 会基于真实 UserRole 重新计算有效权限。

### Prompt Injection Defense

Retrieved Document 被视为 **Untrusted External Data**。

当前检测：

```text
instruction_override
system_prompt_extraction
authority_hijacking
tool_manipulation
security_bypass
```

恶意文档在进入可信 Runtime State 前被拦截。

### Risk Policy & HITL

```text
Verification FAIL
→ refuse

Verification PASS + low / medium
→ allow

Verification PASS + high
→ require_human
```

高风险任务可以进入人工审核：

```text
approve → resume → answer
reject  → refused
```

人工审批不能覆盖 Trust Verification Failure。

### Generator Hard Gate

即使调用方绕过 AgentRuntime 直接调用 Answer Generator，也必须重新满足：

```text
Verification PASS
+
Policy permits release
+
Human approval if required
```

---

## 5. Evaluation

### Runtime Control Dev V1

固定 50-case 控制流评测，覆盖：

- Intent；
- Argument；
- Plan；
- Tool Selection；
- Tool Sequence；
- Termination；
- Replay。

结果：

```text
50 / 50 passed
```

### Week 7 Safety Eval V1

固定 40-case 安全评测：

| Category | Count |
|---|---:|
| Evidence / Citation | 6 |
| Numeric / Scope | 6 |
| RBAC | 5 |
| Prompt Injection | 6 |
| Unsupported / Boundary | 5 |
| Risk / HITL | 6 |
| Normal Safe | 6 |
| **Total** | **40** |

结果：

```text
40 / 40 passed
```

Safety Eval 曾实际发现并修复：

1. Permission Snapshot Tampering 被拦截后错误分类为 `internal_error`；
2. Unsupported Write Operation 被错误路由到支持型 Runtime。

完整说明：

```text
docs/week07/01_week7_acceptance.md
docs/week07/02_responsible_ai_control_matrix.md
```

---

## 6. Quick Start

安装依赖：

```powershell
uv sync
```

运行完整测试：

```powershell
uv run pytest -q
```

运行 Week 7 Safety Eval：

```powershell
uv run pytest tests/services/test_runtime_safety_eval.py -s -q -k "full_real"
```

---

## 7. Project Structure

```text
app/
├── schemas/          # Pydantic domain / runtime / trust schemas
├── services/         # Runtime, tools, retrieval, trust & safety
└── ...

data/
├── evaluation/
└── processed/

docs/
├── week06/
└── week07/

tests/
├── schemas/
└── services/
```

---

## 8. Current Boundary

当前项目已经具备完整的可信 Runtime 与安全控制基线，但仍有以下边界：

- Prompt Injection Detector 主要基于 deterministic patterns；
- Risk Classification 仍较粗粒度；
- 尚未接入真实企业 IAM / SSO 权限系统；
- Safety Eval V1 目前为固定 40-case；
- 40/40 不等于对所有真实攻击绝对安全；
- Retrieval、Citation Quality 与最终 Answer Quality 仍需继续扩大真实评测。

后续能力建设应继续保持现有安全 Gate：

```text
Trust Verification
RBAC
Prompt Injection Defense
Risk Policy
HITL
Generator Hard Gate
Safety Regression
```
