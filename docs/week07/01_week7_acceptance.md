# Week 7：Evidence, Permission & Responsible AI 验收报告

## 1. 本周目标

Week 7 的目标是在 Week 6 Auditable Agent Runtime 基础上建立可信与安全控制层，使系统不仅能够完成任务，还能够回答：

1. 最终 Claim 是否真的有证据支持？
2. 数值、年份、单位和财务口径是否一致？
3. Citation 是否真正指向支持该 Claim 的 Evidence？
4. 当前用户是否有权限调用该 Tool？
5. Retrieved Document 是否试图通过 Prompt Injection 控制 Agent？
6. 已通过 Trust Verification 的任务是否仍因业务风险需要人工审核？
7. 是否存在绕过 Runtime 直接发布最终答案的路径？
8. 安全机制是否会误拒正常请求？

---

## 2. Week7 最终架构

```text
User Query
    ↓
RuntimeQueryParser
    ↓
RuntimeIntentRouter
    ↓
RuntimePlanner
    ↓
RBAC
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
VerificationReport
    ↓
RuntimeRiskPolicy
  ↙       ↓        ↘
allow    refuse   require_human
  ↓                  ↓
  │                 HITL
  │             approve/reject
  └──────────┬───────┘
             ↓
RuntimeAnswerGenerator
             ↓
Generator Policy Hard Gate
             ↓
Final Answer
```

---

## 3. Trust Verification

Week7 将最终回答拆成：

```text
AnswerDraft
    ↓
Claim[]
    ↓
ClaimSupport
```

每条 Claim 都必须关联：

```text
FinancialFact
CalculationTrace
CitationRecord
RetrievedDocument
```

`RuntimeTrustVerifier` 对回答进行确定性校验，包括：

- missing evidence；
- unsupported claim；
- year mismatch；
- unit mismatch；
- statement scope mismatch；
- calculation input mismatch；
- citation mismatch；
- evidence conflict。

Trust Verification FAIL 时：

```text
status = refused
stop_reason = insufficient_evidence
```

不允许进入最终 Generator。

---

## 4. RBAC

Week7 增加 Role-Based Access Control：

```text
viewer
reviewer
admin
```

Tool Permission：

```text
read_financial_data
read_documents
execute_calculation
```

权限判断的 Authority 为：

```text
UserRole
    ↓
RuntimeAccessController
    ↓
Effective Permissions
```

`AgentState.granted_permissions` 只作为审计快照。

即使权限快照被人为扩大，Runtime 仍会重新从 Role 推导权限并拒绝越权执行。

---

## 5. Prompt Injection Defense

Retrieved Document 被明确视为：

```text
Untrusted External Data
```

系统在文档进入 Runtime State 前执行 Prompt Injection Scan。

当前确定性规则覆盖：

```text
instruction_override
system_prompt_extraction
authority_hijacking
tool_manipulation
security_bypass
```

检测后：

```text
status = refused
stop_reason = prompt_injection_detected
```

并持久化：

```text
PromptInjectionFinding
```

审计记录只保存：

- chunk_id；
- document_id；
- severity；
- rule IDs；
- safe reason；

不保存恶意正文作为运行指令。

---

## 6. Risk Policy

Trust Verification 完成后继续执行 Risk Policy：

```text
Trust FAIL
→ refuse

Trust PASS + low
→ allow

Trust PASS + medium
→ allow

Trust PASS + high
→ require_human
```

Risk Policy 与 RBAC 分离：

```text
RBAC
→ 这个用户有没有权限执行？

Risk Policy
→ 即使有权限，这个任务是否仍需人工审核？
```

Admin 权限不会自动绕过 high-risk HITL。

---

## 7. Human-in-the-loop

当 Policy 返回：

```text
require_human
```

Runtime 停止在：

```text
awaiting_human
```

Reviewer 可以：

```text
approve
→ resume
→ generate_answer
```

或：

```text
reject
→ refused
→ human_rejected
```

同时检查 reviewer role：

```text
required reviewer
→ reviewer / admin

required admin
→ admin only
```

Trust FAIL 不能被人工审批覆盖。

---

## 8. Generator Hard Gate

Week7 在 `RuntimeAnswerGenerator` 中增加第二层发布 Gate。

即使调用方绕过 `AgentRuntime`，直接调用 Generator，也必须满足：

```text
VerificationReport PASS
+
PolicyDecision exists
+
Policy permits release
+
HumanReviewDecision approved if required
```

因此 Runtime routing 不是唯一安全边界。

---

## 9. Safety Evaluation

Week7 构造固定 40-case Safety Eval：

| Category | Cases |
|---|---:|
| Evidence / Citation | 6 |
| Numeric / Scope | 6 |
| RBAC | 5 |
| Prompt Injection | 6 |
| Unsupported / Boundary | 5 |
| Risk / HITL | 6 |
| Normal Safe | 6 |
| **Total** | **40** |

最终结果：

```text
Passed:                            40 / 40

Trust Violation Detection Rate:    1.000
Prompt Injection Detection Rate:   1.000
Permission Denial Accuracy:        1.000
HITL Routing Accuracy:             1.000

False Refusal Rate:                0.000
Unsafe Answer Release Rate:        0.000
Overall Safety Success Rate:       1.000
```

---

## 10. Safety Evaluation 发现的问题

首次 Runtime-backed Safety Eval：

```text
20 / 22
```

暴露两个问题：

1. Permission Snapshot Tampering 虽然被阻止，但错误分类为 `internal_error`；
2. Unsupported Write Operation 会错误进入支持型 Runtime 流程。

修复后：

```text
22 / 22
```

Trust Tampering：

```text
12 / 12
```

Risk / HITL：

```text
6 / 6
```

最终：

```text
40 / 40
```

这说明 Safety Evaluation 不只是已有单元测试的重复，而能够发现跨组件安全问题。

---

## 11. 工程回归

Week7 最终完整工程测试：

```text
963 passed
```

运行：

```powershell
uv run pytest -q
```

Week7 Safety Eval：

```powershell
uv run pytest tests/services/test_runtime_safety_eval.py -s -q -k "full_real"
```

---

## 12. Week7 验收结果

- Evidence / Numeric Verification：PASS
- Citation Verification：PASS
- RBAC / Permission Control：PASS
- Prompt Injection Defense：PASS
- Risk Policy：PASS
- HITL：PASS
- 40 Safety Cases：40 / 40 PASS
- Normal Request False Refusal：0 / 6
- Unsafe Answer Release：0

---

## 13. Known Limitations

当前仍存在以下边界：

1. Prompt Injection Detector 主要是 deterministic pattern-based detector；
2. 尚未覆盖大规模 semantic jailbreak / obfuscation attack；
3. 当前 Risk Classification 仍较粗粒度；
4. 尚未接入真实企业身份认证与组织权限系统；
5. Safety Eval V1 只有 40 个固定 Case；
6. 40/40 只说明固定 Safety Set 上行为符合预期；
7. Retrieval 与最终 Answer Quality 仍需独立评测。

---

## 14. Week7 结论

Week7 将 Week6 的：

```text
Auditable Agent Runtime
```

升级为：

```text
Auditable
+
Evidence-grounded
+
Permission-aware
+
Injection-resistant
+
Risk-controlled
+
Human-reviewable
Agent Runtime
```

当前系统不再把“模型决定”直接等同于“允许执行 / 允许发布”，而是通过独立的 Trust、Permission、Risk 与 Human Review Gate 控制最终行为。
