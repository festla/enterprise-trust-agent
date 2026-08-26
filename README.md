# Enterprise Trust Agent — Competition Edition

面向**银行业监管制度与统计报表**的可信问答系统。

系统将 Word/PDF 文本检索与 Excel 确定性求解统一到同一套 Agent Runtime 中：

```text
Question
  ↓
Source Resolution
  ├─ Word / PDF → BM25 Retrieval → Evidence → Sufficiency → LLM Answer
  └─ Excel      → Deterministic Solver → Evidence
                                      ↓
                               Citation Binding
                                      ↓
                         Answered / Refused / Failed
```

## 核心能力

- Word / PDF 单事实、多事实检索问答
- Excel 取数、比较、计算
- Evidence Sufficiency Gate
- Evidence / Citation 绑定
- Answered / Refused / Failed 状态区分
- LangGraph + SQLite Checkpoint / Resume
- FastAPI + Docker 部署

## 评测结果

### Dev

```text
Cases          102
Correct        102 / 102
Citation       102 / 102
Trace          102 / 102
Text LLM       138
Excel LLM      0
```

### Held-out

```text
Cases          198
Correct        195 / 198
Accuracy       98.48%
Answered       195
Refused        2
Failed/Except  1
```

| Source | Correct |
|---|---:|
| PDF | 67 / 67 |
| Word | 62 / 64 |
| Excel | 66 / 67 |

其中 195 个 Answered Case 全部回答正确。

## 评测复现

源码仓库提供：

```text
scripts/evaluate_competition_unified_agent_dev.py
scripts/evaluate_competition_unified_agent_final.py
```

Dev：

```bash
uv run --env-file .env   python -m scripts.evaluate_competition_unified_agent_dev
```

Held-out：

```bash
uv run --env-file .env   python -m scripts.evaluate_competition_unified_agent_final   --route excel

uv run --env-file .env   python -m scripts.evaluate_competition_unified_agent_final   --route text   --resume

uv run --env-file .env   python -m scripts.evaluate_competition_unified_agent_final   --score
```

输出：

```text
data/competition/processed/eval/competition_unified_agent_dev_v1.json
data/competition/processed/eval/competition_unified_agent_final_v1.json
data/competition/processed/eval/competition_unified_agent_final_score_v1.json
```

`--resume` 用于中断恢复；`--score` 用于推理完成后的独立评分。

> 评测脚本属于源码仓库的开发复现工具，正式 Docker 服务不需要运行。

## 模型与部署

```dotenv
QWEN_MODEL=qwen3.8-max
```

项目通过 OpenAI-compatible API 调用 Qwen，不依赖本地大模型权重，也不需要 GPU、CUDA、PyTorch、Transformers 或 sentence-transformers。

正式 Docker Image：

```text
enterprise-trust-agent:competition
```

## 快速运行

```bash
docker load -i docker/enterprise-trust-agent-competition.tar
cp .env.example .env
```

填写：

```dotenv
QWEN_API_KEY=YOUR_API_KEY
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.8-max
```

代码同时兼容旧变量 `DASHSCOPE_API_KEY`。

启动：

```bash
docker run -d   --name enterprise-trust-agent   --env-file .env   -v enterprise-trust-agent-checkpoints:/app/data/runtime/checkpoints   -p 8000:8000   enterprise-trust-agent:competition
```

检查：

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

示例请求：

```text
samples/Q105.json
samples/Q003.json
```

完整部署步骤见 `DEPLOY.md`。

## 项目结构

```text
final_project/
├── app/
├── data/competition/
├── deploy/
├── docker/
├── samples/
├── scripts/
├── tests/
├── .env.example
├── DEPLOY.md
├── main.py
└── pyproject.toml
```

## 安全说明

提交包不包含：

```text
真实 .env
真实 API Key
Gold Answer
Held-out Evaluation Output
本地 Checkpoint
.git
.venv
```

运行方只需通过环境变量提供自己的 API Key。
