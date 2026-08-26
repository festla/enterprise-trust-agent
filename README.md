# Enterprise Trust Agent — Competition Edition

面向**银行业监管制度与统计报表**的可信问答系统。

系统将 Word/PDF 文本检索与 Excel 确定性求解统一到同一套 Agent Runtime 中，重点解决三类问题：

- **可信回答**：证据不足时拒答，不强行生成；
- **证据可追溯**：答案可追溯到原始文件、章节/段落或表格证据；
- **数值可靠**：Excel 取数、比较、计算由确定性 Solver 完成，不交给 LLM 自由计算。

## 核心流程

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
- Excel 表格取数、比较、计算
- Evidence Sufficiency Gate
- Evidence / Citation 绑定
- Answered / Refused / Failed 状态区分
- LangGraph + SQLite Checkpoint / Resume
- FastAPI + Docker 部署

文本回答可返回：

```text
answer
answer_text
citation_ids
evidence_ids
source
location
raw_content
```

因此可以从答案反向定位到原始证据。

## 评测结果

最终 held-out evaluation：

| 指标 | 结果 |
|---|---:|
| Cases | 198 |
| Correct | **195 / 198** |
| Accuracy | **98.48%** |
| Answered | 195 |
| Refused | 2 |
| Failed / Exception | 1 |

其中 **195 个 answered case 全部回答正确**。

按数据源：

| Source | Accuracy |
|---|---:|
| PDF | 67 / 67 |
| Word | 62 / 64 |
| Excel | 66 / 67 |

> 以上结果仅代表固定 held-out evaluation 数据集。

## 模型与部署

文本证据充分性判断与答案生成通过阿里云百炼 OpenAI-compatible API 调用 Qwen：

```dotenv
QWEN_MODEL=qwen3.8-max
```

项目不依赖本地大模型权重，也不需要 GPU、CUDA、PyTorch、Transformers 或 sentence-transformers。

正式 Docker Image：

```text
enterprise-trust-agent:competition
```

镜像 Content Size 约 **231 MB**。

已完成：

```text
Windows → Windows 跨设备部署  PASS
Windows → Apple Silicon Mac   PASS
真实 Agent API 请求            PASS
Evidence / Citation 输出       PASS
```

## 快速运行

```bash
docker load -i docker/enterprise-trust-agent-competition.tar
cp .env.example .env
```

填写：

```dotenv
DASHSCOPE_API_KEY=YOUR_API_KEY
```

启动：

```bash
docker run -d   --name enterprise-trust-agent   --env-file .env   -v enterprise-trust-agent-checkpoints:/app/data/runtime/checkpoints   -p 8000:8000   enterprise-trust-agent:competition
```

检查：

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

项目提供两个示例请求：

```text
samples/Q105.json   # Word
samples/Q003.json   # Excel
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
真实 DASHSCOPE_API_KEY
Gold Answer
Held-out Evaluation Output
本地 Checkpoint
.git
.venv
```

运行方只需提供自己的 DashScope API Key。
