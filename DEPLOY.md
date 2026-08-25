# Enterprise Trust Agent - Competition Deployment

## 1. Overview

Enterprise Trust Agent 是面向 Competition QA 的可信文档问答服务。

当前 Competition Runtime 支持：

- Word / PDF 文本检索问答
- Excel 表格取数
- Excel 表格比较
- Excel 表格计算
- Evidence Citation
- Evidence Sufficiency Gate
- Refusal / Failure 状态区分
- LangGraph Checkpoint
- SQLite Durable Resume

服务接口：

```text
GET  /health
POST /api/competition/answer
```

---

## 2. Deployment Environment

推荐环境：

```text
OS: Ubuntu 22.04
Architecture: x86_64
CPU: >= 4 cores
Memory: >= 16 GB
Disk: >= 30 GB
Docker: supported Linux Docker Engine
```

当前 Competition Runtime 不依赖本地 GPU 推理。

文本回答和证据充分性判断通过百炼 OpenAI-compatible API
调用 Qwen 模型。

因此部署环境需要能够访问配置的：

```text
DASHSCOPE_API_BASE
```

---

## 3. Required Runtime Data

运行服务需要以下数据：

```text
data/
└── competition/
    ├── deploy/
    │   └── attachments/
    │       ├── source_manifest.json
    │       └── files/
    │
    └── processed/
        ├── corpora/
        │   └── competition_corpus_8cbec682dfce4962/
        │
        └── indexes/
            └── bm25/
                └── competition_bm25_index_07bc5262330bc2a5/
```

`deploy/attachments` 使用 Linux-safe 短文件名，
并通过 `source_manifest.json` 保留原始逻辑文件身份。

---

## 4. Environment Variables

复制环境变量模板：

```bash
cp .env.example .env
```

填写：

```dotenv
DASHSCOPE_API_KEY=YOUR_API_KEY
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.8-max
```

Competition Runtime 路径：

```dotenv
COMPETITION_ATTACHMENTS_ROOT=data/competition/deploy/attachments
COMPETITION_CORPUS_DIR=data/competition/processed/corpora/competition_corpus_8cbec682dfce4962
COMPETITION_INDEX_DIR=data/competition/processed/indexes/bm25/competition_bm25_index_07bc5262330bc2a5
COMPETITION_CHECKPOINT_DB=data/runtime/checkpoints/competition_agent.sqlite3
```

不要提交真实 `.env` 文件。

---

## 5. Docker Deployment

### 5.1 Build Image

如果提交包中没有预构建镜像：

```bash
docker build \
  -t enterprise-trust-agent:competition \
  .
```

### 5.2 Create Checkpoint Volume

```bash
docker volume create \
  enterprise-trust-agent-checkpoints
```

该 Volume 用于持久化 SQLite Agent Checkpoint。

### 5.3 Start Service

```bash
docker run -d \
  --name enterprise-trust-agent \
  --env-file .env \
  -e COMPETITION_ATTACHMENTS_ROOT=data/competition/deploy/attachments \
  -e COMPETITION_CORPUS_DIR=data/competition/processed/corpora/competition_corpus_8cbec682dfce4962 \
  -e COMPETITION_INDEX_DIR=data/competition/processed/indexes/bm25/competition_bm25_index_07bc5262330bc2a5 \
  -e COMPETITION_CHECKPOINT_DB=data/runtime/checkpoints/competition_agent.sqlite3 \
  -v enterprise-trust-agent-checkpoints:/app/data/runtime/checkpoints \
  -p 8000:8000 \
  enterprise-trust-agent:competition
```

当前推荐单进程运行，不配置多个 Uvicorn workers。

---

## 6. Health Check

```bash
curl http://127.0.0.1:8000/health
```

正常返回：

```json
{
  "status": "ok",
  "service": "enterprise-trust-agent",
  "ready": true
}
```

只有 `ready=true` 时，Competition Agent Runtime
才已经完成初始化。

---

## 7. Logs

查看运行日志：

```bash
docker logs -f enterprise-trust-agent
```

---

## 8. Stop Service

```bash
docker stop enterprise-trust-agent
```

如需重新启动：

```bash
docker start enterprise-trust-agent
```

---

## 9. Offline Docker Image Deployment

为避免部署环境现场访问 Docker Hub，
推荐同时提供预构建 Docker Image。

构建完成后：

```bash
docker save \
  -o enterprise-trust-agent-competition.tar \
  enterprise-trust-agent:competition
```

部署机器执行：

```bash
docker load \
  -i enterprise-trust-agent-competition.tar
```

然后按照第 5.3 节启动服务。

这种方式不需要现场重新下载 Python Base Image
或 Python Dependencies。

---

## 10. Direct Python Deployment

如果部署环境不使用 Docker，也可以直接运行：

```bash
uv sync --frozen --no-dev
```

然后：

```bash
uv run --env-file .env \
  uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000
```

---

## 11. Deployment Validation

部署完成后至少检查：

```text
GET /health
    status=ok
    ready=true

Word / PDF Question
    HTTP=200
    status=answered
    citation valid

Excel Question
    HTTP=200
    status=answered
    citation valid
```

开发阶段已经使用 Word Case Q105 和
Excel Case Q003 完成 Docker API Smoke Test。