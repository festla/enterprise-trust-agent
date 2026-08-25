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

当前 Competition Runtime 使用：

- 冻结 BM25 检索索引
- 确定性 Excel Solver
- LangGraph Agent Workflow
- Qwen OpenAI-compatible API

Competition Docker Runtime 不依赖本地：

- PyTorch
- Transformers
- sentence-transformers
- CUDA
- GPU 推理环境

文本答案生成与证据充分性判断通过百炼 OpenAI-compatible API 调用 Qwen 模型完成。

---

## 2. Deployment Environment

推荐环境：

```text
OS: Ubuntu 22.04
Architecture: x86_64
CPU: >= 4 cores
Memory: >= 16 GB
Disk: >= 10 GB
Docker: supported Linux Docker Engine
```

当前 Competition Runtime 不依赖本地 GPU。

部署机器需要能够访问：

```text
DASHSCOPE_API_BASE
```

例如：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

---

## 3. Deployment Structure

Competition 部署相关文件：

```text
enterprise-trust-agent/
│
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements-competition.txt
│
├── .env.example
├── .dockerignore
├── DEPLOY.md
│
├── app/
├── main.py
│
└── data/
```

`deploy/requirements-competition.txt` 仅包含 Competition Runtime 所需依赖，不包含本地深度学习推理依赖。

---

## 4. Required Runtime Data

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

`deploy/attachments` 使用 Linux-safe 短文件名，并通过 `source_manifest.json` 保留原始逻辑文件身份。

这些 Runtime Data 会在构建 Docker Image 时复制到镜像内部。

---

## 5. Environment Variables

复制环境变量模板：

```bash
cp .env.example .env
```

配置模型：

```dotenv
DASHSCOPE_API_KEY=YOUR_API_KEY
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.8-max
```

配置 Competition Runtime 路径：

```dotenv
COMPETITION_ATTACHMENTS_ROOT=data/competition/deploy/attachments
COMPETITION_CORPUS_DIR=data/competition/processed/corpora/competition_corpus_8cbec682dfce4962
COMPETITION_INDEX_DIR=data/competition/processed/indexes/bm25/competition_bm25_index_07bc5262330bc2a5
COMPETITION_CHECKPOINT_DB=data/runtime/checkpoints/competition_agent.sqlite3
```

不要提交包含真实 API Key 的 `.env` 文件。

---

## 6. Docker Deployment

### 6.1 Build Image

在项目根目录执行：

```bash
docker build \
  -f deploy/Dockerfile \
  -t enterprise-trust-agent:competition \
  .
```

当前 Competition Docker 仅安装实际运行所需依赖，不安装 PyTorch、Transformers、sentence-transformers 或 CUDA Runtime。

### 6.2 Recommended: Docker Compose

推荐使用 Docker Compose 启动：

```bash
docker compose \
  -f deploy/docker-compose.yml \
  up -d
```

如果需要重新构建镜像：

```bash
docker compose \
  -f deploy/docker-compose.yml \
  up -d --build
```

Compose 会自动创建用于持久化 Agent Checkpoint 的 Docker Volume。

### 6.3 Docker Run

也可以直接启动：

```bash
docker run -d \
  --name enterprise-trust-agent \
  --env-file .env \
  -v enterprise-trust-agent-checkpoints:/app/data/runtime/checkpoints \
  -p 8000:8000 \
  enterprise-trust-agent:competition
```

Docker 会自动创建不存在的 named volume。

当前推荐单进程运行，不配置多个 Uvicorn workers。

---

## 7. Health Check

执行：

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

只有 `ready=true` 时，Competition Agent Runtime 才已经完成初始化。

---

## 8. API Smoke Test

项目中提供 Competition API Smoke Test：

```bash
uv run --env-file .env \
  python -m scripts.check_competition_api
```

Smoke Test 检查：

```text
GET /health

Word Case Q105
    HTTP=200
    status=answered
    answer correct
    citation valid

Excel Case Q003
    HTTP=200
    status=answered
    answer correct
    citation valid
```

成功时输出：

```text
COMPETITION API SMOKE PASS
```

当前 Slim Docker 已完成：

```text
Q105 Word  PASS
Q003 Excel PASS
```

---

## 9. Logs

Docker Compose：

```bash
docker compose \
  -f deploy/docker-compose.yml \
  logs -f
```

Docker CLI：

```bash
docker logs -f enterprise-trust-agent
```

---

## 10. Stop Service

Docker Compose：

```bash
docker compose \
  -f deploy/docker-compose.yml \
  down
```

Docker CLI：

```bash
docker stop enterprise-trust-agent
```

重新启动：

```bash
docker start enterprise-trust-agent
```

---

## 11. Offline Docker Image Deployment

导出镜像：

```bash
docker save \
  -o enterprise-trust-agent-competition.tar \
  enterprise-trust-agent:competition
```

部署服务器加载：

```bash
docker load \
  -i enterprise-trust-agent-competition.tar
```

然后使用 `.env` 启动服务：

```bash
docker run -d \
  --name enterprise-trust-agent \
  --env-file .env \
  -v enterprise-trust-agent-checkpoints:/app/data/runtime/checkpoints \
  -p 8000:8000 \
  enterprise-trust-agent:competition
```

这种方式不需要在部署服务器重新下载 Python Base Image 或 Python Dependencies。

---

## 12. Direct Python Deployment

Docker 是推荐部署方式。

如果部署环境不使用 Docker，可以只安装 Competition Runtime 所需依赖。

创建虚拟环境：

```bash
python3 -m venv .venv
```

激活：

```bash
source .venv/bin/activate
```

安装 Competition Runtime：

```bash
python -m pip install \
  -r deploy/requirements-competition.txt
```

加载环境变量：

```bash
set -a
source .env
set +a
```

启动：

```bash
uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000
```

Direct Python Deployment 同样不需要安装 PyTorch、Transformers 或 sentence-transformers。

---

## 13. Deployment Validation

最终部署至少需要通过：

```text
1. Service Startup
    Application startup complete

2. Health Check
    GET /health
    status=ok
    ready=true

3. Word / PDF Question
    HTTP=200
    status=answered
    citation valid

4. Excel Question
    HTTP=200
    status=answered
    citation valid

5. Checkpoint
    SQLite checkpoint directory writable
```

当前 Competition Slim Docker 已完成本地验证：

```text
Health:
    ready=true

Word Q105:
    HTTP=200
    status=answered
    prediction=A
    citation=PASS

Excel Q003:
    HTTP=200
    status=answered
    prediction=A
    citation=PASS
```

同时确认 Slim Runtime 中不存在：

```text
torch
sentence_transformers
transformers
```

说明当前 Competition Deployment 不依赖本地深度学习推理环境。
