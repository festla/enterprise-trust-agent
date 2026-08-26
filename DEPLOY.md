# Enterprise Trust Agent — Deployment Guide

推荐直接使用提交包中的预构建 Docker Image。

## 1. 环境要求

```text
Docker Engine / Docker Desktop
CPU >= 4 cores
Memory >= 8 GB
Disk >= 10 GB
GPU 不需要
```

部署机器需要能够访问 Qwen OpenAI-compatible API。

## 2. 配置环境变量

Linux / macOS：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
QWEN_API_KEY=YOUR_API_KEY
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.8-max

COMPETITION_ATTACHMENTS_ROOT=data/competition/deploy/attachments
COMPETITION_CORPUS_DIR=data/competition/processed/corpora/competition_corpus_8cbec682dfce4962
COMPETITION_INDEX_DIR=data/competition/processed/indexes/bm25/competition_bm25_index_07bc5262330bc2a5
COMPETITION_CHECKPOINT_DB=data/runtime/checkpoints/competition_agent.sqlite3
```

代码同时兼容旧变量 `DASHSCOPE_API_KEY`。提交包不包含真实 API Key。

## 3. 加载并启动 Docker

```bash
docker load -i docker/enterprise-trust-agent-competition.tar
```

```bash
docker run -d   --name enterprise-trust-agent   --env-file .env   -v enterprise-trust-agent-checkpoints:/app/data/runtime/checkpoints   -p 8000:8000   enterprise-trust-agent:competition
```

Apple Silicon Mac 可增加：

```text
--platform linux/amd64
```

## 4. 健康检查

```bash
curl http://127.0.0.1:8000/health
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

接口：

```text
POST /api/competition/answer
```

## 5. 示例请求

```bash
curl -X POST   http://127.0.0.1:8000/api/competition/answer   -H "Content-Type: application/json"   --data-binary @samples/Q105.json
```

Excel 示例：

```text
samples/Q003.json
```

## 6. 从源码重新构建

```bash
docker build   -f deploy/Dockerfile   -t enterprise-trust-agent:competition   .
```

## 7. 开发评测复现

仅用于源码仓库开发评测，正式 Docker 部署不需要执行。

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

结果：

```text
data/competition/processed/eval/competition_unified_agent_dev_v1.json
data/competition/processed/eval/competition_unified_agent_final_v1.json
data/competition/processed/eval/competition_unified_agent_final_score_v1.json
```

Final evaluator 固定使用 `qwen3.8-max`；`--resume` 用于中断恢复，`--score` 用于独立评分。

## 8. 常见问题

查看日志：

```bash
docker logs -f enterprise-trust-agent
```

Qwen API 调用失败时检查：

```text
QWEN_API_KEY
DASHSCOPE_API_BASE
QWEN_MODEL
网络连接
API 配额
```

停止并删除容器：

```bash
docker rm -f enterprise-trust-agent
```

## 9. 本地模型说明

```text
本地大模型权重：无
GPU：不需要
CUDA：不需要
```
