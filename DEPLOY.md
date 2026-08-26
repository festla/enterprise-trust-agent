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

部署机器需要能够访问阿里云百炼 API。

## 2. 配置环境变量

在 `final_project/` 根目录：

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
DASHSCOPE_API_KEY=YOUR_API_KEY
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.8-max

COMPETITION_ATTACHMENTS_ROOT=data/competition/deploy/attachments
COMPETITION_CORPUS_DIR=data/competition/processed/corpora/competition_corpus_8cbec682dfce4962
COMPETITION_INDEX_DIR=data/competition/processed/indexes/bm25/competition_bm25_index_07bc5262330bc2a5
COMPETITION_CHECKPOINT_DB=data/runtime/checkpoints/competition_agent.sqlite3
```

提交包不包含真实 API Key。

## 3. 方式 A：直接加载 Docker Image（推荐）

加载：

```bash
docker load -i docker/enterprise-trust-agent-competition.tar
```

启动：

```bash
docker run -d   --name enterprise-trust-agent   --env-file .env   -v enterprise-trust-agent-checkpoints:/app/data/runtime/checkpoints   -p 8000:8000   enterprise-trust-agent:competition
```

Apple Silicon Mac 建议：

```bash
docker run -d   --platform linux/amd64   --name enterprise-trust-agent   --env-file .env   -v enterprise-trust-agent-checkpoints:/app/data/runtime/checkpoints   -p 8000:8000   enterprise-trust-agent:competition
```

Windows PowerShell 只需将续行符 `\` 改为反引号 `` ` ``。

## 4. 健康检查

Linux / macOS：

```bash
curl http://127.0.0.1:8000/health
```

Windows PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

正常返回：

```json
{
  "status": "ok",
  "service": "enterprise-trust-agent",
  "ready": true
}
```

浏览器访问：

```text
http://127.0.0.1:8000/docs
```

即可通过 Swagger 调用：

```text
POST /api/competition/answer
```

## 5. 示例请求

Word：

```bash
curl -X POST   http://127.0.0.1:8000/api/competition/answer   -H "Content-Type: application/json"   --data-binary @samples/Q105.json
```

Excel：

```bash
curl -X POST   http://127.0.0.1:8000/api/competition/answer   -H "Content-Type: application/json"   --data-binary @samples/Q003.json
```

正常结果：

```text
HTTP 200
result.status = answered
```

## 6. 方式 B：从源码重新构建

在 `final_project/` 根目录：

```bash
docker build   -f deploy/Dockerfile   -t enterprise-trust-agent:competition   .
```

构建完成后使用第 3 节的 `docker run` 命令启动。

运行所需附件、冻结 Corpus 与 BM25 Index 已包含在提交包中。

## 7. 常见问题

查看日志：

```bash
docker logs -f enterprise-trust-agent
```

如果 `/` 返回 404：正常，请访问 `/health` 或 `/docs`。

如果 Qwen API 调用失败，检查：

```text
DASHSCOPE_API_KEY
DASHSCOPE_API_BASE
QWEN_MODEL
网络连接
API 配额
```

如果 8000 端口被占用：

```bash
-p 8001:8000
```

然后访问：

```text
http://127.0.0.1:8001/health
```

停止并删除容器：

```bash
docker rm -f enterprise-trust-agent
```

## 8. 本地模型说明

本项目使用远程 Qwen API，因此：

```text
本地大模型权重：无
GPU：不需要
CUDA：不需要
```

提交包已包含运行所需的附件数据、冻结 Corpus、BM25 Index 和预构建 Docker Image。
