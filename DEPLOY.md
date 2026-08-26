# Enterprise Trust Agent — Deployment Guide

## 0. 最短部署路径

本交付包已经包含：

- 完整 Agent 运行代码；
- 比赛运行所需附件；
- 已构建的文本 Corpus；
- BM25 检索索引；
- 已构建完成的 Docker 镜像。

因此，**标准部署无需重新构建 Docker 镜像，也无需额外下载比赛数据、本地模型权重、GPU 或 CUDA 环境**。

如果收到的是：

```text
zhixinyance_project.tar.gz
```

先解压：

```bash
tar -xzf zhixinyance_project.tar.gz
cd final_project
```

随后只需要：

```text
配置 .env
  ↓
docker load
  ↓
docker run
  ↓
检查 /health
```

预构建 Docker 镜像位于：

```text
docker/enterprise-trust-agent-competition.tar
```

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

说明：

- `QWEN_API_KEY` 由部署方自行提供；
- 提交包不包含真实 API Key；
- 代码同时兼容旧变量 `DASHSCOPE_API_KEY`；
- 如果部署方使用其他兼容的 HTTPS API 地址，只需相应修改 `DASHSCOPE_API_BASE`；
- 正式模型为 `qwen3.8-max`。

## 3. 加载预构建 Docker 镜像

```bash
docker load -i docker/enterprise-trust-agent-competition.tar
```

加载成功后应得到：

```text
enterprise-trust-agent:competition
```

正常部署无需再次执行 `docker build`。

## 4. 启动服务

```bash
docker run -d \
  --name enterprise-trust-agent \
  --env-file .env \
  -v enterprise-trust-agent-checkpoints:/app/data/runtime/checkpoints \
  -p 8000:8000 \
  enterprise-trust-agent:competition
```

Apple Silicon Mac 如有平台兼容提示，可增加：

```text
--platform linux/amd64
```

查看容器状态：

```bash
docker ps
```

## 5. 健康检查

Linux / macOS：

```bash
curl http://127.0.0.1:8000/health
```

Windows PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

正常情况下应看到：

```text
status = ok
ready = true
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

接口：

```text
POST /api/competition/answer
```

## 6. 示例请求

Word 示例。

Linux / macOS：

```bash
curl -X POST \
  http://127.0.0.1:8000/api/competition/answer \
  -H "Content-Type: application/json" \
  --data-binary @samples/Q105.json
```

Windows PowerShell：

```powershell
$response = Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/competition/answer `
  -Method Post `
  -ContentType "application/json" `
  -InFile samples/Q105.json

$response.result
```

Excel 示例：

```text
samples/Q003.json
```

## 7. 从源码重新构建（可选）

正常部署**不需要执行本节**。

仅在需要修改源码并重新构建镜像时执行：

```bash
docker build \
  -f deploy/Dockerfile \
  -t enterprise-trust-agent:competition \
  .
```

## 8. 开发评测复现（可选）

仅用于源码仓库开发评测，正式 Docker 部署不需要执行。

Dev：

```bash
uv run --env-file .env \
  python -m scripts.evaluate_competition_unified_agent_dev
```

Held-out：

```bash
uv run --env-file .env \
  python -m scripts.evaluate_competition_unified_agent_final \
  --route excel

uv run --env-file .env \
  python -m scripts.evaluate_competition_unified_agent_final \
  --route text \
  --resume

uv run --env-file .env \
  python -m scripts.evaluate_competition_unified_agent_final \
  --score
```

结果：

```text
data/competition/processed/eval/competition_unified_agent_dev_v1.json
data/competition/processed/eval/competition_unified_agent_final_v1.json
data/competition/processed/eval/competition_unified_agent_final_score_v1.json
```

Final evaluator 固定使用 `qwen3.8-max`；`--resume` 用于中断恢复，`--score` 用于独立评分。

## 9. 常见问题

### 容器启动失败

查看日志：

```bash
docker logs -f enterprise-trust-agent
```

### Qwen API 调用失败

检查：

```text
QWEN_API_KEY
DASHSCOPE_API_BASE
QWEN_MODEL
网络连接
API 配额
```

如果返回 `401 invalid_api_key`，优先确认：

- API Key 是否有效；
- `DASHSCOPE_API_BASE` 是否与部署方实际使用的模型服务地址匹配。

### 停止并删除容器

```bash
docker rm -f enterprise-trust-agent
```

## 10. 本地模型说明

```text
本地大模型权重：无
GPU：不需要
CUDA：不需要
```

系统运行所需比赛数据已经包含在交付包和预构建 Docker 镜像中，无需额外挂载附件、Corpus 或 BM25 索引。
