# xjtu-back

基于 FastAPI 的知识库问答后端（第一轮可用版）。

## 已实现

- 知识库管理：创建、列表、克隆、编辑、删除（逻辑/物理）。
- 文档管理：批量上传、解析、拆分、预览、删除、重建向量。
- 混合检索问答：BM25 + Chroma 双引擎融合。
- OpenAI 兼容接口：`POST /chat/completions`（含 `conversation_id` 与 `sources`）。
- RBAC 基础：登录、用户角色管理（参考 `xjtuexer/db.sql` 字段体系）。

## 目录结构

```text
app/
  api/routes/auth.py
  api/routes/chat.py
  api/routes/documents.py
  api/routes/knowledge_bases.py
  api/routes/rbac.py
  api/deps.py
  core/security.py
  models/document.py
  models/chat.py
  models/rbac.py
  core/config.py
  db/session.py
  main.py
  services/chat_service.py
  services/document_service.py
  services/retrieval_service.py
  services/kb_service.py
  services/auth_service.py
  services/rbac_service.py
  vectorstore/chroma_manager.py
```

## 安装与启动

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

推荐使用带保护启动脚本（防止重复占用 8000 端口）：

```bash
python scripts/ops.py start --host 127.0.0.1 --port 8000 --reload
```

启动后自检：

```bash
python scripts/ops.py check --probe --base http://127.0.0.1:8000
```

停止服务：

```bash
python scripts/ops.py stop --port 8000
```

启动后可访问：

- `GET /health`
- `POST /auth/login`
- `GET /auth/me`
- `POST /knowledge-bases`
- `GET /knowledge-bases`
- `PUT /knowledge-bases/{kb_id}`
- `DELETE /knowledge-bases/{kb_id}`
- `POST /knowledge-bases/{kb_id}/clone`
- `POST /knowledge-bases/{kb_id}/documents/upload`
- `GET /knowledge-bases/{kb_id}/documents`
- `GET /knowledge-bases/{kb_id}/documents/{document_id}/preview`
- `POST /knowledge-bases/{kb_id}/documents/{document_id}/reindex`
- `DELETE /knowledge-bases/{kb_id}/documents/{document_id}`
- `POST /knowledge-bases/{kb_id}/documents/batch-delete`
- `POST /chat/completions`
- `POST /chat/retrieval-debug`（返回融合前后分、重排分）
- `GET /debug/runtime`（运行时配置诊断）
- `DELETE /chat/conversations/{conversation_id}/context`
- `POST /chat/conversations/{conversation_id}/rollback?keep_rounds=2`
- `WS /ws/chat/completions?token=<access_token>`（WebSocket流式输出）
- `GET /chat/logs`（按用户/时间/关键词/知识库筛选）
- `GET /chat/logs/export`（CSV导出）
- `DELETE /chat/logs/cleanup`（手动清理）
- `GET /system-config`
- `PUT /system-config/{config_key}`
- `GET /system-config/context-policy`
- `GET /retrieval-config/global`
- `PUT /retrieval-config/global`
- `GET /retrieval-config/sessions/{conversation_id}`
- `PUT /retrieval-config/sessions/{conversation_id}`
- `GET/POST/PUT /rbac/*`
- `GET /transformer/runtime`
- `POST /transformer/chat/completions`
- `POST /transformer/classify`
- `POST /transformer/cluster`
- `POST /transformer/rag/analyze`
- `POST /transformer/eval/run`

## 环境变量

- `API_KEY`：模型调用密钥（唯一必需密钥配置，按要求仅用此变量）
- `DB_URL`：数据库连接串，默认 `sqlite:///data/app.db`
- `CHROMA_ROOT`：Chroma 向量目录根路径，默认 `data/chroma`
- `DOCS_ROOT`：文档存储根目录，默认 `data/docs`
- `DEFAULT_EMBEDDING_MODEL`：默认嵌入模型，默认 `bge-small-zh-v1.5`
- `EMBEDDING_MODEL_ROOT`：本地Embedding模型根目录（默认跟随 `D:/xjtu/local_models`）
- `LOCAL_MODULES_ROOT`：本地扩展模块目录（默认 `D:/xjtu/local_modules`，存在才启用）
- `RETRIEVAL_FUSION_MODE`：`weighted` 或 `rrf`，默认 `weighted`
- `RETRIEVAL_ALPHA`：加权融合系数，默认 `0.6`

以下参数已改为**代码内固定配置**（不再依赖环境变量）：
- 本地模型根目录：`D:/xjtu/local_models`
- 默认Embedding模型：`BAAI/bge-base-zh-v1.5`
- 重排开关：`true`
- 重排模型：`BAAI/bge-reranker-base`
- 重排候选数：`20`
- 重排权重：`0.7`

如需修改，请编辑 `app/core/config.py` 顶部常量：
- `FIXED_LOCAL_MODELS_ROOT`
- `FIXED_DEFAULT_EMBEDDING_MODEL`
- `FIXED_RERANKER_ENABLED`
- `FIXED_RERANKER_MODEL`
- `FIXED_RERANKER_TOP_N`
- `FIXED_RERANKER_WEIGHT`
- `FIXED_LOCAL_TRANSFORMER_ENABLED`
- `FIXED_LOCAL_TRANSFORMER_MODEL`
- `FIXED_LOCAL_TRANSFORMER_MAX_NEW_TOKENS`
- `FIXED_LOCAL_TRANSFORMER_TEMPERATURE`
- `FIXED_TRANSFORMER_DEVICE`

### 本地 Embedding 模型配置

1. 将模型放到本地目录（推荐）：`D:/xjtu/local_modules/models/BAAI/bge-small-zh-v1.5`。
   或直接使用你当前目录：`D:/xjtu/local_models/models--BAAI--bge-base-zh-v1.5/snapshots/<snapshot_id>`。
2. 若要换模型或目录，直接改 `app/core/config.py` 中上述常量即可。

3. 启动后创建知识库时不传 `embedding_model`，会自动使用 `DEFAULT_EMBEDDING_MODEL`。

### WebSocket 流式调用示例

前端连接地址：`ws://127.0.0.1:8000/ws/chat/completions?token=<access_token>`

发送 JSON：

```json
{
  "messages": [{"role": "user", "content": "葡萄有哪些主要品种？"}],
  "conversation_id": "conv-stream-1"
}
```

服务端回包：
- `type=meta`：会话信息
- `type=delta`：逐字增量
- `type=done`：结束及 `sources`

### docs 页面加载异常排查

1. 先执行 `python scripts/health_check.py`，确认 `/health` 和 `/openapi.json` 正常。
2. 如果 `:8000` 已被多个进程占用，先执行 `python scripts/ops.py stop --port 8000`，再启动。
3. 当前后端固定输出 OpenAPI `3.0.3`，可兼容 Swagger UI，不会再出现版本字段不兼容错误。

### 运维脚本说明

- `python scripts/ops.py check`：仅检查端口占用并输出 PID+命令行
- `python scripts/ops.py check --probe`：额外探测 `/health` 与 `/openapi.json`
- `python scripts/ops.py start --reload`：安全启动
- `python scripts/ops.py start --reload --force-stop`：自动清理冲突进程后启动
- `python scripts/ops.py stop --port 8000`：停止占用 8000 的服务

### 敏感词过滤

- 通过系统参数 `sensitive_words` 管理敏感词（逗号分隔）
- 设置示例：`PUT /system-config/sensitive_words`，值如 `违规词,测试词`
- 生效范围：用户提问和机器人回答都会做掩码替换

### 自动日志清理

- 服务启动后自动启动后台清理任务（默认每 60 分钟执行一次）
- 保留天数来自 `log_retention_days`（可通过系统参数调整）

### 检索调试

- 调试接口：`POST /chat/retrieval-debug`
- 输入：`query`、可选 `kb_ids/top_k/score_threshold/fusion_mode/alpha`
- 输出：
  - `top_k_results`：最终入选结果
  - `all_candidates`：候选全量评分（`bm25_raw/bm25_norm/dense_raw/dense_norm/fused_score/rerank_score/final_score`）

### 自动化测试

- 最小回归测试文件：`tests/test_api_smoke.py`
- 覆盖链路（逻辑删除主流程）：登录、建库、上传、检索调试、问答、日志查询、批量删文档、逻辑删知识库
- 物理删除专项：`tests/test_api_physical_delete.py`（验证物理删除+异步兜底）
- 运行：`pytest -q`

### 物理删除异步兜底

- `DELETE /knowledge-bases/{kb_id}?physical=true` 会优先尝试立即删除向量目录。
- 若遇到 Windows 文件锁，会返回 `cleanup_queued=true`，并将目录清理任务入队后台重试。
- 知识库数据库记录会先删除，向量目录由后台任务最终清理。

## 快速示例

登录获取 token：

```bash
curl -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"login_name\":\"admin\",\"password\":\"admin123\"}"
```

创建知识库：

```bash
curl -X POST "http://127.0.0.1:8000/knowledge-bases" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"grape-kb\",\"description\":\"葡萄知识库\",\"department\":\"agri\",\"owner\":\"admin\"}"
```

混合检索问答：

```bash
curl -X POST "http://127.0.0.1:8000/chat/completions" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"葡萄有哪些主要品种？\"}],\"conversation_id\":\"conv-demo\"}"
```
