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
python scripts/start_backend.py --host 127.0.0.1 --port 8000 --reload
```

启动后自检：

```bash
python scripts/health_check.py --base http://127.0.0.1:8000
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
- `POST /chat/completions`
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

## 环境变量

- `API_KEY`：模型调用密钥（唯一必需密钥配置，按要求仅用此变量）
- `DB_URL`：数据库连接串，默认 `sqlite:///data/app.db`
- `CHROMA_ROOT`：Chroma 向量目录根路径，默认 `data/chroma`
- `DOCS_ROOT`：文档存储根目录，默认 `data/docs`
- `DEFAULT_EMBEDDING_MODEL`：默认嵌入模型，默认 `bge-small-zh-v1.5`
- `EMBEDDING_MODEL_ROOT`：本地Embedding模型根目录（默认 `D:/xjtu/local_modules/models`）
- `LOCAL_MODULES_ROOT`：本地扩展模块目录（默认 `D:/xjtu/local_modules`）
- `RETRIEVAL_FUSION_MODE`：`weighted` 或 `rrf`，默认 `weighted`
- `RETRIEVAL_ALPHA`：加权融合系数，默认 `0.6`

### 本地 Embedding 模型配置

1. 将模型放到本地目录（推荐）：`D:/xjtu/local_modules/models/BAAI/bge-small-zh-v1.5`。
2. 设置环境变量（可选，不设也会走默认目录）：

```bash
set EMBEDDING_MODEL_ROOT=D:/xjtu/local_modules/models
set DEFAULT_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

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
2. 如果 `:8000` 已被多个进程占用，先结束旧进程后再用 `scripts/start_backend.py` 重启。
3. 当前后端固定输出 OpenAPI `3.0.3`，可兼容 Swagger UI，不会再出现版本字段不兼容错误。

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
