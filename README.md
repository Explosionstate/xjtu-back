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
- `GET/POST/PUT /rbac/*`

## 环境变量

- `API_KEY`：模型调用密钥（唯一必需密钥配置，按要求仅用此变量）
- `DB_URL`：数据库连接串，默认 `sqlite:///data/app.db`
- `CHROMA_ROOT`：Chroma 向量目录根路径，默认 `data/chroma`
- `DOCS_ROOT`：文档存储根目录，默认 `data/docs`
- `DEFAULT_EMBEDDING_MODEL`：默认嵌入模型，默认 `bge-small-zh-v1.5`
- `RETRIEVAL_FUSION_MODE`：`weighted` 或 `rrf`，默认 `weighted`
- `RETRIEVAL_ALPHA`：加权融合系数，默认 `0.6`

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
