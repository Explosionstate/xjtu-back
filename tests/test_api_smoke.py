from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
import app.services.document_service as document_service
import app.services.retrieval_service as retrieval_service


def _auth_headers(client: TestClient) -> dict[str, str]:
    resp = client.post(
        "/auth/login",
        json={"login_name": "admin", "password": "admin123"},
    )
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _patch_embeddings() -> None:
    # Use deterministic fake vectors to avoid model dependency in CI/local smoke tests.
    document_service.embed_texts = lambda texts, model_name: [
        [0.1, 0.2, 0.3] for _ in texts
    ]
    retrieval_service.embed_query = lambda query, model_name: [0.1, 0.2, 0.3]
    object.__setattr__(settings, "reranker_enabled", False)


def test_login_upload_chat_logs_flow() -> None:
    _patch_embeddings()
    kb_name = f"kb-smoke-{uuid.uuid4().hex[:8]}"

    with TestClient(app) as client:
        headers = _auth_headers(client)

        create_kb = client.post(
            "/knowledge-bases",
            json={
                "name": kb_name,
                "description": "smoke test kb",
                "department": "qa",
                "owner": "admin",
            },
            headers=headers,
        )
        assert create_kb.status_code == 200, create_kb.text
        kb_id = create_kb.json()["id"]

        upload = client.post(
            f"/knowledge-bases/{kb_id}/documents/upload",
            files=[
                (
                    "files",
                    (
                        "qa.txt",
                        "葡萄常见品种包括巨峰和夏黑。".encode("utf-8"),
                        "text/plain",
                    ),
                )
            ],
            headers=headers,
        )
        assert upload.status_code == 200, upload.text
        upload_payload = upload.json()
        assert len(upload_payload) == 1
        doc_id = upload_payload[0]["id"]

        debug_resp = client.post(
            "/chat/retrieval-debug",
            json={"query": "葡萄常见品种有哪些", "kb_ids": [kb_id], "top_k": 5},
            headers=headers,
        )
        assert debug_resp.status_code == 200, debug_resp.text
        debug_json = debug_resp.json()
        assert "all_candidates" in debug_json

        chat_resp = client.post(
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "葡萄常见品种有哪些"}],
                "kb_ids": [kb_id],
                "conversation_id": f"conv-{uuid.uuid4().hex[:8]}",
            },
            headers=headers,
        )
        assert chat_resp.status_code == 200, chat_resp.text
        chat_json = chat_resp.json()
        assert chat_json["choices"][0]["message"]["content"]

        logs_resp = client.get("/chat/logs", headers=headers)
        assert logs_resp.status_code == 200, logs_resp.text
        assert logs_resp.json()["total"] >= 1

        batch_delete = client.post(
            f"/knowledge-bases/{kb_id}/documents/batch-delete",
            json={"document_ids": [doc_id]},
            headers=headers,
        )
        assert batch_delete.status_code == 200, batch_delete.text
        assert batch_delete.json()["deleted"] >= 1

        delete_kb = client.delete(
            f"/knowledge-bases/{kb_id}",
            params={"physical": True},
            headers=headers,
        )
        assert delete_kb.status_code == 200, delete_kb.text
