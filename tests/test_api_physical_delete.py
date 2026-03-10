from __future__ import annotations

import time
import uuid

from fastapi.testclient import TestClient

from app.main import app


def _auth_headers(client: TestClient) -> dict[str, str]:
    resp = client.post(
        "/auth/login",
        json={"login_name": "admin", "password": "admin123"},
    )
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_physical_delete_kb_with_async_fallback() -> None:
    kb_name = f"kb-physical-{uuid.uuid4().hex[:8]}"
    with TestClient(app) as client:
        headers = _auth_headers(client)
        create_kb = client.post(
            "/knowledge-bases",
            json={
                "name": kb_name,
                "description": "physical delete test",
                "department": "qa",
                "owner": "admin",
            },
            headers=headers,
        )
        assert create_kb.status_code == 200, create_kb.text
        kb_id = create_kb.json()["id"]

        delete_kb = client.delete(
            f"/knowledge-bases/{kb_id}",
            params={"physical": True},
            headers=headers,
        )
        assert delete_kb.status_code == 200, delete_kb.text

        # Physical deletion may be queued due to transient file lock on Windows.
        cleanup_queued = delete_kb.json().get("cleanup_queued")
        assert cleanup_queued in {True, False}

        # Knowledge base record should be gone immediately.
        list_resp = client.get(
            "/knowledge-bases",
            params={"name": kb_name},
            headers=headers,
        )
        assert list_resp.status_code == 200, list_resp.text
        assert list_resp.json()["total"] == 0

        # Give async cleanup loop a short window when queued.
        if cleanup_queued is True:
            time.sleep(1.5)
