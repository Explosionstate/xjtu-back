from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def _login_and_get_token(client: TestClient) -> str:
    login_resp = client.post(
        "/api/auth/login",
        json={"loginName": "admin", "password": "admin123"},
    )
    assert login_resp.status_code == 200, login_resp.text
    login_payload = login_resp.json()
    assert login_payload["status"] is True
    assert login_payload["code"] == 0
    token = login_payload["data"]["access_token"]
    assert token
    return str(token)


def test_api_prefix_envelope_and_alias_login() -> None:
    with TestClient(app) as client:
        token = _login_and_get_token(client)

        me_resp = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert me_resp.status_code == 200, me_resp.text
        me_payload = me_resp.json()
        assert me_payload["status"] is True
        assert me_payload["code"] == 0
        assert me_payload["data"]["login_name"] == "admin"

        raw_login_resp = client.post(
            "/auth/login",
            json={"login_name": "admin", "password": "admin123"},
        )
        assert raw_login_resp.status_code == 200, raw_login_resp.text
        assert "access_token" in raw_login_resp.json()


def test_api_prefix_error_envelope() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/auth/me")
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is False
        assert payload["code"] == 70005
        assert isinstance(payload["message"], str)


def test_api_agents_catalog_is_exposed_and_complete() -> None:
    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.get(
            "/api/chat/agents",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        assert payload["code"] == 0
        data = payload["data"]
        assert isinstance(data, dict)
        assert int(data.get("total") or 0) >= 6
        items = data.get("items") or []
        keys = {item.get("key") for item in items if isinstance(item, dict)}
        assert {
            "student-growth",
            "teacher-assistant",
            "counselor-ideology",
            "risk-warning",
            "report-assistant",
            "policy-qa",
        }.issubset(keys)


def test_api_retrieval_config_alias_and_simple_mode_mapping() -> None:
    with TestClient(app) as client:
        token = _login_and_get_token(client)
        conversation_id = "compat-test-conv"
        resp = client.put(
            f"/api/retrieval-config/sessions/{conversation_id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"topK": 20, "fusionMode": "simple", "scoreThreshold": 0.18},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        assert payload["code"] == 0
        data = payload["data"]
        assert int(data["retrieval_top_k"]) == 20
        # Explicit compatibility mapping: simple -> weighted
        assert data["fusion_mode"] == "weighted"


def test_cloud_only_mode_skips_rag_and_keeps_direct_answer(monkeypatch) -> None:
    from app.services import chat_service

    def _fake_answer_with_llm(*args, **kwargs):
        assert kwargs.get("allow_general_knowledge") is True
        assert kwargs.get("kb_hit") is False
        return chat_service.LLMAnswerResult(
            answer="这是云端 Qwen 直答内容。",
            mode="llm",
        )

    def _unexpected_retrieve(*args, **kwargs):
        raise AssertionError("cloud_only mode should not call retrieval")

    monkeypatch.setattr(chat_service, "answer_with_llm", _fake_answer_with_llm)
    monkeypatch.setattr(chat_service, "hybrid_retrieve", _unexpected_retrieve)
    monkeypatch.setattr(chat_service, "_format_rule_answer", lambda _question: "")

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversationId": "cloud-direct-test",
                "agentKey": "student-growth",
                "llmEnabled": True,
                "localTransformerEnabled": False,
                "kbIds": ["kb-x"],
                "documentIds": ["doc-x"],
                "messages": [{"role": "user", "content": "请解释牛顿第一定律"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        answer = payload["data"]["choices"][0]["message"]["content"]
        assert "云端 Qwen 直答内容" in answer
        assert "资料不足" not in answer
        assert "知识库" not in answer


def test_hybrid_mode_kb_miss_uses_cloud_completion(monkeypatch) -> None:
    from app.services import chat_service

    cloud_calls = {"count": 0}

    def _fake_answer_with_llm(*args, **kwargs):
        cloud_calls["count"] += 1
        if kwargs.get("allow_general_knowledge") is True:
            assert kwargs.get("kb_hit") is False
        return chat_service.LLMAnswerResult(
            answer="混合模式下的云端补全回答。",
            mode="llm",
        )

    def _empty_retrieve(*args, **kwargs):
        return []

    def _unexpected_local(*args, **kwargs):
        raise AssertionError("hybrid mode with kb miss should not call local transformer")

    monkeypatch.setattr(chat_service, "answer_with_llm", _fake_answer_with_llm)
    monkeypatch.setattr(chat_service, "hybrid_retrieve", _empty_retrieve)
    monkeypatch.setattr(
        chat_service,
        "generate_answer_with_local_transformer",
        _unexpected_local,
    )
    monkeypatch.setattr(chat_service, "_format_rule_answer", lambda _question: "")

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversationId": "hybrid-kb-miss-test",
                "agentKey": "teacher-assistant",
                "llmEnabled": True,
                "localTransformerEnabled": True,
                "kbIds": ["kb-x"],
                "messages": [{"role": "user", "content": "给我一份英语学习规划"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        answer = payload["data"]["choices"][0]["message"]["content"]
        assert "混合模式下的云端补全回答" in answer
        assert cloud_calls["count"] >= 1
