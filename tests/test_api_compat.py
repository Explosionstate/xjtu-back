from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models.knowledge_base import KnowledgeBase


def _ensure_active_kb(name: str) -> str:
    db = SessionLocal()
    try:
        kb = db.scalar(select(KnowledgeBase).where(KnowledgeBase.name == name))
        if kb is None:
            kb = KnowledgeBase(
                name=name,
                description="compat test kb",
                department="qa",
                owner="admin",
                status="active",
                embedding_model=settings.default_embedding_model,
            )
            db.add(kb)
            db.commit()
            db.refresh(kb)
        elif kb.status != "active":
            kb.status = "active"
            db.commit()
            db.refresh(kb)
        return str(kb.id)
    finally:
        db.close()


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
                "agentKey": "default",
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


def test_bound_agent_generation_first_prefetches_kb_context(monkeypatch) -> None:
    from app.services import chat_service

    _ensure_active_kb("学生成长助手知识库")
    observed: dict[str, object] = {}

    def _fake_answer_with_llm(*args, **kwargs):
        observed["allow_general_knowledge"] = kwargs.get("allow_general_knowledge")
        observed["kb_hit"] = kwargs.get("kb_hit")
        observed["retrieval_contexts"] = list(kwargs.get("retrieval_contexts") or [])
        observed["contexts"] = list(kwargs.get("contexts") or [])
        return chat_service.LLMAnswerResult(
            answer="这是带知识库依据的选课说明。",
            mode="llm",
        )

    def _fake_retrieve(*args, **kwargs):
        observed["kb_ids"] = list(kwargs.get("kb_ids") or [])
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "content": "本科生选课应关注培养方案、学分要求、时间窗口和课程容量限制。",
                "source_location": "学生成长助手知识库/选课要求.md",
                "score": 0.93,
            }
        ]

    monkeypatch.setattr(chat_service, "answer_with_llm", _fake_answer_with_llm)
    monkeypatch.setattr(chat_service, "hybrid_retrieve", _fake_retrieve)
    monkeypatch.setattr(chat_service, "_format_rule_answer", lambda _question: "")
    monkeypatch.setattr(chat_service, "load_user_profile_context", lambda _login: "")

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversationId": "bound-agent-prefetch-test",
                "agentKey": "student-growth",
                "llmEnabled": True,
                "localTransformerEnabled": False,
                "messages": [
                    {
                        "role": "user",
                        "content": "我是一名大一软件学院学生，我现在要选课了，选课有什么要求吗",
                    }
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        assert observed["kb_hit"] is True
        assert observed["allow_general_knowledge"] is False
        assert observed["retrieval_contexts"]
        thinking = payload["data"].get("thinking") or {}
        content = str(thinking.get("content") or "")
        assert "已结合知识库证据组织回答" in content
        assert "已先由模型独立回答" not in content


def test_bound_fact_query_prefers_retrieval_led_generation(monkeypatch) -> None:
    from app.services import chat_service

    _ensure_active_kb("学生成长助手知识库")
    observed: dict[str, object] = {"retrieve_calls": 0}

    def _fake_answer_with_llm(*args, **kwargs):
        observed["allow_general_knowledge"] = kwargs.get("allow_general_knowledge")
        observed["kb_hit"] = kwargs.get("kb_hit")
        observed["retrieval_contexts"] = list(kwargs.get("retrieval_contexts") or [])
        return chat_service.LLMAnswerResult(
            answer="请优先核对培养方案、学分限制和选课时间。",
            mode="llm",
        )

    def _fake_retrieve(*args, **kwargs):
        observed["retrieve_calls"] = int(observed["retrieve_calls"] or 0) + 1
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "content": "本科生选课前需核对培养方案、学分要求和课程修读顺序。",
                "source_location": "学生成长助手知识库/选课要求.md",
                "score": 0.95,
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "doc-1",
                "content": "选课时还需关注先修课程限制、时间冲突和课程容量。",
                "source_location": "学生成长助手知识库/选课要求.md",
                "score": 0.92,
            },
            {
                "chunk_id": "chunk-3",
                "document_id": "doc-1",
                "content": "提交前应确认退补改时间窗口及是否需要学院审批。",
                "source_location": "学生成长助手知识库/选课要求.md",
                "score": 0.90,
            },
        ]

    def _fake_workflow(**kwargs):
        kwargs["load_profile_fn"](kwargs["question"])
        contexts, _retrieved = kwargs["retrieve_fn"](kwargs["question"])
        answer = kwargs["generate_fn"](kwargs["question"], contexts, "")
        return {
            "error": "",
            "system_instruction": "",
            "answer": answer,
            "blocked_stage": "",
            "blocked_word": "",
        }

    monkeypatch.setattr(chat_service, "answer_with_llm", _fake_answer_with_llm)
    monkeypatch.setattr(chat_service, "hybrid_retrieve", _fake_retrieve)
    monkeypatch.setattr(chat_service, "run_chat_workflow_graph", _fake_workflow)
    monkeypatch.setattr(chat_service, "_format_rule_answer", lambda _question: "")
    monkeypatch.setattr(chat_service, "load_user_profile_context", lambda _login: "")

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversationId": "bound-fact-led-test",
                "agentKey": "student-growth",
                "llmEnabled": True,
                "localTransformerEnabled": False,
                "messages": [
                    {
                        "role": "user",
                        "content": "我是大一软件学院学生，我现在要选课了，选课有什么要求吗",
                    }
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        assert observed["retrieve_calls"] >= 1
        assert observed["allow_general_knowledge"] is False
        assert observed["kb_hit"] is True
        assert len(list(observed["retrieval_contexts"] or [])) >= 2
        thinking = payload["data"].get("thinking") or {}
        content = str(thinking.get("content") or "")
        assert "已结合知识库证据组织回答" in content


def test_course_selection_is_split_from_generic_service_process() -> None:
    from app.services import chat_service

    assert (
        chat_service._looks_like_course_selection_query("我现在要选课，给出选课要求")
        is True
    )
    assert (
        chat_service._looks_like_service_process_query("我现在要选课，给出选课要求")
        is False
    )


def test_student_growth_fact_query_uses_cloud_before_local(monkeypatch) -> None:
    from app.services import chat_service

    _ensure_active_kb("学生成长助手知识库")
    observed = {"cloud_calls": 0, "local_calls": 0}

    def _fake_answer_with_llm(*args, **kwargs):
        observed["cloud_calls"] += 1
        assert kwargs.get("kb_hit") is True
        assert len(list(kwargs.get("retrieval_contexts") or [])) >= 2
        return chat_service.LLMAnswerResult(answer="云端检索增强回答。", mode="llm")

    def _unexpected_local(*args, **kwargs):
        observed["local_calls"] += 1
        raise AssertionError(
            "student-growth fact query should not use local model as first answer"
        )

    def _fake_retrieve(*args, **kwargs):
        return [
            {
                "content": "先核对培养方案和学分要求。",
                "source_location": "a",
                "score": 0.9,
            },
            {
                "content": "再检查先修条件和时间冲突。",
                "source_location": "b",
                "score": 0.88,
            },
        ]

    def _fake_workflow(**kwargs):
        kwargs["load_profile_fn"](kwargs["question"])
        contexts, _retrieved = kwargs["retrieve_fn"](kwargs["question"])
        answer = kwargs["generate_fn"](kwargs["question"], contexts, "")
        return {
            "error": "",
            "system_instruction": "",
            "answer": answer,
            "blocked_stage": "",
            "blocked_word": "",
        }

    monkeypatch.setattr(chat_service, "answer_with_llm", _fake_answer_with_llm)
    monkeypatch.setattr(
        chat_service, "generate_answer_with_local_transformer", _unexpected_local
    )
    monkeypatch.setattr(chat_service, "hybrid_retrieve", _fake_retrieve)
    monkeypatch.setattr(chat_service, "run_chat_workflow_graph", _fake_workflow)
    monkeypatch.setattr(chat_service, "_format_rule_answer", lambda _question: "")
    monkeypatch.setattr(chat_service, "load_user_profile_context", lambda _login: "")

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversationId": "student-growth-cloud-first-test",
                "agentKey": "student-growth",
                "llmEnabled": True,
                "localTransformerEnabled": True,
                "messages": [{"role": "user", "content": "我现在要选课，给出选课要求"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        assert observed["cloud_calls"] == 1
        assert observed["local_calls"] == 0


def test_course_selection_local_backup_only_for_cloud_timeout(monkeypatch) -> None:
    from app.services import chat_service

    observed: dict[str, object] = {"local_calls": 0}

    def _fake_answer_with_llm(*args, **kwargs):
        return chat_service.LLMAnswerResult(
            answer="云端超时兜底。", mode="timeout_fallback"
        )

    def _fake_local(*args, **kwargs):
        observed["local_calls"] = int(observed["local_calls"] or 0) + 1
        observed["local_contexts"] = list(args[1] if len(args) > 1 else [])
        observed["local_instruction"] = str(args[6] if len(args) > 6 else "")
        return "本地兜底选课回答。", "qwen-local", {}

    def _fake_retrieve(*args, **kwargs):
        return [
            {
                "content": "先核对培养方案中的本学期课程修读要求。",
                "source_location": "doc-a",
                "score": 0.91,
            },
            {
                "content": "再确认学分上下限、先修课程和时间冲突限制。",
                "source_location": "doc-b",
                "score": 0.9,
            },
            {
                "content": "提交前检查退补改窗口和课程容量。",
                "source_location": "doc-c",
                "score": 0.88,
            },
        ]

    def _fake_workflow(**kwargs):
        kwargs["load_profile_fn"](kwargs["question"])
        contexts, _retrieved = kwargs["retrieve_fn"](kwargs["question"])
        answer = kwargs["generate_fn"](kwargs["question"], contexts, "")
        return {
            "error": "",
            "system_instruction": "",
            "answer": answer,
            "blocked_stage": "",
            "blocked_word": "",
        }

    monkeypatch.setattr(chat_service, "answer_with_llm", _fake_answer_with_llm)
    monkeypatch.setattr(
        chat_service, "generate_answer_with_local_transformer", _fake_local
    )
    monkeypatch.setattr(chat_service, "hybrid_retrieve", _fake_retrieve)
    monkeypatch.setattr(chat_service, "run_chat_workflow_graph", _fake_workflow)
    monkeypatch.setattr(chat_service, "_format_rule_answer", lambda _question: "")
    monkeypatch.setattr(chat_service, "load_user_profile_context", lambda _login: "")

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversationId": "course-selection-timeout-local-backup-test",
                "agentKey": "student-growth",
                "llmEnabled": True,
                "localTransformerEnabled": True,
                "messages": [{"role": "user", "content": "我现在要选课，给出选课要求"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        answer = payload["data"]["choices"][0]["message"]["content"]
        assert "本地兜底选课回答" in answer
        assert int(observed["local_calls"] or 0) == 1
        assert len(list(observed.get("local_contexts") or [])) >= 2
        assert "禁止输出挂失/补办/冻结" in str(observed.get("local_instruction") or "")


def test_non_course_fact_timeout_does_not_trigger_local_backup(monkeypatch) -> None:
    from app.services import chat_service

    observed = {"local_calls": 0}

    def _fake_answer_with_llm(*args, **kwargs):
        return chat_service.LLMAnswerResult(
            answer="云端超时后的稳定回答。",
            mode="timeout_fallback",
        )

    def _unexpected_local(*args, **kwargs):
        observed["local_calls"] += 1
        raise AssertionError("non-course fact timeout should not trigger local backup")

    def _fake_retrieve(*args, **kwargs):
        return [
            {
                "content": "博士生最长修业年限应以学院和研究生院最新规定为准。",
                "source_location": "doc-x",
                "score": 0.86,
            },
            {
                "content": "涉及延期时需关注申请条件和审批流程。",
                "source_location": "doc-y",
                "score": 0.84,
            },
        ]

    def _fake_workflow(**kwargs):
        kwargs["load_profile_fn"](kwargs["question"])
        contexts, _retrieved = kwargs["retrieve_fn"](kwargs["question"])
        answer = kwargs["generate_fn"](kwargs["question"], contexts, "")
        return {
            "error": "",
            "system_instruction": "",
            "answer": answer,
            "blocked_stage": "",
            "blocked_word": "",
        }

    monkeypatch.setattr(chat_service, "answer_with_llm", _fake_answer_with_llm)
    monkeypatch.setattr(
        chat_service,
        "generate_answer_with_local_transformer",
        _unexpected_local,
    )
    monkeypatch.setattr(chat_service, "hybrid_retrieve", _fake_retrieve)
    monkeypatch.setattr(chat_service, "run_chat_workflow_graph", _fake_workflow)
    monkeypatch.setattr(chat_service, "_format_rule_answer", lambda _question: "")
    monkeypatch.setattr(chat_service, "load_user_profile_context", lambda _login: "")

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversationId": "non-course-fact-timeout-test",
                "agentKey": "student-growth",
                "llmEnabled": True,
                "localTransformerEnabled": True,
                "messages": [{"role": "user", "content": "博士最长修业年限一般是多少"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        assert (
            "云端超时后的稳定回答"
            in payload["data"]["choices"][0]["message"]["content"]
        )
        assert observed["local_calls"] == 0


def test_private_reasoning_is_hidden_from_thinking_payload(monkeypatch) -> None:
    from app.services import chat_service

    monkeypatch.setattr(chat_service, "_format_rule_answer", lambda _question: "")
    monkeypatch.setattr(chat_service, "hybrid_retrieve", lambda *args, **kwargs: [])
    monkeypatch.setattr(chat_service, "load_user_profile_context", lambda _login: "")

    def _fake_answer_with_llm(*args, **kwargs):
        return chat_service.LLMAnswerResult(
            answer="这是正常回答。",
            mode="llm",
            reasoning=(
                "<system-reminder>\n"
                "Your operational mode has changed from plan to build.\n"
                "You are no longer in read-only mode.\n"
                "</system-reminder>\n"
                "CRITICAL: Plan mode ACTIVE\n"
                "READ-ONLY"
            ),
        )

    monkeypatch.setattr(chat_service, "answer_with_llm", _fake_answer_with_llm)

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversationId": "private-reasoning-hidden-test",
                "agentKey": "default",
                "llmEnabled": True,
                "localTransformerEnabled": False,
                "messages": [{"role": "user", "content": "请解释一下选课要求"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        thinking = payload["data"].get("thinking") or {}
        assert thinking.get("title") == "处理摘要"
        assert "system-reminder" not in str(thinking.get("content") or "").lower()


def test_fast_retrieval_answer_for_course_selection_is_not_generic_loss_flow() -> None:
    from app.services import chat_service

    answer = chat_service._fast_retrieval_answer(
        "我是大一软件学院学生，我现在要选课了，选课有什么要求吗",
        [
            {
                "content": "本科生选课前需核对培养方案、学分要求和课程修读顺序。",
                "source_location": "学生成长助手知识库/选课要求.md",
            }
        ],
        agent_key="student-growth",
    )
    assert "培养方案" in answer
    assert "学分" in answer
    assert "挂失" not in answer


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
        raise AssertionError(
            "hybrid mode with kb miss should not call local transformer"
        )

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


def test_chat_completion_uses_agent_bound_kb_scope(monkeypatch) -> None:
    from app.services import chat_service

    bound_kb_id = _ensure_active_kb("学生成长助手知识库")

    observed: dict[str, object] = {}

    def _fake_retrieve(*args, **kwargs):
        observed["kb_ids"] = list(kwargs.get("kb_ids") or [])
        observed["document_ids"] = kwargs.get("document_ids")
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "kb_id": bound_kb_id,
                "content": "学生成长助手知识库中的测试内容。",
                "source_location": "学生成长助手知识库/qa.txt",
                "score": 0.91,
            }
        ]

    monkeypatch.setattr(chat_service, "hybrid_retrieve", _fake_retrieve)
    monkeypatch.setattr(chat_service, "_format_rule_answer", lambda _question: "")

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversationId": "agent-bound-kb-chat-test",
                "agentKey": "student-growth",
                "llmEnabled": False,
                "localTransformerEnabled": False,
                "kbIds": ["kb-should-be-ignored"],
                "documentIds": ["doc-should-be-filtered"],
                "messages": [{"role": "user", "content": "请给我一条学习建议"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        assert observed["kb_ids"] == [bound_kb_id]
        assert observed["document_ids"] is None


def test_retrieval_debug_uses_agent_bound_kb_scope(monkeypatch) -> None:
    from app.api.routes import chat as chat_route

    bound_kb_id = _ensure_active_kb("学情报告助手知识库")

    observed: dict[str, object] = {}

    def _fake_retrieve_with_debug(*args, **kwargs):
        observed["kb_ids"] = list(kwargs.get("kb_ids") or [])
        observed["document_ids"] = kwargs.get("document_ids")
        observed["agent_key"] = kwargs.get("agent_key")
        return [], []

    monkeypatch.setattr(
        chat_route, "hybrid_retrieve_with_debug", _fake_retrieve_with_debug
    )

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/retrieval-debug",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "query": "请生成月度学情报告",
                "agentKey": "report-assistant",
                "kbIds": ["kb-should-be-ignored"],
                "documentIds": ["doc-should-be-filtered"],
                "topK": 5,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        assert observed["kb_ids"] == [bound_kb_id]
        assert observed["document_ids"] is None
        assert observed["agent_key"] == "report-assistant"


def test_risk_warning_calls_academic_tool_in_chat(monkeypatch) -> None:
    from app.services import chat_service
    from app.services.tooling_service import ToolCallResult

    observed: dict[str, object] = {"tool_called": 0, "tool_context_seen": False}

    def _fake_answer_with_llm(*args, **kwargs):
        contexts = list(kwargs.get("contexts") or [])
        observed["tool_context_seen"] = any(
            "[TOOL:get_academic_analysis]" in str(item) for item in contexts
        )
        return chat_service.LLMAnswerResult(answer="已给出预警分级。", mode="llm")

    def _fake_tool_call(tool_name, arguments, *, current_user):
        observed["tool_called"] = int(observed["tool_called"] or 0) + 1
        return ToolCallResult(
            name=str(tool_name),
            ok=True,
            data={"risk_level": "high", "key_findings": ["最近两周缺勤次数上升"]},
            error=None,
            source="academic_service",
            generated_at="2026-03-26T10:43:34+00:00",
        )

    def _fake_retrieve(*args, **kwargs):
        return [
            {
                "content": "预警分级相关：一级预警、二级预警、三级预警、触发条件。",
                "source_location": "risk-kb.md",
                "score": 0.91,
            }
        ]

    monkeypatch.setattr(chat_service, "answer_with_llm", _fake_answer_with_llm)
    monkeypatch.setattr(chat_service, "execute_tool_call", _fake_tool_call)
    monkeypatch.setattr(chat_service, "hybrid_retrieve", _fake_retrieve)
    monkeypatch.setattr(chat_service, "_format_rule_answer", lambda _question: "")

    with TestClient(app) as client:
        token = _login_and_get_token(client)
        resp = client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversationId": "risk-warning-tool-call-test",
                "agentKey": "risk-warning",
                "llmEnabled": True,
                "localTransformerEnabled": False,
                "messages": [
                    {
                        "role": "user",
                        "content": "请识别近两周学习行为中的预警信号并给出处置优先级",
                    }
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] is True
        assert "已给出预警分级" in payload["data"]["choices"][0]["message"]["content"]
        assert int(observed["tool_called"] or 0) >= 1
        assert observed["tool_context_seen"] is True
