from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
import time

from rank_bm25 import BM25Okapi
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import DocumentChunk
from app.models.knowledge_base import KnowledgeBase
from app.services.agent_profile_service import (
    get_agent_retrieval_focus_terms,
    get_agent_source_bias,
    normalize_agent_key,
)
from app.services.embedding_service import embed_query, normalize_embedding_model_name
from app.services.reranker_service import rerank_candidates
from app.vectorstore.chroma_manager import search_similar_chunks

_RETRIEVAL_CACHE_TTL_SECONDS = 18.0
_RETRIEVAL_CACHE_MAX_SIZE = 256
_RETRIEVAL_CACHE: dict[str, tuple[float, list[dict], list[dict]]] = {}


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", (text or "").lower())
    return [tok for tok in cleaned.split() if tok]


def _is_small_talk(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return True
    small_talk_keywords = ["你好", "在吗", "hello", "hi", "嗨", "你是谁"]
    return any(keyword in q for keyword in small_talk_keywords) and len(q) <= 12


def _looks_like_schema_chunk(content: str) -> bool:
    lowered = (content or "").lower()
    markers = [
        "create table",
        "alter table",
        "insert into",
        "update ",
        "delete from",
        "values (",
        "primary key",
        "foreign key",
        " unique key",
        " key `",
        " idx_",
        " index ",
        "constraint ",
        "comment '",
        "engine = innodb",
        "drop table",
    ]
    return any(marker in lowered for marker in markers)


def _looks_like_command_chunk(content: str) -> bool:
    lowered = (content or "").lower()
    markers = [
        "pip install",
        "npm install",
        "python -m",
        "requirements.txt",
        "localhost:",
        "127.0.0.1",
        "uvicorn",
        "taskkill",
        "--reload",
    ]
    return any(marker in lowered for marker in markers)


def _is_summary_style_query(query: str) -> bool:
    q = (query or "").strip().lower()
    flags = ["总结", "概述", "重点", "新增文档", "指南", "手册"]
    return any(flag in q for flag in flags)


def _is_guidance_query(query: str) -> bool:
    q = (query or "").strip().lower()
    flags = ["学生", "建议", "指南", "学习", "生活", "成长"]
    return any(flag in q for flag in flags)


def _is_stress_conflict_query(query: str) -> bool:
    q = (query or "").strip().lower()
    flags = [
        "期末",
        "考试",
        "焦虑",
        "熬夜",
        "效率",
        "冲突",
        "换届",
        "选举",
        "任务太多",
        "看不进",
    ]
    return any(flag in q for flag in flags)


def _is_student_growth_agent(agent_key: str | None) -> bool:
    key = normalize_agent_key(agent_key)
    return key == "student-growth"


def _expand_query_for_agent(agent_key: str | None, query: str) -> str:
    base = (query or "").strip()
    if not base:
        return base
    focus_terms = list(get_agent_retrieval_focus_terms(agent_key))
    if not focus_terms:
        return base
    supplements = [term for term in focus_terms if term and term not in base][:4]
    if not supplements:
        return base
    return f"{base}\n检索侧重点: {'、'.join(supplements)}"


def _is_mojibake_text(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    replacement_ratio = text.count("�") / max(1, len(text))
    if replacement_ratio >= 0.03:
        return True
    mojibake_markers = ["脙", "脗", "盲赂", "莽", "氓", "忙", "茂录", "茅"]
    marker_hits = sum(text.count(marker) for marker in mojibake_markers)
    return marker_hits >= 12 and marker_hits / max(1, len(text)) > 0.05


def _source_weight(agent_key: str | None, source_location: str, query: str) -> float:
    src = (source_location or "").lower()
    q = (query or "").lower()
    positive, negative = get_agent_source_bias(agent_key)

    weight = 1.0
    if any(token.lower() in src for token in positive):
        weight *= 1.2
    if any(token.lower() in src for token in negative):
        weight *= 0.5

    normalized_agent_key = normalize_agent_key(agent_key)
    if normalized_agent_key == "risk-warning" and any(
        token in q for token in ["风险", "预警", "异常", "warning"]
    ):
        if "risk" in src or "warning" in src or "预警" in src:
            weight *= 1.18
    if normalized_agent_key == "policy-qa" and any(
        token in q for token in ["政策", "条例", "制度", "规范"]
    ):
        if any(token in src for token in ["policy", "条例", "制度", "规范", "办法"]):
            weight *= 1.22

    if normalized_agent_key == "student-growth" and _is_stress_conflict_query(q):
        if any(token in src for token in ["成长", "学情", "预警", "辅导员", "心理"]):
            weight *= 1.15
        if any(token in src for token in ["问答", "政策", "条例", "制度"]):
            weight *= 0.72

    return max(0.2, min(1.5, weight))


def _is_academic_query(query: str) -> bool:
    q = (query or "").strip().lower()
    flags = ["学业", "成绩", "课堂", "学习", "课程", "预警"]
    return any(flag in q for flag in flags)


def _is_precise_fact_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if _is_summary_style_query(q):
        return False
    fact_tokens = [
        "政策",
        "规定",
        "流程",
        "步骤",
        "办理",
        "申请",
        "材料",
        "要求",
        "时间",
        "地点",
        "学分",
        "绩点",
        "是什么",
        "多少",
        "几天",
        "多久",
        "挂失",
        "补办",
    ]
    open_tokens = [
        "焦虑",
        "迷茫",
        "纠结",
        "压力",
        "方向",
        "职业",
        "怎么办",
        "怎么选",
        "要不要",
    ]
    fact_hits = sum(1 for token in fact_tokens if token in q)
    open_hits = sum(1 for token in open_tokens if token in q)
    return fact_hits >= 2 and open_hits == 0 and len(q) <= 72


def _is_complex_retrieval_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if len(q) >= 64:
        return True
    complex_markers = [
        "分析",
        "对比",
        "比较",
        "趋势",
        "报告",
        "风险",
        "干预",
        "规划",
        "方案",
        "评估",
    ]
    return any(marker in q for marker in complex_markers)


def _should_skip_reranker(
    *,
    query: str,
    top_k: int,
    candidate_count: int,
    precise_fact_query: bool,
) -> bool:
    if candidate_count <= 0:
        return True
    if precise_fact_query and candidate_count <= 16:
        return True
    if not _is_complex_retrieval_query(query) and top_k <= 3 and candidate_count <= 18:
        return True
    return False


def _token_overlap_score(query_tokens: set[str], content: str) -> float:
    if not query_tokens:
        return 0.0
    content_tokens = set(_tokenize(content))
    if not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens)
    return overlap / max(1, len(query_tokens))


def _looks_like_noise_chunk(content: str) -> bool:
    text = (content or "").strip()
    lowered = text.lower()
    if not text:
        return True
    if _looks_like_schema_chunk(text) or _looks_like_command_chunk(text):
        return True
    noisy_tokens = ["`", "{", "}", "=>", "::", "localhost", "http://", "https://"]
    return sum(token in lowered for token in noisy_tokens) >= 4


def _keyword_bonus(query: str, source_location: str, content: str) -> float:
    q = (query or "").strip()
    if not q:
        return 0.0
    src = (source_location or "").lower()
    body = (content or "").lower()
    bonus = 0.0
    keyword_weights = [
        ("学生", 0.12),
        ("指南", 0.12),
        ("手册", 0.10),
        ("学业", 0.08),
        ("成长", 0.08),
        ("建议", 0.08),
        ("生活", 0.06),
        ("政策", 0.10),
        ("条例", 0.08),
        ("预警", 0.09),
    ]
    for keyword, weight in keyword_weights:
        if keyword in q and (keyword in src or keyword in body[:260]):
            bonus += weight
    return min(0.35, bonus)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s == min_s:
        return {key: 1.0 for key in scores}
    return {key: (val - min_s) / (max_s - min_s) for key, val in scores.items()}


def _fuse_weighted(
    bm25_scores: dict[str, float], dense_scores: dict[str, float], alpha: float
) -> dict[str, float]:
    bm25_norm = _normalize_scores(bm25_scores)
    dense_norm = _normalize_scores(dense_scores)
    keys = set(bm25_norm.keys()) | set(dense_norm.keys())
    return {
        key: alpha * bm25_norm.get(key, 0.0) + (1.0 - alpha) * dense_norm.get(key, 0.0)
        for key in keys
    }


def _fuse_rrf(
    bm25_rank: list[str], dense_rank: list[str], k: int = 60
) -> dict[str, float]:
    score_map: dict[str, float] = defaultdict(float)
    for rank, chunk_id in enumerate(bm25_rank):
        score_map[chunk_id] += 1.0 / (k + rank + 1)
    for rank, chunk_id in enumerate(dense_rank):
        score_map[chunk_id] += 1.0 / (k + rank + 1)
    return dict(score_map)


def _build_retrieval_cache_key(
    *,
    query: str,
    kb_ids: list[str],
    document_ids: list[str] | None,
    top_k: int,
    score_threshold: float,
    fusion_mode: str,
    alpha: float,
    agent_key: str | None,
) -> str:
    kb_key = ",".join(sorted(dict.fromkeys(kb_ids)))
    doc_key = ",".join(sorted(dict.fromkeys(document_ids or [])))
    return "|".join(
        [
            query.strip(),
            kb_key,
            doc_key,
            str(int(top_k)),
            f"{float(score_threshold):.4f}",
            fusion_mode.strip().lower(),
            f"{float(alpha):.4f}",
            normalize_agent_key(agent_key),
        ]
    )


def _retrieval_cache_get(cache_key: str) -> tuple[list[dict], list[dict]] | None:
    entry = _RETRIEVAL_CACHE.get(cache_key)
    if not entry:
        return None
    expires_at, cached_results, cached_debug_rows = entry
    if expires_at <= time.time():
        _RETRIEVAL_CACHE.pop(cache_key, None)
        return None
    return deepcopy(cached_results), deepcopy(cached_debug_rows)


def _retrieval_cache_put(
    cache_key: str,
    *,
    results: list[dict],
    debug_rows: list[dict],
) -> None:
    if len(_RETRIEVAL_CACHE) >= _RETRIEVAL_CACHE_MAX_SIZE:
        oldest_key = next(iter(_RETRIEVAL_CACHE))
        _RETRIEVAL_CACHE.pop(oldest_key, None)
    _RETRIEVAL_CACHE[cache_key] = (
        time.time() + _RETRIEVAL_CACHE_TTL_SECONDS,
        deepcopy(results),
        deepcopy(debug_rows),
    )


def hybrid_retrieve(
    db: Session,
    query: str,
    kb_ids: list[str],
    document_ids: list[str] | None,
    top_k: int,
    score_threshold: float,
    fusion_mode: str,
    alpha: float,
    agent_key: str | None = None,
) -> list[dict]:
    results, _ = hybrid_retrieve_with_debug(
        db=db,
        query=query,
        kb_ids=kb_ids,
        document_ids=document_ids,
        top_k=top_k,
        score_threshold=score_threshold,
        fusion_mode=fusion_mode,
        alpha=alpha,
        agent_key=agent_key,
    )
    return results


def hybrid_retrieve_with_debug(
    db: Session,
    query: str,
    kb_ids: list[str],
    document_ids: list[str] | None,
    top_k: int,
    score_threshold: float,
    fusion_mode: str,
    alpha: float,
    agent_key: str | None = None,
) -> tuple[list[dict], list[dict]]:
    if _is_small_talk(query):
        return [], []

    query_text = _expand_query_for_agent(agent_key, query)
    cache_key = _build_retrieval_cache_key(
        query=query_text,
        kb_ids=kb_ids,
        document_ids=document_ids,
        top_k=top_k,
        score_threshold=score_threshold,
        fusion_mode=fusion_mode,
        alpha=alpha,
        agent_key=agent_key,
    )
    cached_payload = _retrieval_cache_get(cache_key)
    if cached_payload is not None:
        return cached_payload

    unique_kb_ids = list(dict.fromkeys(kb_ids or []))
    if not unique_kb_ids:
        return [], []

    local_top_k = int(top_k)
    local_threshold = float(score_threshold)
    if _is_student_growth_agent(agent_key):
        local_top_k = min(local_top_k, 12)
        local_threshold = max(local_threshold, 0.22)
    precise_fact_query = _is_precise_fact_query(query_text)
    query_is_complex = _is_complex_retrieval_query(query_text)

    max_candidates = (
        min(320, max(56, int(local_top_k) * 20))
        if query_is_complex
        else min(220, max(36, int(local_top_k) * 14))
    )
    if precise_fact_query:
        max_candidates = min(max_candidates, 140)
    if document_ids:
        max_candidates = min(360, max(max_candidates, 80))

    stmt = select(DocumentChunk).where(DocumentChunk.kb_id.in_(unique_kb_ids))
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    stmt = stmt.order_by(desc(DocumentChunk.created_at)).limit(max_candidates)
    chunk_rows = list(db.scalars(stmt).all())

    chunk_rows = [
        row for row in chunk_rows if not _looks_like_schema_chunk(row.content)
    ]
    if _is_summary_style_query(query_text):
        chunk_rows = [
            row for row in chunk_rows if not _looks_like_command_chunk(row.content)
        ]
    if _is_summary_style_query(query_text) or _is_guidance_query(query_text):
        chunk_rows = [
            row for row in chunk_rows if not _looks_like_noise_chunk(row.content)
        ]
    chunk_rows = [row for row in chunk_rows if not _is_mojibake_text(row.content)]

    if _is_guidance_query(query_text) or _is_academic_query(query_text):
        q_tokens = set(_tokenize(query_text))
        if q_tokens:
            chunk_rows = sorted(
                chunk_rows,
                key=lambda row: _token_overlap_score(q_tokens, row.content),
                reverse=True,
            )[: max(64, local_top_k * 16)]

    if not chunk_rows:
        return [], []

    corpus = [row.content for row in chunk_rows]
    corpus_tokens = [_tokenize(text) for text in corpus]
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return [], []
    bm25 = BM25Okapi(corpus_tokens)
    bm25_raw_scores = bm25.get_scores(query_tokens)

    bm25_scores: dict[str, float] = {}
    bm25_norm: dict[str, float] = {}
    bm25_ranked: list[str] = []
    for idx, row in enumerate(chunk_rows):
        bm25_scores[row.id] = float(bm25_raw_scores[idx])
    bm25_ranked = [
        cid
        for cid, _ in sorted(
            bm25_scores.items(), key=lambda item: item[1], reverse=True
        )
    ][: local_top_k * 2]
    bm25_norm = _normalize_scores(bm25_scores)

    dense_scores: dict[str, float] = {}
    dense_norm: dict[str, float] = {}
    dense_ranked: list[str] = []
    chunk_cache = {row.id: row for row in chunk_rows}

    use_dense_search = not precise_fact_query
    if use_dense_search:
        kb_model_map = {
            item.id: item.embedding_model
            for item in db.scalars(
                select(KnowledgeBase).where(KnowledgeBase.id.in_(unique_kb_ids))
            ).all()
        }

        dense_top_k = (
            min(10, max(local_top_k + 1, local_top_k * 2 - 1))
            if query_is_complex
            else min(7, max(local_top_k, local_top_k + 1))
        )
        query_embedding_cache: dict[str, list[float]] = {}
        for kb_id in unique_kb_ids:
            model_name = normalize_embedding_model_name(
                kb_model_map.get(kb_id, settings.default_embedding_model)
            )
            query_embedding = query_embedding_cache.get(model_name)
            if query_embedding is None:
                query_embedding = embed_query(query=query_text, model_name=model_name)
                query_embedding_cache[model_name] = query_embedding

            for item in search_similar_chunks(
                kb_id=kb_id,
                query=query_text,
                top_k=dense_top_k,
                query_embedding=query_embedding,
            ):
                chunk_id = item["chunk_id"]
                if chunk_id in chunk_cache:
                    dense_scores[chunk_id] = max(
                        dense_scores.get(chunk_id, 0.0), float(item["score"])
                    )

    dense_ranked = [
        cid
        for cid, _ in sorted(
            dense_scores.items(), key=lambda item: item[1], reverse=True
        )
    ][: local_top_k * 2]
    dense_norm = _normalize_scores(dense_scores)

    if fusion_mode == "rrf":
        fused = _fuse_rrf(bm25_ranked, dense_ranked)
    else:
        fused = _fuse_weighted(bm25_scores, dense_scores, alpha=alpha)

    sorted_ids = [
        chunk_id
        for chunk_id, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)
    ][: max(local_top_k * 2 + 2, local_top_k)]

    results: list[dict] = []
    seen_content_keys: set[str] = set()
    source_counter: dict[str, int] = defaultdict(int)
    source_limit = 1 if precise_fact_query else 2
    for chunk_id in sorted_ids:
        row = chunk_cache[chunk_id]
        source_key = str(row.source_location or "").strip().lower()
        if source_key:
            if source_counter.get(source_key, 0) >= source_limit:
                continue
        compact_key = re.sub(r"\s+", " ", str(row.content or "").strip().lower())[:120]
        if compact_key and compact_key in seen_content_keys:
            continue
        if compact_key:
            seen_content_keys.add(compact_key)
        if source_key:
            source_counter[source_key] = source_counter.get(source_key, 0) + 1
        results.append(
            {
                "chunk_id": row.id,
                "document_id": row.document_id,
                "kb_id": row.kb_id,
                "content": row.content,
                "source_location": row.source_location,
                "bm25_raw": float(bm25_scores.get(chunk_id, 0.0)),
                "bm25_norm": float(bm25_norm.get(chunk_id, 0.0)),
                "dense_raw": float(dense_scores.get(chunk_id, 0.0)),
                "dense_norm": float(dense_norm.get(chunk_id, 0.0)),
                "fused_score": float(fused.get(chunk_id, 0.0)),
                "score_before_rerank": float(fused.get(chunk_id, 0.0)),
                "score": float(fused.get(chunk_id, 0.0)),
                "source_weight": _source_weight(
                    agent_key, row.source_location or "", query_text
                ),
            }
        )
        if len(results) >= max(local_top_k * 3, local_top_k + 4):
            break

    if _should_skip_reranker(
        query=query_text,
        top_k=local_top_k,
        candidate_count=len(results),
        precise_fact_query=precise_fact_query,
    ):
        reranked = list(results)
        for item in reranked:
            item["rerank_score"] = float(item.get("fused_score", 0.0))
    else:
        reranked = rerank_candidates(
            query=query_text,
            candidates=results,
            model_name=settings.reranker_model,
        )

    for item in reranked:
        base_score = float(item.get("score", 0.0)) + _keyword_bonus(
            query=query_text,
            source_location=str(item.get("source_location") or ""),
            content=str(item.get("content") or ""),
        )
        item["score"] = base_score * float(item.get("source_weight", 1.0))

    reranked = sorted(
        reranked, key=lambda it: float(it.get("score", 0.0)), reverse=True
    )

    debug_rows = [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "source_location": item["source_location"],
            "bm25_raw": round(float(item.get("bm25_raw", 0.0)), 6),
            "bm25_norm": round(float(item.get("bm25_norm", 0.0)), 6),
            "dense_raw": round(float(item.get("dense_raw", 0.0)), 6),
            "dense_norm": round(float(item.get("dense_norm", 0.0)), 6),
            "fused_score": round(float(item.get("fused_score", 0.0)), 6),
            "rerank_score": round(float(item.get("rerank_score", 0.0)), 6),
            "final_score": round(float(item.get("score", 0.0)), 6),
            "source_weight": round(float(item.get("source_weight", 1.0)), 4),
            "content": item.get("content", ""),
        }
        for item in reranked
    ]

    filtered = [
        item for item in reranked if float(item.get("score", 0.0)) >= local_threshold
    ]
    result_payload = filtered[:local_top_k]
    _retrieval_cache_put(
        cache_key,
        results=result_payload,
        debug_rows=debug_rows,
    )
    return result_payload, debug_rows
