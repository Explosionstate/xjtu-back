from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models.document import DocumentChunk
from app.models.knowledge_base import KnowledgeBase
from app.core.config import settings
from app.services.embedding_service import embed_query, normalize_embedding_model_name
from app.services.reranker_service import rerank_candidates
from app.vectorstore.chroma_manager import search_similar_chunks


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
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


def _is_student_growth_agent(agent_key: str | None) -> bool:
    key = (agent_key or "").strip().lower()
    return key in {"student-growth", "student_growth", "student"}


def _is_mojibake_text(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    replacement_ratio = text.count("�") / max(1, len(text))
    if replacement_ratio >= 0.03:
        return True
    mojibake_markers = ["Ã", "Â", "ä¸", "ç", "å", "æ", "ï¼", "é"]
    marker_hits = sum(text.count(marker) for marker in mojibake_markers)
    return marker_hits >= 12 and marker_hits / max(1, len(text)) > 0.05


def _source_weight(agent_key: str | None, source_location: str, query: str) -> float:
    if not _is_student_growth_agent(agent_key):
        return 1.0

    src = (source_location or "").lower()
    q = (query or "").lower()
    positive = ["学生", "学业", "成长", "指南", "手册", "学习", "预警", "辅导"]
    negative = [
        "xjtu-back",
        "xjtu-front",
        "readme",
        "ops.py",
        "接口",
        "部署",
        "开发",
        "脚本",
        "api",
    ]

    weight = 1.0
    if any(key in q for key in ["学业", "学生", "学习", "成长", "指南", "建议"]):
        if any(token in src for token in positive):
            weight *= 1.22
        if any(token in src for token in negative):
            weight *= 0.35
    return max(0.2, min(1.4, weight))


def _is_academic_query(query: str) -> bool:
    q = (query or "").strip().lower()
    flags = ["学业", "成绩", "课堂", "学习", "课程", "预警"]
    return any(flag in q for flag in flags)


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

    local_top_k = int(top_k)
    local_threshold = float(score_threshold)
    if _is_student_growth_agent(agent_key):
        local_top_k = min(local_top_k, 5)
        local_threshold = max(local_threshold, 0.28)
    max_candidates = max(120, int(local_top_k) * 60)
    stmt = select(DocumentChunk).where(DocumentChunk.kb_id.in_(kb_ids))
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    stmt = stmt.order_by(desc(DocumentChunk.created_at)).limit(max_candidates)
    chunk_rows = list(db.scalars(stmt).all())
    chunk_rows = [
        row for row in chunk_rows if not _looks_like_schema_chunk(row.content)
    ]
    if _is_summary_style_query(query):
        chunk_rows = [
            row for row in chunk_rows if not _looks_like_command_chunk(row.content)
        ]
    if _is_summary_style_query(query) or _is_guidance_query(query):
        chunk_rows = [
            row for row in chunk_rows if not _looks_like_noise_chunk(row.content)
        ]
    chunk_rows = [row for row in chunk_rows if not _is_mojibake_text(row.content)]

    if _is_guidance_query(query) or _is_academic_query(query):
        q_tokens = set(_tokenize(query))
        if q_tokens:
            chunk_rows = sorted(
                chunk_rows,
                key=lambda row: _token_overlap_score(q_tokens, row.content),
                reverse=True,
            )[: max(100, local_top_k * 20)]
    if not chunk_rows:
        return [], []

    corpus = [row.content for row in chunk_rows]
    corpus_tokens = [_tokenize(text) for text in corpus]
    query_tokens = _tokenize(query)
    bm25 = BM25Okapi(corpus_tokens)
    bm25_raw_scores = bm25.get_scores(query_tokens)

    bm25_scores: dict[str, float] = {}
    bm25_norm: dict[str, float] = {}
    bm25_ranked: list[str] = []
    for idx, row in enumerate(chunk_rows):
        score = float(bm25_raw_scores[idx])
        bm25_scores[row.id] = score
    bm25_ranked = [
        cid
        for cid, _ in sorted(
            bm25_scores.items(), key=lambda item: item[1], reverse=True
        )[: local_top_k * 2]
    ]
    bm25_norm = _normalize_scores(bm25_scores)

    dense_scores: dict[str, float] = {}
    dense_norm: dict[str, float] = {}
    dense_ranked: list[str] = []
    chunk_cache = {row.id: row for row in chunk_rows}
    kb_model_map = {
        item.id: item.embedding_model
        for item in db.scalars(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(kb_ids))
        ).all()
    }
    for kb_id in kb_ids:
        model_name = normalize_embedding_model_name(
            kb_model_map.get(kb_id, settings.default_embedding_model)
        )
        query_embedding = embed_query(query=query, model_name=model_name)
        for item in search_similar_chunks(
            kb_id=kb_id,
            query=query,
            top_k=local_top_k * 2,
            query_embedding=query_embedding,
        ):
            chunk_id = item["chunk_id"]
            if chunk_id in chunk_cache:
                dense_scores[chunk_id] = max(
                    dense_scores.get(chunk_id, 0.0), item["score"]
                )
    dense_ranked = [
        cid
        for cid, _ in sorted(
            dense_scores.items(), key=lambda item: item[1], reverse=True
        )[: local_top_k * 2]
    ]
    dense_norm = _normalize_scores(dense_scores)

    if fusion_mode == "rrf":
        fused = _fuse_rrf(bm25_ranked, dense_ranked)
    else:
        fused = _fuse_weighted(bm25_scores, dense_scores, alpha=alpha)

    sorted_ids = [
        chunk_id
        for chunk_id, score in sorted(
            fused.items(), key=lambda item: item[1], reverse=True
        )
    ][: max(local_top_k * 3, local_top_k)]

    results: list[dict] = []
    for chunk_id in sorted_ids:
        row = chunk_cache[chunk_id]
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
                    agent_key, row.source_location or "", query
                ),
            }
        )

    reranked = rerank_candidates(
        query=query,
        candidates=results,
        model_name=settings.reranker_model,
    )

    for item in reranked:
        base_score = float(item.get("score", 0.0)) + _keyword_bonus(
            query=query,
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
    return filtered[:local_top_k], debug_rows
