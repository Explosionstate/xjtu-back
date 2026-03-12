from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sqlalchemy import select
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
) -> tuple[list[dict], list[dict]]:
    stmt = select(DocumentChunk).where(DocumentChunk.kb_id.in_(kb_ids))
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    chunk_rows = list(db.scalars(stmt).all())
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
        )[: top_k * 2]
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
            top_k=top_k * 2,
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
        )[: top_k * 2]
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
    ][: max(top_k * 3, top_k)]

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
            }
        )

    reranked = rerank_candidates(
        query=query,
        candidates=results,
        model_name=settings.reranker_model,
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
            "content": item.get("content", ""),
        }
        for item in reranked
    ]
    filtered = [
        item for item in reranked if float(item.get("score", 0.0)) >= score_threshold
    ]
    return filtered[:top_k], debug_rows
