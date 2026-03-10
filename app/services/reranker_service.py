from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.embedding_service import (
    embed_query,
    embed_texts,
    resolve_model_reference,
)

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]


@lru_cache(maxsize=2)
def _get_cross_encoder(model_reference: str) -> Any:
    if CrossEncoder is None:
        return None
    try:
        return CrossEncoder(model_reference)
    except Exception:
        return None


def _cosine_fallback(query: str, contents: list[str], model_name: str) -> list[float]:
    query_vec = embed_query(query=query, model_name=model_name)
    if not query_vec:
        return [0.0 for _ in contents]
    content_vecs = embed_texts(contents, model_name=model_name)
    if not content_vecs:
        return [0.0 for _ in contents]
    scores: list[float] = []
    for vec in content_vecs:
        score = sum(a * b for a, b in zip(query_vec, vec, strict=False))
        scores.append(float(score))
    return scores


def rerank_candidates(
    query: str, candidates: list[dict], model_name: str
) -> list[dict]:
    if not candidates or not settings.reranker_enabled:
        return candidates

    top_n = max(1, min(len(candidates), settings.reranker_top_n))
    head = candidates[:top_n]
    tail = candidates[top_n:]
    contents = [item["content"] for item in head]

    reference = resolve_model_reference(model_name)
    cross_encoder = _get_cross_encoder(reference) if Path(reference).exists() else None
    if cross_encoder is not None:
        pairs = [[query, text] for text in contents]
        rerank_scores = [float(v) for v in cross_encoder.predict(pairs)]
    else:
        rerank_scores = _cosine_fallback(
            query=query,
            contents=contents,
            model_name=settings.default_embedding_model,
        )

    min_score = min(rerank_scores) if rerank_scores else 0.0
    max_score = max(rerank_scores) if rerank_scores else 1.0
    denom = max(max_score - min_score, 1e-8)

    for idx, item in enumerate(head):
        rerank_norm = (rerank_scores[idx] - min_score) / denom
        base = float(item.get("score", 0.0))
        item["rerank_score"] = rerank_norm
        item["score"] = (
            settings.reranker_weight * rerank_norm
            + (1.0 - settings.reranker_weight) * base
        )

    head.sort(key=lambda it: float(it.get("score", 0.0)), reverse=True)
    return head + tail
