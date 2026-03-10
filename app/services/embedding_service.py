from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.core.errors import BusinessError

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]


def _register_local_modules() -> None:
    if settings.local_modules_root and settings.local_modules_root.exists():
        path_value = str(settings.local_modules_root)
        if path_value not in sys.path:
            sys.path.insert(0, path_value)


def _candidate_paths(model_name: str) -> list[Path]:
    candidates: list[Path] = []
    value = Path(model_name)
    if value.exists():
        candidates.append(value)

    if settings.embedding_model_root:
        candidates.append(settings.embedding_model_root / model_name)
        normalized = model_name.replace("/", "--")
        candidates.append(settings.embedding_model_root / normalized)

    if settings.local_modules_root:
        candidates.append(settings.local_modules_root / model_name)
        candidates.append(settings.local_modules_root / "models" / model_name)
    return candidates


def resolve_model_reference(model_name: str) -> str:
    for item in _candidate_paths(model_name):
        if item.exists():
            return str(item)
    return model_name


@lru_cache(maxsize=8)
def _get_embedder(model_reference: str) -> SentenceTransformer:
    _register_local_modules()
    if SentenceTransformer is None:
        raise BusinessError(
            "未安装 sentence-transformers，无法加载本地 embedding 模型", status_code=500
        )
    return SentenceTransformer(model_reference)


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    if not texts:
        return []
    reference = resolve_model_reference(model_name)
    embedder = _get_embedder(reference)
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return [vec.tolist() for vec in vectors]


def embed_query(query: str, model_name: str) -> list[float]:
    values = embed_texts([query], model_name=model_name)
    if not values:
        return []
    return values[0]
