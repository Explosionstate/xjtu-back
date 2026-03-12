from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

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
        candidates.append(settings.embedding_model_root / f"models--{normalized}")

    if settings.local_modules_root:
        candidates.append(settings.local_modules_root / model_name)
        candidates.append(settings.local_modules_root / "models" / model_name)
    if settings.local_models_root:
        normalized = model_name.replace("/", "--")
        candidates.append(settings.local_models_root / f"models--{normalized}")
    return candidates


def _resolve_hf_cache_snapshot(root: Path) -> Path | None:
    refs_main = root / "refs" / "main"
    if refs_main.exists():
        snapshot_id = refs_main.read_text(encoding="utf-8", errors="ignore").strip()
        if snapshot_id:
            snap_dir = root / "snapshots" / snapshot_id
            if snap_dir.exists():
                return snap_dir
    snapshots_dir = root / "snapshots"
    if snapshots_dir.exists():
        dirs = [item for item in snapshots_dir.iterdir() if item.is_dir()]
        if dirs:
            dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
            return dirs[0]
    return None


def resolve_model_reference(model_name: str) -> str:
    for item in _candidate_paths(model_name):
        if item.exists():
            if item.is_dir() and item.name.startswith("models--"):
                snapshot = _resolve_hf_cache_snapshot(item)
                if snapshot is not None:
                    return str(snapshot)
            return str(item)
    return model_name


def normalize_embedding_model_name(model_name: str | None) -> str:
    value = (model_name or "").strip()
    if not value or value.lower() in {"string", "none", "null", "default"}:
        return settings.default_embedding_model
    return value


def resolve_local_model_reference(model_name: str | None) -> str:
    normalized = normalize_embedding_model_name(model_name)
    reference = resolve_model_reference(normalized)
    if Path(reference).exists():
        return reference

    # Keep deployment offline/local-first: do not auto-download from HuggingFace.
    fallback_reference = resolve_model_reference(settings.default_embedding_model)
    if Path(fallback_reference).exists():
        return fallback_reference

    raise BusinessError(
        (
            f"未找到本地Embedding模型: {normalized}。"
            "请确认模型位于 D:/xjtu/local_models 或更新 app/core/config.py 固定模型配置。"
        ),
        status_code=500,
    )


@lru_cache(maxsize=8)
def _get_embedder(model_reference: str) -> Any:
    _register_local_modules()
    if SentenceTransformer is None:
        raise BusinessError(
            "未安装 sentence-transformers，无法加载本地 embedding 模型", status_code=500
        )
    return SentenceTransformer(model_reference)


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    if not texts:
        return []
    reference = resolve_local_model_reference(model_name)
    embedder = _get_embedder(reference)
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return [vec.tolist() for vec in vectors]


def embed_query(query: str, model_name: str) -> list[float]:
    values = embed_texts([query], model_name=model_name)
    if not values:
        return []
    return values[0]
