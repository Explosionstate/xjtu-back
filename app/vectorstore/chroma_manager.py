from __future__ import annotations

import gc
import logging
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any, cast

from app.core.config import settings

# Reduce noisy telemetry errors in local/offline environments.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault(
    "CHROMA_TELEMETRY_IMPL",
    "app.vectorstore.chroma_noop_telemetry.NoopProductTelemetryClient",
)
os.environ.setdefault(
    "CHROMA_PRODUCT_TELEMETRY_IMPL",
    "app.vectorstore.chroma_noop_telemetry.NoopProductTelemetryClient",
)

try:
    import chromadb
    from chromadb.config import Settings as ChromaClientSettings
except ImportError:
    chromadb = None
    ChromaClientSettings = None


logger = logging.getLogger(__name__)


def _create_client(path: Path):
    if chromadb is None:
        return None
    if ChromaClientSettings is not None:
        return chromadb.PersistentClient(
            path=str(path),
            settings=ChromaClientSettings(
                anonymized_telemetry=False,
            ),
        )
    return chromadb.PersistentClient(path=str(path))


def kb_vectorstore_path(kb_id: str) -> Path:
    return settings.chroma_root / kb_id


def ensure_kb_vectorstore(kb_id: str) -> None:
    target_dir = kb_vectorstore_path(kb_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    if chromadb is not None:
        client = _create_client(target_dir)
        if client is None:
            return
        client.get_or_create_collection(name="documents")


def _get_collection(kb_id: str):
    try:
        client = _create_client(kb_vectorstore_path(kb_id))
        if client is None:
            return None
        return client.get_or_create_collection(name="documents")
    except Exception as exc:
        logger.warning("chroma open collection failed for kb %s: %s", kb_id, exc)
        return None


def upsert_chunks(
    kb_id: str,
    chunk_ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]] | None = None,
) -> bool:
    collection = _get_collection(kb_id)
    if collection is None or not chunk_ids:
        return False
    payload: dict[str, Any] = {
        "ids": chunk_ids,
        "documents": texts,
        "metadatas": cast(Any, metadatas),
    }
    if embeddings is not None:
        payload["embeddings"] = embeddings
    try:
        collection.upsert(**payload)
        return True
    except Exception as exc:
        logger.warning("chroma upsert failed for kb %s: %s", kb_id, exc)
        return False


def delete_chunks_by_document(kb_id: str, document_id: str) -> None:
    collection = _get_collection(kb_id)
    if collection is None:
        return
    try:
        collection.delete(where={"document_id": document_id})
    except Exception as exc:
        # Keep document deletion path available even if local Chroma metadata
        # is temporarily inconsistent (common in Windows dev envs).
        logger.warning(
            "chroma delete failed for kb %s doc %s: %s",
            kb_id,
            document_id,
            exc,
        )


def search_similar_chunks(
    kb_id: str,
    query: str,
    top_k: int,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    collection = _get_collection(kb_id)
    if collection is None:
        return []
    max_results = top_k
    try:
        current_count = int(collection.count())
        if current_count <= 0:
            return []
        max_results = max(1, min(top_k, current_count))
    except Exception:
        pass

    query_payload: dict[str, Any] = {"n_results": max_results}
    if query_embedding is not None:
        query_payload["query_embeddings"] = [query_embedding]
    else:
        query_payload["query_texts"] = [query]
    try:
        result = collection.query(**query_payload)
    except Exception as exc:
        # Chroma metadata can be temporarily inconsistent on Windows/local dev.
        # Keep chat path available by degrading to BM25-only retrieval.
        logger.warning("chroma query failed for kb %s: %s", kb_id, exc)
        return []
    ids_all = result.get("ids") or [[]]
    docs_all = result.get("documents") or [[]]
    distances_all = result.get("distances") or [[]]
    metadatas_all = result.get("metadatas") or [[]]
    ids = ids_all[0]
    docs = docs_all[0]
    distances = distances_all[0]
    metadatas = metadatas_all[0]
    items: list[dict[str, Any]] = []
    for idx, chunk_id in enumerate(ids):
        dist = float(distances[idx]) if idx < len(distances) else 1.0
        score = max(0.0, 1.0 - dist)
        items.append(
            {
                "chunk_id": chunk_id,
                "content": docs[idx] if idx < len(docs) else "",
                "score": score,
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
            }
        )
    return items


def delete_kb_vectorstore(kb_id: str, raise_on_failure: bool = True) -> bool:
    target_dir = kb_vectorstore_path(kb_id)
    if not target_dir.exists():
        return True

    def _onerror(func, path, exc_info):
        # Windows may keep files readonly/locked briefly.
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    last_error: Exception | None = None
    for _ in range(6):
        try:
            shutil.rmtree(target_dir, onerror=_onerror)
            return True
        except PermissionError as exc:
            last_error = exc
            gc.collect()
            time.sleep(0.2)

    if last_error is not None and raise_on_failure:
        raise last_error
    return False


def clone_kb_vectorstore(source_kb_id: str, target_kb_id: str) -> None:
    source_dir = kb_vectorstore_path(source_kb_id)
    target_dir = kb_vectorstore_path(target_kb_id)

    if target_dir.exists():
        shutil.rmtree(target_dir)

    if source_dir.exists():
        shutil.copytree(source_dir, target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
