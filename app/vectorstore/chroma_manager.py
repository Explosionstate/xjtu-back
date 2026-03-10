from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.core.config import settings

try:
    import chromadb
except ImportError:
    chromadb = None


def kb_vectorstore_path(kb_id: str) -> Path:
    return settings.chroma_root / kb_id


def ensure_kb_vectorstore(kb_id: str) -> None:
    target_dir = kb_vectorstore_path(kb_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    if chromadb is not None:
        client = chromadb.PersistentClient(path=str(target_dir))
        client.get_or_create_collection(name="documents")


def _get_collection(kb_id: str):
    if chromadb is None:
        return None
    client = chromadb.PersistentClient(path=str(kb_vectorstore_path(kb_id)))
    return client.get_or_create_collection(name="documents")


def upsert_chunks(
    kb_id: str,
    chunk_ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    collection = _get_collection(kb_id)
    if collection is None or not chunk_ids:
        return
    collection.upsert(ids=chunk_ids, documents=texts, metadatas=metadatas)


def delete_chunks_by_document(kb_id: str, document_id: str) -> None:
    collection = _get_collection(kb_id)
    if collection is None:
        return
    collection.delete(where={"document_id": document_id})


def search_similar_chunks(kb_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
    collection = _get_collection(kb_id)
    if collection is None:
        return []
    result = collection.query(query_texts=[query], n_results=top_k)
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
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


def delete_kb_vectorstore(kb_id: str) -> None:
    target_dir = kb_vectorstore_path(kb_id)
    if target_dir.exists():
        shutil.rmtree(target_dir)


def clone_kb_vectorstore(source_kb_id: str, target_kb_id: str) -> None:
    source_dir = kb_vectorstore_path(source_kb_id)
    target_dir = kb_vectorstore_path(target_kb_id)

    if target_dir.exists():
        shutil.rmtree(target_dir)

    if source_dir.exists():
        shutil.copytree(source_dir, target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
