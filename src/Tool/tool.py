from typing import List, Dict
from src.Indexing.Index import get_vector_store

RETRIEVE_K = 30
TOP_K = 10



def retrieve_documents(query: str, retrieve_k: int = RETRIEVE_K, top_k: int = TOP_K) -> List:
    """
    Shared retrieval path. Cross-encoder rerank removed — Render free tier's
    512MB memory limit can't fit embeddings + cross-encoder + torch overhead
    simultaneously. Falls back to plain FAISS similarity ranking.
    """
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query.strip(), k=retrieve_k)
    return docs[:top_k]


def _format_result(doc) -> Dict:
    meta = doc.metadata
    return {
        "id": meta.get("id"),
        "name": meta.get("name"),
        "url": meta.get("url"),
        "test_type": meta.get("test_type"),
        "description": meta.get("description", ""),
        "duration": meta.get("duration", ""),
        "languages": meta.get("languages", []),
        "job_levels": meta.get("job_levels", []),
        "remote_testing": meta.get("remote_testing", False),
        "adaptive_irt": meta.get("adaptive_irt", False),
    }


def _build_name_index(vector_store) -> Dict[str, object]:
    index = {}
    docstore = vector_store.docstore._dict
    for doc in docstore.values():
        name = doc.metadata.get("name", "")
        if name:
            index[name.lower()] = doc
    return index


_name_index_cache = None


def compare_assessments_lookup(names: List[str]) -> List[Dict]:
    global _name_index_cache
    vector_store = get_vector_store()

    if _name_index_cache is None:
        _name_index_cache = _build_name_index(vector_store)

    results = []
    for name in names:
        name_lower = name.lower()
        doc = _name_index_cache.get(name_lower)

        if doc is None:
            candidates = [
                d for n, d in _name_index_cache.items() if name_lower in n or n in name_lower
            ]
            doc = candidates[0] if candidates else None

        if doc is None:
            hits = vector_store.similarity_search(name, k=1)
            doc = hits[0] if hits else None

        if doc:
            results.append(_format_result(doc))

    return results