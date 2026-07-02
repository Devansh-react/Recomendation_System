from langchain.tools import tool
from typing import List, Dict
from src.Indexing.Index import get_vector_store


def rerank_by_query_overlap(query: str, docs: List) -> List:
    query_tokens = set(query.lower().split())

    def overlap_score(doc):
        text = doc.page_content.lower()
        return sum(1 for token in query_tokens if token in text)

    return sorted(docs, key=overlap_score, reverse=True)


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


@tool
def rag_retrieve(query: str) -> List[Dict]:
    """
    Retrieve SHL assessments for a query using embedding search + re-ranking.
    Use this when the user wants recommendations/a shortlist, or when refining
    an existing shortlist with new constraints.
    """
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=50)
    docs = rerank_by_query_overlap(query, docs)
    docs = docs[:10]

    return [_format_result(doc) for doc in docs]


def _build_name_index(vector_store) -> Dict[str, object]:
    """
    Build an exact-match name -> Document index from the FAISS docstore.
    Cached at module level so this only runs once per process.
    """
    index = {}
    docstore = vector_store.docstore._dict  # internal, but stable in langchain-community FAISS
    for doc in docstore.values():
        name = doc.metadata.get("name", "")
        if name:
            index[name.lower()] = doc
    return index


_name_index_cache = None


@tool
def compare_assessments(names: List[str]) -> List[Dict]:
    """
    Look up specific SHL assessments by exact or near-exact name for a
    grounded comparison. Use this when the user asks to compare two or
    more named assessments (e.g. "difference between OPQ and GSA").
    Do NOT use rag_retrieve for this — semantic search can return the
    wrong specific product when names are similar (e.g. "OPQ32r" vs
    "OPQ32i").
    """
    global _name_index_cache
    vector_store = get_vector_store()

    if _name_index_cache is None:
        _name_index_cache = _build_name_index(vector_store)

    results = []
    for name in names:
        name_lower = name.lower()

        # 1. Exact match first
        doc = _name_index_cache.get(name_lower)

        # 2. Fallback: substring match against indexed names
        if doc is None:
            candidates = [
                d for n, d in _name_index_cache.items() if name_lower in n or n in name_lower
            ]
            doc = candidates[0] if candidates else None

        # 3. Last resort: semantic search
        if doc is None:
            hits = vector_store.similarity_search(name, k=1)
            doc = hits[0] if hits else None

        if doc:
            results.append(_format_result(doc))

    return results