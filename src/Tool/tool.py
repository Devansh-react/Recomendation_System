from langchain.tools import tool
from typing import List, Dict
from sentence_transformers import CrossEncoder
from src.Indexing.Index import get_vector_store

RETRIEVE_K = 50   # wide candidate pool from embedding search
RERANK_K = 25     # how many of those go through the (slower) cross-encoder
TOP_K = 10        # final returned count

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------
# Cross-encoder reranker — loaded once, module-level
# --------------------------------------------------
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print("Loading cross-encoder reranker...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank_cross_encoder(query: str, docs: List, top_k: int = TOP_K) -> List:
    """
    Score each (query, doc) pair jointly with a cross-encoder, which is far
    more accurate than embedding similarity alone or token-overlap counting —
    it actually reads both texts together rather than comparing precomputed
    vectors or counting shared words.
    """
    if not docs:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_k]]


def retrieve_documents(query: str, retrieve_k: int = RETRIEVE_K,
                        rerank_k: int = RERANK_K, top_k: int = TOP_K) -> List:
    """
    Shared retrieval path used by the tool and batch submission.

    Pipeline: wide embedding recall (k=retrieve_k) -> cross-encoder rerank
    on the top rerank_k candidates -> final top_k.
    """
    vector_store = get_vector_store()
    query = query.strip()
    prefixed_query = BGE_QUERY_PREFIX + query

    # Stage 1: dense retrieval, wide net
    docs = vector_store.similarity_search(prefixed_query, k=retrieve_k)

    # Stage 2: cross-encoder rerank on the top rerank_k of those
    # (reranking all 50 is unnecessary cost; the true positive is almost
    # always already within the top 25 dense-retrieved candidates)
    candidates = docs[:rerank_k]
    reranked = rerank_cross_encoder(query, candidates, top_k=top_k)

    return reranked


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
    Retrieve SHL assessments for a query using embedding search + cross-encoder
    reranking. Use this when the user wants recommendations/a shortlist, or
    when refining an existing shortlist with new constraints.
    """
    docs = retrieve_documents(query)
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