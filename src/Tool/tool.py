from langchain.tools import tool
from typing import List, Dict
from src.Indexing.Index import get_vector_store


def rerank_by_query_overlap(query: str, docs: List) -> List:
    query_tokens = set(query.lower().split())

    def overlap_score(doc):
        text = doc.page_content.lower()
        return sum(1 for token in query_tokens if token in text)

    return sorted(docs, key=overlap_score, reverse=True)


@tool
def rag_retrieve(query: str) -> List[Dict]:
    """
    Retrieve SHL assessments for a query using
    embedding search + lightweight re-ranking.
    """

    vector_store = get_vector_store()

    # Step 1: Retrieve MORE candidates (critical for recall)
    docs = vector_store.similarity_search(query, k=30)

    # Step 2: Re-rank using query overlap
    docs = rerank_by_query_overlap(query, docs)

    # Step 3: Select top 10
    docs = docs[:10]

    # Step 4: Format output
    results = []
    for doc in docs:
        meta = doc.metadata
        results.append({
            "id": meta.get("id"),
            "name": meta.get("name"),
            "url": meta.get("url"),
            "test_type": meta.get("test_type")
        })

    return results
