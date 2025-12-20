from langchain.tools import tool
from typing import List, Dict

# import your vector store & embedding utils
from Indexing.Indexing import get_vector_store # FAISS / Chroma


@tool
def rag_retrieve(query: str) -> List[Dict]:
    """
    Retrieve similar SHL assessments for a user query.
    """
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=10)

    results = []
    for doc in docs:
        meta = doc.metadata
        results.append({
            "id": meta["id"],
            "name": meta["name"],
            "url": meta["url"],
            "test_type": meta["test_type"]
        })

    return results
