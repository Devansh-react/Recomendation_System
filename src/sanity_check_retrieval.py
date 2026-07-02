# rank_check.py — run from repo root
from Indexing.Index import get_vector_store

vs = get_vector_store()
query = "senior data analyst SQL Excel Python"

docs = vs.similarity_search_with_score(query, k=30)

for rank, (doc, score) in enumerate(docs, 1):
    marker = "  <-- HERE" if "python" in doc.metadata["name"].lower() else ""
    print(f"{rank:2d}. {doc.metadata['name']:50s} score={score:.4f}{marker}")