from langchain.tools import tool
from typing import List, Dict
from collections import defaultdict
from Indexing.Indexing import get_vector_store


# ------------------------------------------------------------
# 1. Utility: normalize test_type metadata
# ------------------------------------------------------------

def normalize_test_types(test_types):
    if not test_types:
        return []

    if isinstance(test_types, str):
        return [test_types.strip().lower()]

    return [t.strip().lower() for t in test_types]


# ------------------------------------------------------------
# 2. Dynamic domain-balanced reranking (NO hard coding)
# ------------------------------------------------------------

def rerank_with_domain_balance(docs, k=10):
    """
    Dynamically balances results across test_type domains
    inferred from catalog metadata.
    """

    # Bucket documents by test_type domain
    domain_buckets = defaultdict(list)

    for doc in docs:
        test_types = normalize_test_types(doc.metadata.get("test_type"))

        # If no test_type exists, treat as generic
        if not test_types:
            domain_buckets["unknown"].append(doc)
        else:
            for t in test_types:
                domain_buckets[t].append(doc)

    # Round-robin selection across domains
    final_docs = []
    pointers = {domain: 0 for domain in domain_buckets}

    while len(final_docs) < k:
        added = False

        for domain, bucket in domain_buckets.items():
            idx = pointers[domain]
            if idx < len(bucket):
                final_docs.append(bucket[idx])
                pointers[domain] += 1
                added = True
                if len(final_docs) == k:
                    break

        # Stop if nothing more can be added
        if not added:
            break

    return final_docs


# ------------------------------------------------------------
# 3. LangChain tool used by main.py / agent
# ------------------------------------------------------------

@tool
def rag_retrieve(query: str) -> List[Dict]:
    """
    Retrieve SHL assessments for a query with
    dynamic domain-balanced recommendations.
    """

    vector_store = get_vector_store()

    # Step 1: Retrieve more candidates than needed
    docs = vector_store.similarity_search(query, k=20)

    # Step 2: Apply dynamic domain balancing
    docs = rerank_with_domain_balance(docs, k=10)

    # Step 3: Format output
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
