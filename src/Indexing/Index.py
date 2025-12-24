import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --------------------------------------------------
# Constants (repo-root relative paths)
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FAISS_PATH = os.path.join(PROJECT_ROOT, "data", "faiss_index")



# --------------------------------------------------
# Singleton vector store (cached per process)
# --------------------------------------------------
_vector_store = None


def get_vector_store():
    """
    Load FAISS index from disk.
    This function MUST NOT create or modify the index.
    """

    global _vector_store

    # ✅ Return cached instance (important for tools)
    if _vector_store is not None:
        return _vector_store

    # ❌ Do NOT auto-create index at runtime
    if not os.path.exists(FAISS_PATH):
        raise RuntimeError(
            f"FAISS index not found at '{FAISS_PATH}'. "
            "Build the index offline using build_index.py before running the API."
        )

    print("✅ Loading FAISS index from disk...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    _vector_store = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return _vector_store
