import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json
# --------------------------------------------------
# Constants (repo-root relative paths)
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FAISS_PATH = os.path.join(PROJECT_ROOT, "data", "faiss_index")

_vector_store = None


def get_vector_store():
    """
    Load FAISS index from disk.
    This function MUST NOT create or modify the index.
    """

    global _vector_store

    # Return cached instance (important for tools)
    if _vector_store is not None:
        return _vector_store

    try:
        import hashlib
        marker_path = os.path.join(FAISS_PATH, "_build_info.json")
        assessments_path = os.path.join(
            PROJECT_ROOT, "src", "Indexing", "final_assessments.json"
        )
        with open(marker_path) as f:
            marker = json.load(f)
        with open(assessments_path, "rb") as f:
            current_hash = hashlib.md5(f.read()).hexdigest()[:8]
        if marker.get("source_hash") != current_hash:
            print(
                "⚠️  WARNING: FAISS index may be stale — final_assessments.json "
                "has changed since the index was last built. Run build_index.py again."
            )
    except Exception:
        pass  # marker missing or unreadable — don't block startup over this

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

    # Warn if the index looks stale relative to the current source file
