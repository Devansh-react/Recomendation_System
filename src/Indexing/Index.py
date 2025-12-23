import os
import json
import faiss
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from Schema.Schema_index import SHLAssessment

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")



# Embeddings (init once per process)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

_vector_store = None


def get_vector_store():
    """
    Loads FAISS index if it exists.
    Otherwise, creates embeddings and saves FAISS index.
    """
    global _vector_store

    # ✅ Return cached instance (important for tools)
    if _vector_store is not None:
        return _vector_store

    # ✅ Load existing FAISS index if present
    if os.path.exists(FAISS_PATH):
        print("🔁 Loading FAISS index from disk...")
        _vector_store = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return _vector_store

    print("⚙️ FAISS index not found. Creating a new one...")

    # ❗ Kept EXACTLY as you requested
    with open(os.path.join(BASE_DIR, "final_assessments.json"), "r", encoding="utf-8") as f:
        raw_data = json.load(f)


    assessments = [SHLAssessment(**item) for item in raw_data]

    documents = [
        Document(
            page_content=a.text_for_embedding,
            metadata={
                "id": a.id,
                "name": a.name,
                "url": a.url,
                "test_type": a.test_type
            }
        )
        for a in assessments
    ]

    # ✅ Correct way to create FAISS index
    embedding_dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(documents)
    vector_store.save_local(FAISS_PATH)

    print(f"✅ FAISS index created with {len(documents)} documents")

    _vector_store = vector_store
    return _vector_store
