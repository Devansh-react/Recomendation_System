import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAISS_PATH = os.path.join(PROJECT_ROOT, "data", "faiss_index")

_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    if not os.path.exists(FAISS_PATH):
        raise RuntimeError(f"FAISS index not found at '{FAISS_PATH}'. Build it first.")

    print("✅ Loading FAISS index from disk...")
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    )
    _vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return _vector_store