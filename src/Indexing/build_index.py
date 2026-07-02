import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --------------------------------------------------
# Paths — MUST match src/Indexing/Index.py exactly
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
FAISS_PATH = os.path.join(PROJECT_ROOT, "data", "faiss_index")
ASSESSMENTS_PATH = os.path.join(
    PROJECT_ROOT, "src", "Indexing", "final_assessments.json"
)

# --------------------------------------------------
# Safety Checks
# --------------------------------------------------
if not os.path.exists(ASSESSMENTS_PATH):
    raise FileNotFoundError(f"Input data file not found: {ASSESSMENTS_PATH}")

os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

# --------------------------------------------------
# Load assessment data
# --------------------------------------------------
with open(ASSESSMENTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📄 Loaded {len(data)} assessments")

# --------------------------------------------------
# Convert to LangChain Documents
# --------------------------------------------------
documents = [
    Document(
        page_content=item["text_for_embedding"],
        metadata={
            "id": item["id"],
            "name": item["name"],
            "url": item["url"],
            "test_type": item["test_type"],
            "description": item.get("description", ""),
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration": item.get("duration", ""),
            "remote_testing": item.get("remote_testing", False),
            "adaptive_irt": item.get("adaptive_irt", False),
        },
    )
    for item in data
]

# --------------------------------------------------
# Create embeddings + build index
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
)

print("⚙️ Building FAISS index...")
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(FAISS_PATH)

print("✅ FAISS index successfully built and saved")
print(f"📂 Location: {FAISS_PATH}")
# --------------------------------------------------
# Write a build marker so staleness is detectable
# --------------------------------------------------
import hashlib
import time

with open(ASSESSMENTS_PATH, "rb") as f:
    source_hash = hashlib.md5(f.read()).hexdigest()[:8]

marker_path = os.path.join(FAISS_PATH, "_build_info.json")
with open(marker_path, "w") as f:
    json.dump({
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_hash": source_hash,
        "source_file": ASSESSMENTS_PATH,
    }, f, indent=2)

print(f"📝 Build marker written: {marker_path}")