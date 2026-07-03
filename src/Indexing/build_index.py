import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    raise RuntimeError("MISTRAL_API_KEY not set in environment")

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

if not os.path.exists(ASSESSMENTS_PATH):
    raise FileNotFoundError(f"Input data file not found: {ASSESSMENTS_PATH}")

os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

with open(ASSESSMENTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📄 Loaded {len(data)} assessments")

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
# Create embeddings + build index — key passed explicitly
# --------------------------------------------------
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
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