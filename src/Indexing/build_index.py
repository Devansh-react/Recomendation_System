import os
import json
import faiss  
from langchain_community.vectorstores import FAISS  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --------------------------------------------------
# Paths (repo-root relative)
# --------------------------------------------------
FAISS_PATH = "data.faiss_index"
ASSESSMENTS_PATH = "src/Indexing/final_assessments.json"

# --------------------------------------------------
# Safety Checks
# --------------------------------------------------
if not os.path.exists(ASSESSMENTS_PATH):
    raise FileNotFoundError(
        f"Input data file not found: {ASSESSMENTS_PATH}"
    )

os.makedirs(FAISS_PATH, exist_ok=True)

# --------------------------------------------------
# Load assessment data
# --------------------------------------------------
with open(ASSESSMENTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📄 Loaded {len(data)} assessments")

# --------------------------------------------------
# Convert to LangChain Documents
# --------------------------------------------------
documents = []

for item in data:
    documents.append(
        Document(
            page_content=item["text_for_embedding"],
            metadata={
                "id": item["id"],
                "name": item["name"],
                "url": item["url"],
                "test_type": item["test_type"],
            },
        )
    )

# --------------------------------------------------
# Create embeddings
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# --------------------------------------------------
# Build and save FAISS index
# --------------------------------------------------
print("⚙️ Building FAISS index...")
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(FAISS_PATH)

print("✅ FAISS index successfully built and saved")
print(f"📂 Location: {FAISS_PATH}")