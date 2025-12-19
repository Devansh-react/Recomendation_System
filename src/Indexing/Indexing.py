import os
import json
import faiss
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


from Schema import SHLAssessment 


load_dotenv()

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


embedding_dim = len(embeddings.embed_query("test"))
index = faiss.IndexFlatL2(embedding_dim)


vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# ✅ Load JSON manually
with open("./final_assessments.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 

assessments = [SHLAssessment(**item) for item in raw_data]


# ✅ Convert to LangChain Documents
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

doc_ids = vector_store.add_documents(documents)
print(doc_ids[:5])

# Save the FAISS index locally4
vector_store.save_local("faiss_index")
try:
    vector_store =vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
    print("FAISS index loaded successfully.")
    print(vector_store.index.ntotal)
    len(assessments)

except Exception as e:
    print("Error loading FAISS index:", e)
    
