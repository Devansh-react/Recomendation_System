Below is a **complete, submission-ready `README.md`** you can **copy-paste directly** into your GitHub repository.
It strictly aligns with the **SHL Generative AI assignment problem statement**, API requirements, evaluation criteria, and submission expectations.

---

```markdown
# SHL Assessment Recommendation System (Generative AI)

## 📌 Overview

Hiring managers and recruiters often struggle to identify the most relevant assessments for a role due to reliance on keyword-based search and manual filtering.  
This project solves that problem by building an **intelligent, Generative-AI-powered recommendation system** that suggests the most relevant **SHL Individual Test Solutions** based on:

- A natural language query  
- A job description (JD) text  
- A URL containing a JD  

The system leverages **web scraping, embeddings, vector similarity search, and LLM-assisted reasoning** to provide accurate and balanced recommendations.

---

## 🎯 Objectives

- Recommend **minimum 5 and maximum 10** relevant **Individual Test Solutions**
- Ignore **Pre-packaged Job Solutions**
- Ensure recommendations are **contextually relevant and balanced**
- Achieve strong performance on **Mean Recall@10**

---

## 🧠 Core Skills Evaluated

- **Problem Solving** – End-to-end system design and decomposition  
- **Programming Skills** – Clean, modular, production-ready code  
- **Context Engineering** – Correct use of data, constraints, and evaluation metrics  

---

## 🏗️ System Architecture

```

User Query / JD / URL
↓
Query Understanding (LLM)
↓
Embedding Generation
↓
Vector Similarity Search (FAISS)
↓
Re-ranking & Balancing Logic
↓
Final Recommendations (JSON / CSV)

```

---

## 📊 Data Pipeline

### 1. Data Ingestion
- Scraped SHL Product Catalog:
  - https://www.shl.com/solutions/products/product-catalog/
- Only **Individual Test Solutions** were retained
- Ensured **≥ 377 assessments** after crawling

### 2. Data Fields Extracted
- Assessment Name  
- Assessment URL  
- Test Type (A, B, C, D, E, F, K, P)  
- Remote Testing Availability  
- Adaptive IRT Availability  

### 3. Text for Embeddings
Each assessment is converted into a structured textual representation combining:
- Name  
- Test type(s)  
- Technical vs behavioral focus  
- Remote / adaptive capabilities  

---

## 🔍 Recommendation Methodology

### Embeddings
- Sentence-level embeddings generated for:
  - Assessment descriptions
  - User queries / JDs
- Stored in a **FAISS vector index**

### Retrieval
- Top-K similarity search retrieves candidate assessments

### Balancing Logic
- Ensures **hard skills (Knowledge & Skills – K)** and  
  **soft skills (Personality & Behavior – P)** are balanced when required  
- Example:
  > *“Java developer with strong collaboration skills”*  
  → Mix of **technical + behavioral** assessments

---

## 📈 Evaluation Strategy

### Metric Used
**Mean Recall@10**

```

Recall@K = (# relevant assessments in top K) / (total relevant assessments)

Mean Recall@K = average Recall@K across all queries

```

### Evaluation Stages
- Retrieval quality (embedding similarity)
- Final recommendation relevance
- Tested using:
  - **Human-labeled training dataset (10 queries)**
  - **Unlabeled test dataset (9 queries)**

---

## 🔌 API Specification

### Base Requirements
- HTTP/HTTPS
- JSON request & response
- Proper HTTP status codes

---

### 1️⃣ Health Check Endpoint

```

GET /health

````

**Response**
```json
{
  "status": "ok"
}
````

---

### 2️⃣ Assessment Recommendation Endpoint

```
POST /recommend
```

**Request Body**

```json
{
  "query": "Looking to hire a Python developer with analytical and collaboration skills"
}
```

**Response**

```json
{
  "recommendations": [
    {
      "assessment_name": "Python Programming Test",
      "assessment_url": "https://www.shl.com/..."
    }
  ]
}
```

* Returns **1–10 recommendations**
* Output strictly follows the required format

---

## 📁 Submission Artifacts

### 1. API Endpoint URL

* Publicly accessible recommendation API

### 2. GitHub Repository

* Complete source code
* Includes experiments, evaluation scripts, and documentation

### 3. Web Application URL

* Frontend for interactive testing

### 4. CSV Predictions File

* Format (Appendix 3 compliant):

```
Query,Assessment_url
Query 1,Recommendation URL 1
Query 1,Recommendation URL 2
...
```

⚠️ **Strict formatting is mandatory for scoring**

---

## 🧪 Sample Queries

* “I am hiring for Java developers who can collaborate with business teams.”
* “Looking to hire mid-level professionals proficient in Python, SQL, and JavaScript.”
* “Recommend assessments to screen analysts using cognitive and personality tests.”

---

## 🛠️ Tech Stack

* **Python**
* **FAISS** – Vector similarity search
* **LLM (Gemini / OpenAI compatible)** – Query understanding
* **FastAPI / Flask** – API layer
* **Pandas** – Data processing
* **BeautifulSoup / Selenium** – Web scraping

---

## ☁️ Deployment

* API and frontend hosted using **free-tier cloud services**
* Vector index persisted locally
* Designed for easy redeployment and reproducibility

---

## 🚀 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Build embeddings & FAISS index
python build_index.py

# Start API server
python main.py
```

---

## ✅ Key Design Principles

* Modular & maintainable code
* Clear separation of data, retrieval, and ranking logic
* Evaluation-driven optimization
* No hard-coded assumptions
* Fully compliant with SHL submission requirements

---

## 📌 Notes

* Solutions **without scraping SHL catalog** are invalid
* Solutions **without embeddings or retrieval-based logic** are rejected
* Evaluation rigor is critical for scoring

---

## 👤 Author

**Candidate Name**
Devansh sharma 

---