import pandas as pd

from src.Indexing.Index import get_vector_store
from src.Tool.tool import rerank_by_query_overlap


# ======================
# CONFIG
# ======================
TEST_DATA_PATH = "data/Test_data.xlsx"
OUTPUT_CSV_PATH = "submission.csv"
TOP_K = 10
RETRIEVE_K = 30


# ======================
# HELPERS
# ======================
def normalize_url(url: str) -> str:
    """
    Normalize SHL URLs to avoid recall mismatch.
    """
    if not url:
        return url
    return url.rstrip("/").replace("/solutions", "")


def retrieve_top_k(vector_store, query: str, retrieve_k: int):
    """
    Retrieve more candidates than needed for recall.
    """
    return vector_store.similarity_search(query, k=retrieve_k)


def load_test_queries(path: str):
    """
    Load unlabeled test queries from Excel.
    Expects a column named 'query'.
    """
    df = pd.read_excel(path)

    if "query" not in df.columns:
        raise ValueError(
            f"'query' column not found in {path}. "
            f"Found columns: {list(df.columns)}"
        )

    return df["query"].dropna().tolist()


# ======================
# MAIN
# ======================
def main():
    print("🚀 Loading vector store...")
    vector_store = get_vector_store()

    print("📄 Loading test queries...")
    test_queries = load_test_queries(TEST_DATA_PATH)
    print(f"Total test queries: {len(test_queries)}")

    rows = []

    for idx, query in enumerate(test_queries, start=1):
        print(f"\n[{idx}/{len(test_queries)}] Processing query:")
        print(query[:80], "...")

        # 1️⃣ Retrieve more candidates
        docs = retrieve_top_k(
            vector_store=vector_store,
            query=query,
            retrieve_k=RETRIEVE_K
        )

        # 2️⃣ Re-rank
        docs = rerank_by_query_overlap(query, docs)

        # 3️⃣ Take top-10
        docs = docs[:TOP_K]

        # 4️⃣ Add to CSV rows
        for doc in docs:
            rows.append({
                "Query": query,
                "Assessment_url": normalize_url(
                    doc.metadata.get("url")
                )
            })

    # 5️⃣ Save CSV
    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\n✅ Submission file created successfully!")
    print(f"📄 File: {OUTPUT_CSV_PATH}")
    print(f"📊 Total rows: {len(submission_df)} "
          f"(expected {len(test_queries) * TOP_K})")


if __name__ == "__main__":
    main()
