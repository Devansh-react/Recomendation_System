import pandas as pd

from src.Tool.tool import retrieve_documents


# ======================
# CONFIG
# ======================
TEST_DATA_PATH = "data/Test_data.xlsx"
OUTPUT_CSV_PATH = "submission.csv"
TOP_K = 10


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
    print("📄 Loading test queries...")
    test_queries = load_test_queries(TEST_DATA_PATH)
    print(f"Total test queries: {len(test_queries)}")

    rows = []

    for idx, query in enumerate(test_queries, start=1):
        print(f"\n[{idx}/{len(test_queries)}] Processing query:")
        print(query[:80], "...")

        docs = retrieve_documents(query, top_k=TOP_K)

        for doc in docs:
            rows.append({
                "Query": query,
                "Assessment_url": normalize_url(
                    doc.metadata.get("url")
                )
            })

    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\n✅ Submission file created successfully!")
    print(f"📄 File: {OUTPUT_CSV_PATH}")
    print(f"📊 Total rows: {len(submission_df)} "
          f"(expected {len(test_queries) * TOP_K})")


if __name__ == "__main__":
    main()
