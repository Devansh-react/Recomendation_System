# %% [markdown]
# # Recall@10 Evaluation — via live /chat endpoint
# Per the assignment spec, Recall@10 is scored on "final recommendations" —
# i.e. what /chat actually returns in a real conversation, not raw FAISS output.
# This notebook drives each train query through /chat (handling clarification
# turns automatically) and computes recall against the labeled ground truth.

# %%
import requests
import pandas as pd
from collections import defaultdict

# Point this at your local server for dev, or the deployed Render URL before submission
BASE_URL = "http://127.0.0.1:8000"
MAX_CLARIFY_ROUNDS = 3  # safety cap so we never exceed the 8-turn limit

# %% [markdown]
# ## Load train data

# %%
def load_train_data(path):
    return pd.read_excel(path)

data = load_train_data("../../data/train_data.xlsx")
print(data.columns.tolist())
data.head()

# %%
def build_query_to_labels(df):
    query_to_labels = defaultdict(list)
    for _, row in df.iterrows():
        query_to_labels[row["Query"]].append(row["Assessment_url"])
    return query_to_labels

query_to_labels = build_query_to_labels(data)
print(f"Loaded {len(query_to_labels)} unique queries")

# %% [markdown]
# ## Drive a query through /chat, handling clarification automatically
#
# Train queries are often full job-description-style dumps (see PDF example:
# "Here is a text from job description: xx"), so most should get a direct
# recommendation on turn 1. But if the agent asks a clarifying question
# (recommendations: null), we respond with a generic "use your best judgment"
# filler and resend — capped at MAX_CLARIFY_ROUNDS to respect the 8-turn limit
# and avoid infinite loops.

# %%
def get_recommendations_via_chat(query, base_url=BASE_URL, max_rounds=MAX_CLARIFY_ROUNDS):
    history = [{"role": "user", "content": query}]

    for round_num in range(max_rounds):
        resp = requests.post(f"{base_url}/chat", json={"messages": history}, timeout=35)
        resp.raise_for_status()
        result = resp.json()

        recs = result.get("recommendations")
        if recs:  # non-null, non-empty
            return recs, result, history

        # Agent asked a clarifying question — respond generically and retry
        history.append({"role": "assistant", "content": result.get("reply", "")})
        history.append({
            "role": "user",
            "content": "I don't have additional details — please recommend your best options based on what I've already told you."
        })

    # Exhausted clarify rounds without a shortlist — treat as empty prediction
    return [], result, history

# %% [markdown]
# ## Recall computation

# %%
def normalize_url(url):
    if not url:
        return ""
    return url.rstrip("/").replace("/solutions", "")

def recall_at_k(predicted_urls, relevant_urls, k=10):
    predicted = set(predicted_urls[:k])
    relevant = set(relevant_urls)
    if not relevant:
        return 0.0
    return len(predicted & relevant) / len(relevant)

# %%
def evaluate_query(query, relevant_urls, base_url=BASE_URL, k=10):
    recs, raw_result, history = get_recommendations_via_chat(query, base_url)

    predicted_urls = [normalize_url(r.get("url")) for r in recs]
    relevant_norm = [normalize_url(u) for u in relevant_urls]

    r = recall_at_k(predicted_urls, relevant_norm, k)
    return {
        "query": query,
        "recall": r,
        "num_predicted": len(predicted_urls),
        "num_relevant": len(relevant_norm),
        "turns_used": len(history),
        "predicted_urls": predicted_urls,
        "relevant_urls": relevant_norm,
    }

# %% [markdown]
# ## Run full evaluation
# NOTE: make sure `uvicorn src.main:app --reload` is running before executing this.

# %%
results = []
for query, relevant_urls in query_to_labels.items():
    print(f"Evaluating: {query[:70]}...")
    res = evaluate_query(query, relevant_urls)
    results.append(res)
    print(f"  Recall@10: {res['recall']:.2f}  (turns used: {res['turns_used']})")
    print("-" * 60)

# %%
results_df = pd.DataFrame(results)
mean_recall = results_df["recall"].mean()
print(f"\n{'='*60}")
print(f"MEAN RECALL@10: {mean_recall:.4f}")
print(f"{'='*60}")
results_df[["query", "recall", "num_predicted", "num_relevant", "turns_used"]]

# %% [markdown]
# ## Inspect zero-recall queries (diagnostic)
# These are the queries worth investigating further — either a genuine
# retrieval miss, or the agent asked to clarify and used up its rounds.

# %%
zero_recall = results_df[results_df["recall"] == 0.0]
for _, row in zero_recall.iterrows():
    print(f"Query: {row['query']}")
    print(f"  Predicted: {row['predicted_urls']}")
    print(f"  Relevant:  {row['relevant_urls']}")
    print(f"  Turns used: {row['turns_used']}")
    print("-" * 60)