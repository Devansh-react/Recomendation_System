import pandas as pd
import json
import os

# --------------------------------------------------
# Paths
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "Raw_data.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "src", "Indexing", "final_assessments.json")

# --------------------------------------------------
# Load
# --------------------------------------------------
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)
print(f"Loaded {len(df)} rows")
print(df.columns.tolist())
df.head()

# --------------------------------------------------
# Sanity check: confirm this is the NEW schema
# --------------------------------------------------
required_cols = {"entity_id", "name", "link", "description", "job_levels", "keys", "languages", "duration"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(
        f"Raw_data.json is missing columns {missing_cols}. "
        "This looks like the OLD scraped schema, not the new catalog dataset. "
        "Replace Data_cleaning/Raw_data.json with the new JSON before rerunning."
    )

# --------------------------------------------------
# Build text_for_embedding
# --------------------------------------------------
def build_text_for_embedding(row):
    name = row.get("name", "") or ""
    description = (row.get("description") or "").strip()
    job_levels = ", ".join(row.get("job_levels") or []) or "Not specified"
    keys = ", ".join(row.get("keys") or []) or "Not specified"
    languages = ", ".join(row.get("languages") or []) or "Not specified"
    duration = row.get("duration") or "Not specified"

    parts = [f"{name}."]
    if description:
        parts.append(description)
    parts.append(f"Categories: {keys}.")
    parts.append(f"Suitable for job levels: {job_levels}.")
    parts.append(f"Languages available: {languages}.")
    parts.append(f"Duration: {duration}.")

    return " ".join(parts).strip()

df["text_for_embedding"] = df.apply(build_text_for_embedding, axis=1)
df["text_for_embedding"].head(3)

# --------------------------------------------------
# Build final clean schema
# --------------------------------------------------
final_df = pd.DataFrame({
    "id": df["entity_id"],
    "name": df["name"],
    "url": df["link"],
    "description": df["description"].fillna(""),
    "job_levels": df["job_levels"],
    "languages": df["languages"],
    "duration": df["duration"],
    "remote_testing": df["remote"].fillna("no").eq("yes"),
    "adaptive_irt": df["adaptive"].fillna("no").eq("yes"),
    "test_type": df["keys"],  # full category labels, not letter codes
    "text_for_embedding": df["text_for_embedding"],
})

# --------------------------------------------------
# Sanity checks
# --------------------------------------------------
print(f"Total assessments: {len(final_df)}")
print(f"Missing description: {(final_df['description'] == '').sum()}")
print(f"Missing url: {final_df['url'].isna().sum()}")
print(f"Duplicate ids: {final_df['id'].duplicated().sum()}")

# Peek at a few varied rows
final_df.sample(3)[["id", "name", "text_for_embedding"]]

# --------------------------------------------------
# Save
# --------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_df.to_json(OUTPUT_PATH, orient="records", indent=2, force_ascii=False)
print(f"Saved -> {OUTPUT_PATH}")