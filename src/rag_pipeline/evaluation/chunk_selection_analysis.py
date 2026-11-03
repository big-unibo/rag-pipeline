
import pandas as pd
from itertools import combinations

df = pd.read_csv("/workspace/outputs/comparison/novel/retrieved_chunks.csv")

# One set of retrieved chunks per run and question.
retrievals = (
    df.groupby(["run_id", "question_id"])["chunk_id"]
    .agg(lambda s: set(s.dropna()))
    .to_dict()
)

rows = []

# Compare all pairs of runs for each question.
for question_id, group in df.groupby("question_id"):
    run_ids = sorted(group["run_id"].unique())

    for run_a, run_b in combinations(run_ids, 2):
        set_a = retrievals.get((run_a, question_id), set())
        set_b = retrievals.get((run_b, question_id), set())

        union = set_a | set_b
        intersection = set_a & set_b

        # Define distance as 0 when both sets are empty.
        similarity = (
            len(intersection) / len(union)
            if union else 1.0
        )

        rows.append({
            "question_id": question_id,
            "run_a": run_a,
            "run_b": run_b,
            "n_chunks_a": len(set_a),
            "n_chunks_b": len(set_b),
            "n_common": len(intersection),
            "n_union": len(union),
            "jaccard_similarity": similarity,
            "jaccard_distance": 1 - similarity,
        })

pairwise = pd.DataFrame(rows)
pairwise.to_csv("/workspace/outputs/comparison/novel/jaccard_pairwise.csv", index=False)

print(pairwise.head().to_string())



summary = (
    pairwise.groupby(["run_a", "run_b"])
    .agg(
        mean_distance=("jaccard_distance", "mean"),
        median_distance=("jaccard_distance", "median"),
        std_distance=("jaccard_distance", "std"),
        n_questions=("question_id", "nunique"),
    )
    .reset_index()
)

# summary.to_csv("/workspace/outputs/jaccard_summary_bis.csv", index=False)
print(summary)
