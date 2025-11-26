import os
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_amazon_fashion


def build_tfidf(meta_df):
    """
    Build a TF-IDF matrix over product titles.

    Returns
    -------
    vectorizer : TfidfVectorizer
    tfidf_matrix : scipy.sparse matrix of shape (n_items, n_features)
    """
    titles = meta_df["title"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
    )
    tfidf_matrix = vectorizer.fit_transform(titles)
    return vectorizer, tfidf_matrix


def get_top_k_tfidf(item_idx, tfidf_matrix, k=5):
    """
    Get top-k TF-IDF cosine-similar items for a given item index.
    """
    sims = cosine_similarity(tfidf_matrix[item_idx], tfidf_matrix).flatten()
    # sort descending, skip self
    idxs_sorted = sims.argsort()[::-1]
    idxs_sorted = idxs_sorted[idxs_sorted != item_idx]
    top = idxs_sorted[:k]
    return [(int(i), float(sims[i])) for i in top]


def main():
    os.makedirs("results", exist_ok=True)

    # 1) Load metadata (we only need meta_df here)
    t0 = time.time()
    meta_df, _ = load_amazon_fashion()
    print("Loaded data in", time.time() - t0, "seconds")

    # 2) Build TF-IDF matrix
    t1 = time.time()
    vectorizer, tfidf_matrix = build_tfidf(meta_df)
    print("Built TF-IDF matrix in", time.time() - t1, "seconds")
    print("TF-IDF shape:", tfidf_matrix.shape)

    # 3) Choose 10 example items with non-empty TF-IDF rows
    NUM_EXAMPLES = 10
    K = 5

    # indices where row has at least 1 non-zero TF-IDF entry
    non_empty = np.where(tfidf_matrix.getnnz(axis=1) > 0)[0]
    if len(non_empty) == 0:
        print("No non-empty TF-IDF rows found. Exiting.")
        return

    num_examples = min(NUM_EXAMPLES, len(non_empty))
    rng = np.random.default_rng(42)
    example_indices = rng.choice(non_empty, size=num_examples, replace=False)

    out_lines = []
    out_lines.append(
        f"TF-IDF Content-Based Recommendations ({num_examples} examples, top-{K})\n\n"
    )

    for idx in example_indices:
        idx = int(idx)
        query_title = meta_df.loc[idx, "title"]
        out_lines.append(f"Query item idx {idx}: {query_title}\n")
        out_lines.append(f"Top {K} TF-IDF similar items:\n")

        sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        sorted_ids = sims.argsort()[::-1]

        # skip itself, keep next K
        top_ids = [int(i) for i in sorted_ids if i != idx][:K]

        for sim_idx in top_ids:
            title = meta_df.loc[sim_idx, "title"]
            score = sims[sim_idx]
            out_lines.append(
                f"  - idx={sim_idx}, cos_sim={score:.4f}, title={title}\n"
            )

        out_lines.append("\n")

    # 4) Save to file
    out_path = "results/tfidf_examples.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"Saved TF-IDF examples to {out_path}")


if __name__ == "__main__":
    main()
