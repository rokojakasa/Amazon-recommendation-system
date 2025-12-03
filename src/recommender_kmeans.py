import os
import time

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

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

    # 3) K-means clustering
    K = 100  # adjust if needed
    t2 = time.time()
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)
    print("Fitted KMeans in", time.time() - t2, "seconds")

    meta_df = meta_df.copy()
    meta_df["cluster"] = clusters

    # 4) Summarize 10 example clusters: top terms + sample titles
    feature_names = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    NUM_EXAMPLES = 10
    example_clusters = list(range(min(NUM_EXAMPLES, K)))

    out_lines = []
    out_lines.append(
        f"K-means Clustering Examples ({len(example_clusters)} clusters shown, K={K})\n\n"
    )

    for c in example_clusters:
        out_lines.append(f"Cluster {c}:\n")

        # Top terms describing this cluster
        top_terms = [feature_names[idx] for idx in order_centroids[c, :10]]
        out_lines.append(f"  Top terms: {', '.join(top_terms)}\n")

        # Sample up to 5 titles from this cluster
        cluster_items = meta_df[meta_df["cluster"] == c]["title"].dropna()
        sample_titles = cluster_items.head(5).tolist()

        out_lines.append("  Sample items:\n")
        for title in sample_titles:
            out_lines.append(f"    - {title}\n")

        out_lines.append("\n")

    out_path = "results/cluster_examples.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"Saved cluster summaries to {out_path}")


if __name__ == "__main__":
    main()
