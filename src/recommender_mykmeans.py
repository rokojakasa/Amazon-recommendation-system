import os
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import load_amazon_fashion


class MyKMeans:
    """
    Simple K-means implementation from scratch (NumPy).

    Parameters
    ----------
    n_clusters : int
        Number of clusters (K).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on centroid movement.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, n_clusters=100, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # attributes after fitting
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        """
        Simple random initialization of centroids from data points.
        (You can mention in the report that this is the basic variant;
        sklearn uses k-means++ for better stability.)
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[indices]
        return centroids

    def _compute_distances(self, X, centroids):
        """
        Compute squared Euclidean distance from each point to each centroid.

        X: (n_samples, n_features)
        centroids: (n_clusters, n_features)

        Returns: (n_samples, n_clusters)
        """
        # Using (x - c)^2 = x^2 + c^2 - 2 x·c for efficiency
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)           # (n_samples, 1)
        C_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T # (1, n_clusters)
        # Dot product X·C^T: (n_samples, n_clusters)
        XC = X @ centroids.T
        dists = X_sq + C_sq - 2 * XC
        return dists

    def fit(self, X):
        """
        Run K-means on X.

        X should be a dense NumPy array of shape (n_samples, n_features).
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # 1) init
        centroids = self._init_centroids(X)

        for it in range(self.max_iter):
            # 2) assignment step: assign each point to closest centroid
            dists = self._compute_distances(X, centroids)
            labels = np.argmin(dists, axis=1)

            # 3) update step: recompute centroids as mean of assigned points
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # handle empty cluster: reinitialise to random data point
                    idx = np.random.randint(0, n_samples)
                    new_centroids[k] = X[idx]

            # 4) convergence check (max centroid shift < tol)
            shift = np.linalg.norm(new_centroids - centroids)
            # print(f"Iter {it}, centroid shift={shift:.6f}")
            if shift < self.tol:
                centroids = new_centroids
                break

            centroids = new_centroids

        # store results
        self.cluster_centers_ = centroids
        self.labels_ = labels

        # inertia: sum of squared distances to assigned centroid
        final_dists = self._compute_distances(X, centroids)
        min_dists = final_dists[np.arange(n_samples), labels]
        self.inertia_ = float(np.sum(min_dists))

        return self

    def fit_predict(self, X):
        """
        Convenience method: fit and return labels.
        """
        self.fit(X)
        return self.labels_


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

    # 3) K-means clustering (from scratch)
    K = 100  # adjust if needed

    # For memory reasons we cluster on a subset in dense form
    max_items = 10000  # choose depending on RAM
    n_items = tfidf_matrix.shape[0]
    n_use = min(max_items, n_items)
    print(f"Clustering on first {n_use} items (dense) with custom K-means...")

    X_dense = tfidf_matrix[:n_use].toarray()

    t2 = time.time()
    kmeans = MyKMeans(n_clusters=K, max_iter=50, tol=1e-3, random_state=42)
    clusters_subset = kmeans.fit_predict(X_dense)
    print("Fitted custom KMeans in", time.time() - t2, "seconds")
    print("Final inertia:", kmeans.inertia_)

    # For items beyond n_use, you could either:
    # - mark them as "unclustered", or
    # - assign them to nearest centroid using the learned centroids.
    # Here we show how to assign all items to the nearest learned centroid:

    from scipy.sparse import csr_matrix

    def assign_all_items(tfidf_matrix_full, centroids):
        """
        Assign every item in sparse TF-IDF to nearest centroid.

        We do the distance calculation in a sparse-aware way by converting
        centroids to a csr_matrix and using the same (x - c)^2 trick.
        """
        # centroids: (K, d) dense
        # tfidf_matrix_full: (N, d) sparse
        N = tfidf_matrix_full.shape[0]
        K = centroids.shape[0]

        # convert centroids to sparse to reuse sparse dot products
        centroids_sparse = csr_matrix(centroids)

        # x^2 term
        X_sq = np.array(tfidf_matrix_full.multiply(tfidf_matrix_full).sum(axis=1)).reshape(-1, 1)
        # c^2 term
        C_sq = np.array(centroids_sparse.multiply(centroids_sparse).sum(axis=1)).reshape(1, -1)
        # X·C^T
        XC = tfidf_matrix_full @ centroids_sparse.T  # (N, K)
        dists = X_sq + C_sq - 2 * XC
        labels = np.argmin(dists, axis=1).A1  # flatten to 1D array
        return labels

    clusters = assign_all_items(tfidf_matrix, kmeans.cluster_centers_)

    meta_df = meta_df.copy()
    meta_df["cluster"] = clusters

    # 4) Summarize 10 example clusters: top terms + sample titles
    feature_names = vectorizer.get_feature_names_out()
    order_centroids = np.argsort(kmeans.cluster_centers_, axis=1)[:, ::-1]

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

    out_path = "results/mycluster_examples.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"Saved cluster summaries to {out_path}")


if __name__ == "__main__":
    main()
