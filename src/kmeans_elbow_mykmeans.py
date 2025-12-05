import os
import time

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_amazon_fashion
from src.recommender_mykmeans import MyKMeans, build_tfidf
# ^ adjust import if MyKMeans lives in a differently named file


def compute_elbow(tfidf_matrix,
                  k_values,
                  max_items=10000,
                  max_iter=50,
                  tol=1e-3,
                  random_state=42):
    """
    Compute inertia for a range of K using our custom MyKMeans.

    Parameters
    ----------
    tfidf_matrix : scipy.sparse matrix, shape (n_items, n_features)
    k_values     : list[int]
        Different numbers of clusters to try.
    max_items    : int
        Use at most this many items (for speed; converted to dense).
    max_iter, tol, random_state : passed to MyKMeans.

    Returns
    -------
    ks        : np.ndarray of K values
    inertias  : np.ndarray of inertia values
    """
    n_items = tfidf_matrix.shape[0]
    n_use = min(max_items, n_items)

    print(f"Using first {n_use} items (dense) for elbow computation.")
    X_dense = tfidf_matrix[:n_use].toarray()

    inertias = []

    for k in k_values:
        print(f"\n=== Fitting MyKMeans with K = {k} ===")
        t0 = time.time()
        kmeans = MyKMeans(
            n_clusters=k,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        kmeans.fit(X_dense)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.2f} s, inertia = {kmeans.inertia_:.2e}")
        inertias.append(kmeans.inertia_)

    return np.array(k_values, dtype=int), np.array(inertias, dtype=float)


def plot_elbow(ks, inertias, out_dir="results", filename_prefix="mykmeans"):
    """
    Save elbow plot and CSV with (K, inertia).
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Plot
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("Elbow plot for MyKMeans on TF–IDF titles")
    plt.tight_layout()

    plot_path = os.path.join(out_dir, f"{filename_prefix}_elbowK300.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved elbow plot to {plot_path}")

    # 2) CSV with raw values
    csv_path = os.path.join(out_dir, f"{filename_prefix}_elbow_metrics_K300.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("k,inertia\n")
        for k, inertia in zip(ks, inertias):
            f.write(f"{k},{inertia}\n")
    print(f"Saved elbow metrics to {csv_path}")


def main():
    os.makedirs("results", exist_ok=True)

    # 1) Load metadata
    t0 = time.time()
    meta_df, _ = load_amazon_fashion()
    print("Loaded data in", time.time() - t0, "seconds")
    print("meta_df shape:", meta_df.shape)

    # 2) Build TF–IDF matrix over titles
    t1 = time.time()
    vectorizer, tfidf_matrix = build_tfidf(meta_df)
    print("Built TF–IDF matrix in", time.time() - t1, "seconds")
    print("TF–IDF shape:", tfidf_matrix.shape)

    # 3) Define K values to test
    #    Adjust this list depending on how long each run takes.
    k_values = [10, 25, 50, 75, 100]

    # 4) Compute inertia for each K
    ks, inertias = compute_elbow(
        tfidf_matrix,
        k_values,
        max_items=10000,   # same subset size as your main script
        max_iter=50,
        tol=1e-3,
        random_state=42,
    )

    # 5) Save plot + CSV
    plot_elbow(ks, inertias, out_dir="results", filename_prefix="mykmeans")


if __name__ == "__main__":
    main()
