import pickle

def load_lsh():
    with open("../results/lsh_signatures.pkl", "rb") as f:
        product_signatures = pickle.load(f)

    with open("../results/lsh_bands.pkl", "rb") as f:
        bands = pickle.load(f)

    with open("../results/lsh_idx_to_title.pkl", "rb") as f:
        idx_to_title = pickle.load(f)

    return product_signatures, bands, idx_to_title

def get_top_k_similar_products(product_id, k, product_signatures, bands):
    signature = product_signatures[product_id]

    NUM_HASHES = len(signature)
    NUM_BANDS = len(bands)
    ROWS_PER_BAND = NUM_HASHES // NUM_BANDS

    # LSH retrieval
    candidates = set()
    for i in range(NUM_BANDS):
        start = i * ROWS_PER_BAND
        end = (i + 1) * ROWS_PER_BAND
        band_sig = tuple(signature[start:end])
        candidates.update(bands[i].get(band_sig, []))

    candidates.discard(product_id)

    # compute MinHash Jaccard
    def est_jaccard(a, b):
        sig1 = product_signatures[a]
        sig2 = product_signatures[b]
        matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return matches / NUM_HASHES

    sims = [(c, est_jaccard(product_id, c)) for c in candidates]
    return sorted(sims, key=lambda x: x[1], reverse=True)[:k]

if __name__ == "__main__":
    product_signatures, bands, idx_to_title = load_lsh()

    # EXAMPLE QUERY
    query_id = 250
    if query_id not in product_signatures:
        query_id = next(iter(product_signatures.keys()))

    results = get_top_k_similar_products(query_id, 5, product_signatures, bands)

    print(f"Query: {idx_to_title[query_id]}")
    print("Top 5 similar items:")
    for pid, score in results:
        print(f" - {idx_to_title[pid]} (score={score:.4f})")
