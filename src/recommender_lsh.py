import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import mmh3
import numpy as np
from tqdm import tqdm

from src.data_loader import load_amazon_fashion



NUM_HASHES = 128
NUM_BANDS = 32
ROWS_PER_BAND = NUM_HASHES // NUM_BANDS
RATING_THRESHOLD = 3
RANDOM_SEED = 42


def build_user_item_mappings(reviews_df):
    """Create integer indices for users and items and add them to reviews_df."""
    unique_users = reviews_df["user_id"].unique()
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}

    unique_products = reviews_df["parent_asin"].unique()
    product_to_idx = {pid: i for i, pid in enumerate(unique_products)}
    idx_to_product = {i: pid for pid, i in product_to_idx.items()}

    reviews_df["user_idx"] = reviews_df["user_id"].map(user_to_idx)
    reviews_df["product_idx"] = reviews_df["parent_asin"].map(product_to_idx)

    return user_to_idx, idx_to_user, product_to_idx, idx_to_product


def build_item_to_users(reviews_df) -> Dict[int, Set[int]]:
    """Map each product_idx -> set(user_idx) with rating >= threshold."""
    item_to_users = defaultdict(set)
    for _, row in reviews_df.iterrows():
        if row["rating"] >= RATING_THRESHOLD:
            item_to_users[int(row["product_idx"])].add(int(row["user_idx"]))
    return item_to_users


def make_hash_seeds(num_hashes: int, seed: int = RANDOM_SEED) -> List[int]:
    rng = random.Random(seed)
    MAX_SEED = 2**32 - 1
    return rng.sample(range(MAX_SEED), num_hashes)


def minhash_signature(user_set: Set[int], hash_seeds: List[int]) -> List[int]:
    """Compute a MinHash signature for one set of user_ids."""
    signature = []
    for hs in hash_seeds:
        m = min(mmh3.hash(str(u), hs) for u in user_set)
        signature.append(m)
    return signature


def build_signatures(
    item_to_users: Dict[int, Set[int]], hash_seeds: List[int]
) -> Dict[int, List[int]]:
    """Compute MinHash signatures for all items."""
    product_signatures = {}
    for product, users in tqdm(
        item_to_users.items(), desc="Computing MinHash signatures"
    ):
        if not users:
            continue
        product_signatures[product] = minhash_signature(users, hash_seeds)
    return product_signatures


def build_lsh_bands(product_signatures: Dict[int, List[int]]):
    """Assign signatures to LSH bands."""
    bands = [defaultdict(list) for _ in range(NUM_BANDS)]
    for product, signature in tqdm(
        product_signatures.items(), desc="Assigning LSH bands"
    ):
        for i in range(NUM_BANDS):
            start_idx = i * ROWS_PER_BAND
            end_idx = (i + 1) * ROWS_PER_BAND
            band_signature = tuple(signature[start_idx:end_idx])
            bands[i][band_signature].append(product)
    return bands


def get_candidate_products(
    product_id: int,
    product_signatures: Dict[int, List[int]],
    bands,
) -> Set[int]:
    """Find candidate similar items using LSH bands."""
    signature = product_signatures[product_id]
    candidate_products = set()
    for i in range(NUM_BANDS):
        start_idx = i * ROWS_PER_BAND
        end_idx = (i + 1) * ROWS_PER_BAND
        band_signature = tuple(signature[start_idx:end_idx])
        candidate_products.update(bands[i][band_signature])
    candidate_products.discard(product_id)
    return candidate_products


def estimated_jaccard(
    product1: int,
    product2: int,
    product_signatures: Dict[int, List[int]],
) -> float:
    sig1 = product_signatures[product1]
    sig2 = product_signatures[product2]
    matches = sum(1 for i in range(NUM_HASHES) if sig1[i] == sig2[i])
    return matches / NUM_HASHES


def get_top_k_similar_products(
    product_id: int,
    k: int,
    product_signatures: Dict[int, List[int]],
    bands,
) -> List[Tuple[int, float]]:
    candidates = get_candidate_products(product_id, product_signatures, bands)
    similarities = {}
    for candidate in candidates:
        sim = estimated_jaccard(product_id, candidate, product_signatures)
        similarities[candidate] = sim
    top_k = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    return top_k


def main():
    os.makedirs("results", exist_ok=True)

    t0 = time.time()
    meta_df, reviews_df = load_amazon_fashion()
    print("Loaded meta and reviews in", time.time() - t0, "seconds")

    t1 = time.time()
    user_to_idx, idx_to_user, product_to_idx, idx_to_product = build_user_item_mappings(
        reviews_df
    )
    item_to_users = build_item_to_users(reviews_df)
    print("Built mappings and item_to_users in", time.time() - t1, "seconds")

    asin_to_title = dict(zip(meta_df["parent_asin"], meta_df["title"]))
    idx_to_title = {
        idx: asin_to_title.get(asin, "Unknown Title")
        for idx, asin in idx_to_product.items()
    }

    def get_product_title(product_idx: int) -> str:
        return idx_to_title.get(product_idx, "Unknown Title")

    hash_seeds = make_hash_seeds(NUM_HASHES)

    t2 = time.time()
    product_signatures = build_signatures(item_to_users, hash_seeds)
    print("Computed MinHash signatures in", time.time() - t2, "seconds")

    t3 = time.time()
    bands = build_lsh_bands(product_signatures)
    print("Assigned LSH bands in", time.time() - t3, "seconds")

    # ---- EXAMPLES SECTION ----
    # How many different query items you want examples for
    NUM_EXAMPLES = 10
    K = 5  # top-k per query

    # Pick some example product indices that actually have signatures
    all_items = list(product_signatures.keys())
    if len(all_items) < NUM_EXAMPLES:
        example_pids = all_items
    else:
        # deterministic sample so it doesn't change every run
        random.seed(RANDOM_SEED)
        example_pids = random.sample(all_items, NUM_EXAMPLES)

    out_lines = []
    out_lines.append(
        f"Item–item CF with MinHash+LSH examples "
        f"(NUM_HASHES={NUM_HASHES}, NUM_BANDS={NUM_BANDS}, K={K})\n\n"
    )

    for example_pid in example_pids:
        query_title = get_product_title(example_pid)
        header = (
            f"Query item idx {example_pid}: {query_title}\n"
            f"Top {K} similar products:\n"
        )
        print("\n" + header)
        out_lines.append(header)

        top_k = get_top_k_similar_products(
            example_pid, K, product_signatures, bands
        )

        if not top_k:
            line = "  (No candidates found)\n"
            print(line)
            out_lines.append(line)
            continue

        for pid, score in top_k:
            line = (
                f"  - idx={pid}, score={score:.4f}, "
                f"title={get_product_title(pid)}\n"
            )
            print(line, end="")
            out_lines.append(line)

        out_lines.append("\n")

    # Save all examples to file
    os.makedirs("results", exist_ok=True)
    with open("results/lsh_examples.txt", "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print("\nSaved examples to results/lsh_examples.txt")


    print("\nSaved examples to results/lsh_examples.txt")

        # ---- SAVE TRAINED LSH DATA ----
    import pickle

    with open("results/lsh_signatures.pkl", "wb") as f:
        pickle.dump(product_signatures, f)

    with open("results/lsh_bands.pkl", "wb") as f:
        pickle.dump(bands, f)

    with open("results/lsh_idx_to_title.pkl", "wb") as f:
        pickle.dump(idx_to_title, f)

    print("Saved LSH data to results/*.pkl")



if __name__ == "__main__":
    main()
