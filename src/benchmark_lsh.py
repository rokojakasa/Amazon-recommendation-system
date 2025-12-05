import time
import random
import sys
import os
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import recommender_lsh as my_lsh 

# --- CONFIG ---
SAMPLE_SIZE = 20000
NUM_HASHES = 128
THRESHOLD = 0.5
NUM_QUERY_ITEMS = 20 

def run_benchmark():
    print("="*40)
    print("      LSH BENCHMARK: OURS vs DATASKETCH")
    print("="*40)
    
    print(">>> 1. LOADING DATA...")
    # Load 5-core data
    df = my_lsh.src.data_loader.load_preprocess_5core(
        "meta_Electronics.jsonl.gz", 
        "Electronics_5core.csv.gz"
    )
    
    # CRITICAL: Sample down for laptop benchmarking
    if len(df) > SAMPLE_SIZE:
        print(f"    Downsampling from {len(df)} to {SAMPLE_SIZE} rows...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    item_to_users = my_lsh.build_item_to_users(df)
    print(f"    Unique Items: {len(item_to_users)}")

    # ---------------------------------------------------------
    # ROUND 1: YOUR IMPLEMENTATION
    # ---------------------------------------------------------
    print("\n>>> 2. RUNNING YOUR IMPLEMENTATION...")
    t0 = time.time()
    
    hash_seeds = my_lsh.make_hash_seeds(my_lsh.NUM_HASHES)
    sigs = my_lsh.build_signatures(item_to_users, hash_seeds)
    bands = my_lsh.build_lsh_bands(sigs)
    
    my_build_time = time.time() - t0
    print(f"    [Yours] Index Build Time: {my_build_time:.4f} s")

    # ---------------------------------------------------------
    # ROUND 2: DATASKETCH LIBRARY
    # ---------------------------------------------------------
    print("\n>>> 3. RUNNING DATASKETCH LIBRARY...")
    t1 = time.time()
    
    lsh_ds = MinHashLSH(threshold=THRESHOLD, num_perm=my_lsh.NUM_HASHES)
    minhashes_ds = {}
    
    for pid, users in item_to_users.items():
        m = MinHash(num_perm=my_lsh.NUM_HASHES)
        for u in users:
            m.update(str(u).encode('utf8'))
        minhashes_ds[pid] = m
        lsh_ds.insert(str(pid), m)
        
    ds_build_time = time.time() - t1
    print(f"    [Lib]   Index Build Time: {ds_build_time:.4f} s")

    # ---------------------------------------------------------
    # ROUND 3: QUERY SPEED & ACCURACY
    # ---------------------------------------------------------
    print(f"\n>>> 4. BENCHMARKING QUERIES (N={NUM_QUERY_ITEMS})...")
    
    test_pids = random.sample(list(item_to_users.keys()), NUM_QUERY_ITEMS)
    
    # Time Yours
    start_my = time.time()
    for pid in test_pids:
        _ = my_lsh.get_candidate_products(pid, sigs, bands)
    my_avg_query = (time.time() - start_my) / NUM_QUERY_ITEMS

    # Time Library
    start_ds = time.time()
    for pid in test_pids:
        _ = lsh_ds.query(minhashes_ds[pid])
    ds_avg_query = (time.time() - start_ds) / NUM_QUERY_ITEMS

    print(f"    [Yours] Avg Query Time: {my_avg_query:.6f} s")
    print(f"    [Lib]   Avg Query Time: {ds_avg_query:.6f} s")
    
    # ---------------------------------------------------------
    # ROUND 4: SUMMARY
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print(f"INDEX SPEED:  Your code is {ds_build_time/my_build_time:.2f}x FASTER than Datasketch")
    print(f"QUERY SPEED:  Your code is {my_avg_query/ds_avg_query:.2f}x SLOWER than Datasketch")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()