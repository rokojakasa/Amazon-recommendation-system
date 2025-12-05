import time
import numpy as np
import scipy.sparse as sparse
import implicit
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.recommender_cfBPR as my_bpr 
from src.data_loader import load_preprocess_split

def benchmark_bpr():
    print("="*40)
    print("      BPR BENCHMARK: OUR vs IMPLICIT")
    print("="*40)
    
    print(">>> 1. LOADING DATA...")
    # Load Train/Val/Test
    train_df, val_df, test_df = load_preprocess_split(
        my_bpr.TRAIN_FILE, my_bpr.VAL_FILE, my_bpr.TEST_FILE, my_bpr.META_FILE
    )
    
    n_users = int(train_df['user_idx'].max() + 1)
    n_items = int(train_df['product_idx'].max() + 1)
    print(f"    Training Rows: {len(train_df)}")

    # ---------------------------------------------------------
    # ROUND 1: OUR PYTORCH IMPLEMENTATION
    # ---------------------------------------------------------
    print("\n>>> 2. RUNNING OUR PYTORCH IMPLEMENTATION...")
    
    # Force CPU to be fair (Implicit uses CPU by default)
    device = torch.device('cpu') 
    
    trainer = my_bpr.BPRTrainer(
        train_df, val_df, n_users, n_items,
        emb_dim=64, batch_size=4096, device=device
    )
    
    print("    Training 1 Epoch...")
    t0 = time.time()
    
    # Manual 1-epoch loop to time it accurately
    trainer.model.train()
    for users, pos, neg in trainer.train_loader:
        # Move to CPU
        users, pos, neg = users.to(device), pos.to(device), neg.to(device)
        pos_scores = trainer.model(users, pos)
        neg_scores = trainer.model(users, neg)
        loss = -(pos_scores - neg_scores).sigmoid().log().mean()
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        
    my_time = time.time() - t0
    print(f"    [Yours] Time per Epoch: {my_time:.2f} s")

    # ---------------------------------------------------------
    # ROUND 2: IMPLICIT LIBRARY (C++ Optimized)
    # ---------------------------------------------------------
    print("\n>>> 3. RUNNING IMPLICIT LIBRARY...")
    
    # Convert to sparse matrix (Items x Users)
    sparse_matrix = sparse.csr_matrix(
        (np.ones(len(train_df)), (train_df['product_idx'], train_df['user_idx'])),
        shape=(n_items, n_users)
    )
    
    # Configure model (same factors=64, iterations=1)
    model_lib = implicit.bpr.BayesianPersonalizedRanking(
        factors=64, iterations=1, use_gpu=False
    )
    
    print("    Training 1 Epoch...")
    t1 = time.time()
    model_lib.fit(sparse_matrix, show_progress=False)
    lib_time = time.time() - t1
    print(f"    [Lib]   Time per Epoch: {lib_time:.2f} s")
    
    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print(f"SPEEDUP: Library is {my_time / lib_time:.1f}x FASTER than PyTorch (CPU)")
    print("="*40)

if __name__ == "__main__":
    benchmark_bpr()