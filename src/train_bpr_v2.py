# bpr_pipeline.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import heapq

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "bpr_model_electronics_v1.pt")
TRAIN_FILE = "Electronics_train.csv.gz"
VAL_FILE = "Electronics_valid.csv.gz"
TEST_FILE = "Electronics_test.csv.gz"
META_FILE = "meta_Electronics.jsonl.gz"

class BPRMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, emb_dim)
        self.item_factors = nn.Embedding(n_items, emb_dim)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)

        self.user_factors.weight.data.normal_(0, 0.01)
        self.item_factors.weight.data.normal_(0, 0.01)
        self.user_biases.weight.data.zero_()
        self.item_biases.weight.data.zero_()

    def forward(self, users, items):
        u = self.user_factors(users)
        v = self.item_factors(items)
        dot = (u * v).sum(dim=1)
        bu = self.user_biases(users).view(-1)
        bi = self.item_biases(items).view(-1)
        return dot + bu + bi


class BPRDataset(Dataset):
    """BPR dataset with safe negative sampling."""
    def __init__(self, interactions_df, n_items):
        interactions_df = interactions_df[interactions_df['rating'] >= 3]
        interactions_df = interactions_df.dropna(subset=['user_idx', 'product_idx'])
        self.users = interactions_df['user_idx'].values.astype(np.int64)
        self.pos_items = interactions_df['product_idx'].values.astype(np.int64)
        self.n_items = n_items

        self.user_pos = {}
        for u, i in zip(self.users, self.pos_items):
            self.user_pos.setdefault(int(u), set()).add(int(i))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = int(self.users[idx])
        pos = int(self.pos_items[idx])
        # Safe negative sampling
        attempts = 0
        max_attempts = 100
        neg = np.random.randint(0, self.n_items)
        while neg in self.user_pos[user] and attempts < max_attempts:
            neg = np.random.randint(0, self.n_items)
            attempts += 1
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long)
        )


def save_checkpoint(model, optimizer, epoch, path=CHECKPOINT_FILE):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(model, optimizer=None, path=CHECKPOINT_FILE, device='cpu'):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[Checkpoint] Loaded {path}, resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"[Checkpoint] No checkpoint at {path}, starting fresh.")
        return 0


class BPRTrainer:
    """Trainer class for BPR model."""
    def __init__(
        self,
        train_df,
        val_df,
        n_users,
        n_items,
        emb_dim=64,
        batch_size=4096,
        lr=1e-3,
        l2_reg=1e-5,
        device=None,
        checkpoint_path=CHECKPOINT_FILE
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Using device: {self.device}")

        self.train_dataset = BPRDataset(train_df, n_items)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        self.model = BPRMatrixFactorization(n_users, n_items, emb_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=l2_reg)
        self.checkpoint_path = checkpoint_path

        self.start_epoch = load_checkpoint(self.model, self.optimizer, path=self.checkpoint_path, device=self.device)
        self.val_df = val_df

    def train(self, epochs=30, recall_k=50):
        best_loss = float('inf')
        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for users, pos_items, neg_items in self.train_loader:
                users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
                pos_scores = self.model(users, pos_items)
                neg_scores = self.model(users, neg_items)

                # Use softplus for numerical stability
                loss = torch.nn.functional.softplus(neg_scores - pos_scores).mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                epoch_loss += loss.item() * users.size(0)

            t1 = time.time()
            avg_loss = epoch_loss / len(self.train_dataset)

            # Validate every 10 epochs
            # if self.val_df is not None and (epoch + 1) % 10 == 0:
                # recall = recall_at_k_batch(self.model, self.val_df, self.train_dataset, n_items=self.model.item_factors.num_embeddings, k=recall_k, device=self.device)
            #     print(f"[Epoch {epoch+1}] Validation Recall@{recall_k}: {recall:.4f}")

            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.6f} — Time: {t1 - t0:.1f}s")
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(self.model, self.optimizer, epoch, path=self.checkpoint_path)

        return self.model


# -----------------------------
# Evaluation / Recommendation
# -----------------------------
@torch.no_grad()
def recommend_topk(model, user_idx, seen_items, n_items, k=10, device='cpu'):
    model.eval()
    user_tensor = torch.full((n_items,), user_idx, dtype=torch.long, device=device)
    item_tensor = torch.arange(n_items, device=device)
    scores = model(user_tensor, item_tensor).cpu()
    if seen_items:
        scores[list(seen_items)] = -1e9
    topk = torch.topk(scores, k=k).indices.numpy()
    return topk


@torch.no_grad()
def recommend_topk_batch_large(model, user_indices, seen_items_dict, n_items, k=10,
                               batch_size=128, chunk_size=10000, device=None):
    """
    Large-scale batch top-K recommendation for millions of items.
    
    Parameters
    ----------
    model : nn.Module
        Trained BPR model
    user_indices : list or np.array
        User indices to generate recommendations for
    seen_items_dict : dict
        user_id -> set of items already seen
    n_items : int
        Total number of items
    k : int
        Top-K items to recommend
    batch_size : int
        Number of users per batch
    chunk_size : int
        Number of items per scoring chunk
    device : torch.device or None
        Device to use; if None, inferred from model
    """
    model.eval()
    device = device or next(model.parameters()).device
    all_items = torch.arange(n_items, device=device)
    topk_all = []

    for start_user in range(0, len(user_indices), batch_size):
        batch_users = user_indices[start_user:start_user + batch_size]
        batch_size_actual = len(batch_users)
        # Initialize heaps for top-K per user
        topk_heaps = [[] for _ in range(batch_size_actual)]

        for start_item in range(0, n_items, chunk_size):
            end_item = min(start_item + chunk_size, n_items)
            items_chunk = all_items[start_item:end_item]
            chunk_size_actual = len(items_chunk)

            # Flatten user-item pairs for model
            user_tensor = torch.tensor(batch_users, device=device).unsqueeze(1).repeat(1, chunk_size_actual).flatten()
            item_tensor = items_chunk.unsqueeze(0).repeat(batch_size_actual, 1).flatten()

            # Forward pass
            scores_flat = model(user_tensor, item_tensor)
            scores = scores_flat.view(batch_size_actual, chunk_size_actual)

            # Mask seen items
            for i, u in enumerate(batch_users):
                seen = seen_items_dict.get(u, set())
                mask_indices = [idx - start_item for idx in seen if start_item <= idx < end_item]
                if mask_indices:
                    scores[i, mask_indices] = -1e9

            # Update heaps
            scores_cpu = scores.cpu().numpy()
            for i in range(batch_size_actual):
                for idx, score in enumerate(scores_cpu[i]):
                    if len(topk_heaps[i]) < k:
                        heapq.heappush(topk_heaps[i], (score, idx + start_item))
                    else:
                        heapq.heappushpop(topk_heaps[i], (score, idx + start_item))

        # Extract sorted top-K per user
        for heap in topk_heaps:
            topk_sorted = [idx for score, idx in sorted(heap, key=lambda x: -x[0])]
            topk_all.append(topk_sorted)

    return topk_all


def recall_at_k_batch_large(model, df_eval, train_df, n_items, k=10,
                            batch_size=128, chunk_size=10000, device=None):
    """
    Large-scale batched recall@K computation using chunked scoring.
    
    Parameters
    ----------
    model : nn.Module
        Trained BPR model
    df_eval : pd.DataFrame
        Evaluation DataFrame with columns ['user_idx', 'product_idx']
    train_df : pd.DataFrame
        Training DataFrame to mask seen items
    n_items : int
        Total number of items
    k : int
        Recall@K
    batch_size : int
        Number of users per batch
    chunk_size : int
        Number of items per scoring chunk
    device : torch.device or None
        Device to use; inferred from model if None
    """
    model.eval()
    device = device or next(model.parameters()).device

    # Build user -> seen items from training set
    train_user_pos = train_df.groupby('user_idx')['product_idx'].apply(set).to_dict()
    users_eval = df_eval['user_idx'].unique()

    # Get top-K recommendations
    topk_all = recommend_topk_batch_large(
        model, users_eval, train_user_pos, n_items, k=k,
        batch_size=batch_size, chunk_size=chunk_size, device=device
    )

    # Map user -> actual items in eval set
    user_to_eval_items = df_eval.groupby('user_idx')['product_idx'].apply(list).to_dict()
    recalls = []

    for u, topk in zip(users_eval, topk_all):
        actual_items = user_to_eval_items.get(u, [])
        if not actual_items:
            continue
        hits = len(set(actual_items) & set(topk))
        recalls.append(hits / len(actual_items))

    return float(np.mean(recalls)) if recalls else 0.0




def sample_recommendations(model, train_df, item_to_title, users, n_last=5, k=10, n_sample=5, device='cpu'):
    sample_users = np.random.choice(users, size=min(n_sample, len(users)), replace=False)
    print(f"\n[Sample Recommendations] Top-{k} and last {n_last} purchases:")

    for u in sample_users:
        user_train_items = (
            train_df[train_df['user_idx'] == u]
            .sort_values('timestamp', ascending=False)['product_idx']
            .values
        )

        last_titles = [item_to_title.get(int(i), str(i)) for i in user_train_items[:n_last]]
        seen_train = set(user_train_items)

        topk = recommend_topk(
            model,
            u,
            seen_train,
            n_items=model.item_factors.num_embeddings,
            k=k,
            device=device
        )

        topk_titles = [item_to_title.get(int(i), str(i)) for i in topk]

        print("\n" + "=" * 80)
        print(f"User {u}")
        print("-" * 80)

        print(f"Last {n_last} purchases:")
        for i, title in enumerate(last_titles, 1):
            print(f"  {i}. {title}")

        print(f"\nTop {k} recommendations:")
        for i, title in enumerate(topk_titles, 1):
            print(f"  {i}. {title}")



# -----------------------------
# Example Main
# -----------------------------
if __name__ == "__main__":
    from src.data_loader import load_preprocess_split

    train_df, val_df, test_df = load_preprocess_split(TRAIN_FILE, VAL_FILE, TEST_FILE, META_FILE)

    n_users = int(train_df['user_idx'].max() + 1)
    all_items = pd.concat([train_df['product_idx'], val_df['product_idx'], test_df['product_idx']])
    n_items = int(all_items.max() + 1)

    item_to_title = train_df.drop_duplicates('product_idx').set_index('product_idx')['title'].to_dict()

    # -----------------------------
    # TRAIN or LOAD
    # -----------------------------
    trainer = BPRTrainer(
        train_df=train_df,
        val_df=val_df,
        n_users=n_users,
        n_items=n_items,
        emb_dim=128,
        batch_size=4096,
        lr=1e-3,
        l2_reg=1e-5,
        device=None,
        checkpoint_path=CHECKPOINT_FILE
    )

    # # Uncomment to train
    # model = trainer.train(epochs=30, recall_k=50)
    
    

    # Load checkpoint only (skip training)
    trainer.model.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=trainer.device)['model_state_dict'])
    model = trainer.model

    # -----------------------------
    # Sample recommendations
    # -----------------------------
    sample_recommendations(model, train_df, item_to_title, test_df['user_idx'].unique(), n_last=5, k=10, n_sample=5, device=trainer.device)
    
    # -----------------------------
    # Evaluate test set
    # -----------------------------
    test_recall = recall_at_k_batch_large(model, val_df, train_df, n_items, k=50, device=trainer.device, batch_size=128, chunk_size=10000)
    print(f"\n[Test Recall@50] {test_recall:.4f}")
