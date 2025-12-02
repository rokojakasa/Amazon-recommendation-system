# train_bpr_checkpoint.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data_loader import load_preprocess_split

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "bpr_model.pt")

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
        bu = self.user_biases(users).squeeze()
        bi = self.item_biases(items).squeeze()
        return dot + bu + bi

class BPRDataset(Dataset):
    def __init__(self, interactions_df, n_items):
        interactions_df = interactions_df[interactions_df['rating'] >= 3]
        self.users = interactions_df['user_idx'].values
        self.pos_items = interactions_df['product_idx'].values
        self.n_items = n_items

        self.user_pos = {}
        for u, i in zip(self.users, self.pos_items):
            self.user_pos.setdefault(int(u), set()).add(int(i))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = int(self.users[idx])
        pos = int(self.pos_items[idx])
        neg = np.random.randint(0, self.n_items)
        while neg in self.user_pos[user]:
            neg = np.random.randint(0, self.n_items)
        return torch.LongTensor([user]).squeeze(), torch.LongTensor([pos]).squeeze(), torch.LongTensor([neg]).squeeze()

@torch.no_grad()
def recommend_topk(model, user_idx, seen_items, n_items, k=10, device='cpu'):
    model.eval()
    user_tensor = torch.full((n_items,), user_idx, dtype=torch.long, device=device)
    item_tensor = torch.arange(n_items, device=device)
    scores = model(user_tensor, item_tensor).cpu().numpy()
    scores[list(seen_items)] = -1e9
    topk = np.argpartition(-scores, k-1)[:k]
    topk = topk[np.argsort(-scores[topk])]
    return topk

def recall_at_k(model, df_eval, train_df, n_items, k=10, device='cpu'):
    model.eval()
    train_user_pos = train_df.groupby('user_idx')['product_idx'].apply(set).to_dict()
    recalls = []
    for user_idx, group in df_eval.groupby('user_idx'):
        actual_items = group['product_idx'].unique()
        seen_train = train_user_pos.get(user_idx, set())
        topk = recommend_topk(model, user_idx, seen_train, n_items, k, device)
        hits = sum(1 for it in actual_items if it in topk)
        recalls.append(hits / len(actual_items))
    return float(np.mean(recalls)) if len(recalls) > 0 else 0.0

def save_checkpoint(model, optimizer, epoch, path=CHECKPOINT_FILE):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer=None, path=CHECKPOINT_FILE, device='cpu'):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {path}, resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at {path}, starting fresh training.")
        return 0

def train_bpr(
    train_df,
    val_df,
    n_users,
    n_items,
    emb_dim=64,
    batch_size=4096,
    lr=1e-3,
    l2_reg=1e-5,
    epochs=30,
    device=None,
    checkpoint_path=CHECKPOINT_FILE,
):
    """
    Train a BPR model on train_df only.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = BPRDataset(train_df, n_items)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    model = BPRMatrixFactorization(n_users, n_items, emb_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)

    start_epoch = load_checkpoint(model, optimizer, path=checkpoint_path, device=device)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for users, pos_items, neg_items in train_loader:
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)

            pos_scores = model(users, pos_items)
            neg_scores = model(users, neg_items)
            loss_vals = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
            loss = loss_vals.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item() * users.size(0)

        t1 = time.time()
        avg_loss = epoch_loss / len(train_dataset)
        if (epoch + 1) % 10 == 0:
            recall_at_k_batch(model, val_df, train_df, n_items, k=50, device=device)
        print(f"Epoch {epoch+1}/{epochs} — BPR-loss: {avg_loss:.6f} — time: {t1-t0:.1f}s")

        save_checkpoint(model, optimizer, epoch, path=checkpoint_path)

    return model

@torch.no_grad()
def recommend_topk_batch(model, user_indices, seen_items_dict, n_items, k=10, device='cpu', batch_size=512):
    """
    Batch recommendation for multiple users at once.
    
    user_indices : list or array of user indices to evaluate
    seen_items_dict : dict mapping user -> set of already seen item indices
    n_items : total number of items
    k : number of top items to return
    batch_size : how many users to process in a batch
    """
    model.eval()
    topk_all = []

    all_items = torch.arange(n_items, device=device)

    for start in range(0, len(user_indices), batch_size):
        batch_users = user_indices[start:start+batch_size]
        batch_size_actual = len(batch_users)

        # Create user-item grid for batch scoring
        user_tensor = torch.tensor(batch_users, device=device).unsqueeze(1).repeat(1, n_items)
        item_tensor = all_items.unsqueeze(0).repeat(batch_size_actual, 1)

        # Compute scores: (batch_size_actual, n_items)
        scores = model(user_tensor, item_tensor)

        # Mask seen items
        for i, u in enumerate(batch_users):
            seen = seen_items_dict.get(u, set())
            if seen:
                scores[i, list(seen)] = -1e9

        # Top-k per user
        topk_indices = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
        topk_all.extend(topk_indices)

    return topk_all


def recall_at_k_batch(model, df_eval, train_df, n_items, k=10, device='cpu', batch_size=512):
    """
    Batched Recall@K evaluation for multiple users.
    """
    model.eval()
    # Build dict of seen items per user from training
    train_user_pos = train_df.groupby('user_idx')['product_idx'].apply(set).to_dict()
    users_eval = df_eval['user_idx'].unique()

    # Get top-k recommendations for all users in batches
    topk_all = recommend_topk_batch(model, users_eval, train_user_pos, n_items, k=k, device=device, batch_size=batch_size)

    recalls = []
    user_to_eval_items = df_eval.groupby('user_idx')['product_idx'].apply(list).to_dict()

    for u, topk in zip(users_eval, topk_all):
        actual_items = user_to_eval_items.get(u, [])
        if len(actual_items) == 0:
            continue
        hits = sum(1 for item in actual_items if item in topk)
        recalls.append(hits / len(actual_items))

    return float(np.mean(recalls)) if len(recalls) > 0 else 0.0


def sample_recommendations(
    model,
    train_df,
    item_to_title,
    users,
    n_last=5,
    k=10,
    n_sample=5,
    device='cpu'
):
    """
    Sample users and print last n_last items + top-k recommendations.
    """
    sample_users = np.random.choice(users, size=min(n_sample, len(users)), replace=False)
    print("\nSample recommendations (top-{} titles) and last {} purchases:".format(k, n_last))

    for u in sample_users:
        # last n_last purchased items
        user_train_items = (
            train_df[train_df['user_idx'] == u]
            .sort_values('timestamp', ascending=False)['product_idx']
            .values
        )
        last_titles = [item_to_title.get(int(i), str(i)) for i in user_train_items[:n_last]]

        # top-k recommendations
        seen_train = set(train_df[train_df['user_idx'] == u]['product_idx'].values)
        topk = recommend_topk(model, u, seen_train, n_items=item_to_title.__len__(), k=k, device=device)
        topk_titles = [item_to_title.get(int(i), str(i)) for i in topk]

        print(f"User {u} | Last {n_last} purchases: {last_titles} | Recommendations: {topk_titles}")


if __name__ == "__main__":
    train_df, val_df, test_df = load_preprocess_split()

    # Users/items
    n_users = int(train_df['user_idx'].max() + 1)
    all_items = pd.concat([train_df['product_idx'], val_df['product_idx'], test_df['product_idx']])
    n_items = int(all_items.max() + 1)
    
    item_to_title = train_df.drop_duplicates('product_idx').set_index('product_idx')['title'].to_dict()
    # Train model
    model = train_bpr(
        train_df=train_df,
        val_df=val_df,
        n_users=n_users,
        n_items=n_items,
        emb_dim=128,
        batch_size=4096,
        lr=1e-3,
        l2_reg=1e-5,
        epochs=30,
        device=None
    )

    # Evaluate test recall
    test_recall = recall_at_k(model, test_df, train_df, n_items, k=50, device=None)
    print(f"\nFinal Test Recall@50: {test_recall:.4f}")

    # Sample recommendations
    sample_recommendations(
        model=model,
        train_df=train_df,
        item_to_title=item_to_title,
        users=test_df['user_idx'].unique(),
        n_last=5,
        k=10,
        n_sample=5,
        device=None
    )