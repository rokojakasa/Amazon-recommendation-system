from src.data_loader import load_preprocess_5core, get_df, load_preprocess_split
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, k):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, k)
        self.item_factors = nn.Embedding(n_items, k)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)
        self.user_biases.weight.data.fill_(0)
        self.item_biases.weight.data.fill_(0)
        

    def forward(self, users, items):
        u = self.user_factors(users)
        i = self.item_factors(items)
        
        dot = (u * i).sum(dim=1)

        bu = self.user_biases(users).squeeze()
        bi = self.item_biases(items).squeeze()

        return dot + bu + bi
    
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["product_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

@torch.no_grad()
def recommend(model, user_idx, seen_items, n_items, k=10):
    device = next(model.parameters()).device
    
    user = torch.full((n_items, ), user_idx, dtype=torch.long, device=device)
    items = torch.arange(n_items, device=device)
    
    scores = model(user, items).cpu().numpy()
    scores[list(seen_items)] = -1e9  # Exclude seen items   
    
    top_items = scores.argsort()[::-1][:k]
    return top_items

def train(model, device, epochs, loader, dataset, lr, reg):
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
  loss_fn = nn.MSELoss()

  for epoch in range(10):
    total_loss = 0.0

    for users, items, ratings in loader:
        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)

        preds = model(users, items)
        loss = loss_fn(preds, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(ratings)

    rmse = np.sqrt(total_loss / len(dataset))
    print(f"Epoch {epoch+1}, RMSE = {rmse:.4f}")
    
@torch.no_grad()
def recall_at_k(model, df_eval, train_df, n_items, k=10):
    """
    Compute Recall@K over evaluation dataframe.
    Exclude items the user has already seen in training.
    """
    recalls = []

    # Group by user
    for user, user_df in df_eval.groupby('user_idx'):
        seen_train_items = set(train_df[train_df['user_idx']==user]['product_idx'].values)
        actual_items = user_df['product_idx'].values  # 1 or 2 items

        # Get top-k recommendations
        top_k = recommend(model, user, seen_train_items, n_items, k=k)

        # Recall@K = fraction of actual items in top-k
        hits = sum([1 for item in actual_items if item in top_k])
        recalls.append(hits / len(actual_items))

    return np.mean(recalls)

def main():
    # METADATA_PATH = "/zhome/4f/1/223566/home/computational_tools/Amazon-recommendation-system/data/meta_Office_Products.jsonl.gz"
    
    interractive_df = load_preprocess_5core()
    print (interractive_df.columns)
    interractive_df.head(20)

    n_users = interractive_df['user_idx'].nunique()
    n_items = interractive_df['product_idx'].nunique()
    print(f"Number of users: {n_users}")
    print(f"Number of products: {n_items}")

    k = 32
    lr = 0.01
    reg = 0.05
    epochs = 50



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatrixFactorization(n_users, n_items, k).to(device)
    dataset = RatingsDataset(interractive_df)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    train(model, device, epochs, loader, dataset, lr, reg)
    test_user = 0

    user_interactions = interractive_df[interractive_df["user_idx"] == test_user]
    user_interactions_sorted = user_interactions.sort_values("timestamp", ascending=False)
    last_5_seen = user_interactions_sorted["title"].values[:5]

    print(f"Last 5 items seen by user {test_user}: {last_5_seen}")

    # Get all seen items to exclude from recommendation
    seen_items = user_interactions["product_idx"].values

    recs = recommend(model, test_user, set(seen_items), n_items, k=10)
    print(f"Recommendations for user {test_user}:")
    print(recs)
    print(f"Top-10 recommendations for user {test_user}: {recs}")

    rec_titles = interractive_df.drop_duplicates("product_idx").set_index("product_idx").loc[recs]["title"].values
    print(f"Recommended item titles: {rec_titles}")

    
    # train_df, val_df, test_df = load_preprocess_split()
    # meta_df = get_df(METADATA_PATH)
    # print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    # # Config for model and train
    # k = 32
    # lr = 0.01
    # reg = 0.05
    # epochs = 50
    
    # train_df['user_idx'], user_to_idx_arr = pd.factorize(train_df['user_idx'])
    # train_df['product_idx'], product_to_idx_arr = pd.factorize(train_df['product_idx'])

    # def map_idx(x, arr):
    #     matches = np.where(arr == x)[0]
    #     return matches[0] if len(matches) > 0 else -1  # -1 for unseen

    # val_df['user_idx'] = val_df['user_idx'].map(lambda x: map_idx(x, user_to_idx_arr))
    # val_df['product_idx'] = val_df['product_idx'].map(lambda x: map_idx(x, product_to_idx_arr))
    # val_df = val_df[(val_df['user_idx'] != -1) & (val_df['product_idx'] != -1)]

    # test_df['user_idx'] = test_df['user_idx'].map(lambda x: map_idx(x, user_to_idx_arr))
    # test_df['product_idx'] = test_df['product_idx'].map(lambda x: map_idx(x, product_to_idx_arr))
    # test_df = test_df[(test_df['user_idx'] != -1) & (test_df['product_idx'] != -1)]

    # # full_df = pd.concat([train_df, val_df, test_df])
    # n_users = train_df.user_idx.nunique()
    # n_items = train_df.product_idx.nunique()

    # print(f"Number of users: {n_users}")
    # print(f"Number of products: {n_items}")

    # # Create datasets and loaders
    # train_dataset = RatingsDataset(train_df)
    # train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MatrixFactorization(n_users, n_items, k).to(device)

    # train(model, device, epochs, train_loader, train_dataset, lr, reg)
    
    # # After training
    # val_recall = recall_at_k(model, val_df, train_df, n_items, k=10)
    # test_recall = recall_at_k(model, test_df, train_df, n_items, k=10)

    # print(f"Validation Recall@10: {val_recall:.4f}")
    # print(f"Test Recall@10: {test_recall:.4f}")



    
if __name__ == "__main__":
    main()