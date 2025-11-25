import pandas as pd
import gzip
import json
import numpy as np
from tqdm import tqdm

METADATA_PATH = "/path/to/meta_Amazon_Fashion.jsonl.gz"
REVIEWS_PATH = "/path/to/Amazon_Fashion.jsonl.gz"

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

meta_df = getDF(METADATA_PATH)
reviews_df = getDF(REVIEWS_PATH)
# print(reviews_df.columns)
# print(meta_df.columns)
# print(meta_df[['parent_asin', 'title']].head(20))


unique_users = reviews_df['user_id'].unique()
print(f"Number of unique users: {len(unique_users)}")

user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
idx_to_user = {i: uid for i, uid in enumerate(unique_users)}

unique_products = reviews_df['parent_asin'].unique()
print(f"Number of unique products: {len(unique_products)}")

product_to_idx = {pid: i for i, pid in enumerate(unique_products)}
idx_to_product = {i: pid for i, pid in enumerate(unique_products)}

reviews_df['user_idx'] = reviews_df['user_id'].map(user_to_idx)
reviews_df['product_idx'] = reviews_df['parent_asin'].map(product_to_idx)

# reviews_df.head()

print("Loaded all datasets")

# We will need this dictionary later for displaying product titles
asin_to_title = dict(zip(meta_df['parent_asin'], meta_df['title']))

idx_to_title = {
    idx: asin_to_title.get(asin, "Unknown Title")
    for idx, asin in idx_to_product.items()
}

def get_product_title(product_idx):
    return idx_to_title.get(product_idx, "Unknown Title")

# print(reviews_df.iloc[250, :])
# print(get_product_title(250))
# print(get_product_title(1000))

from collections import defaultdict

item_to_users = defaultdict(set)
# We only save the products user rated >= 3, so the recommendation system wouldn't use something a user did not enjoy/like
for _, row in reviews_df.iterrows():
    if row['rating'] >= 3:
        item_to_users[row['product_idx']].add(row['user_idx'])


# pip install mmh3
import mmh3
import random

NUM_HASHES = 128
MAX_SEED = 2**32 - 1
hash_seeds = random.sample(range(MAX_SEED), NUM_HASHES)

def minhash_signature(user_set):
  signature = []
  for i in range (NUM_HASHES):
    m = min([mmh3.hash(str(user), hash_seeds[i]) for user in user_set])
    signature.append(m)
  return signature

product_signatures = {}
for product, users in tqdm(item_to_users.items(), desc="Computing MinHash signatures"):
    product_signatures[product] = minhash_signature(users)



from itertools import combinations

NUM_BANDS = 32
ROWS_PER_BAND = NUM_HASHES // NUM_BANDS

bands = [defaultdict(list) for _ in range(NUM_BANDS)]

for product, signature in tqdm(product_signatures.items(), desc="Assigning LSH bands"):
  for i in range(NUM_BANDS):
    start_idx = i * ROWS_PER_BAND
    end_idx = (i + 1) * ROWS_PER_BAND
    band_signature = tuple(signature[start_idx:end_idx])
    bands[i][band_signature].append(product)


def get_candidate_products(product_id):
  signature = product_signatures[product_id]
  candidate_products = set()
  for i in range(NUM_BANDS):
    start_idx = i * ROWS_PER_BAND
    end_idx = (i + 1) * ROWS_PER_BAND
    band_signature = tuple(signature[start_idx:end_idx])
    candidate_products.update(bands[i][band_signature])
  candidate_products.discard(product_id)
  return candidate_products

def estimated_jaccard(product1, product2):
    sig1 = product_signatures[product1]
    sig2 = product_signatures[product2]
    matches = sum(1 for i in range(NUM_HASHES) if sig1[i] == sig2[i])
    return matches / NUM_HASHES

def get_top_k_similar_products(product_id, k=5):
  candidates = get_candidate_products(product_id)
  similarities = {}
  for candidate in candidates:
    sim = estimated_jaccard(product_id, candidate)
    similarities[candidate] = sim
  top_k = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
  return top_k

# print("Sanity check")
product_id = 250
k = 5
top_k_similar = get_top_k_similar_products(product_id, k)

print(f"Top 5 similar products to: {get_product_title(product_id)}\n")

for pid, score in top_k_similar:
    print(f"- {get_product_title(pid)}  (score={score:.4f})")