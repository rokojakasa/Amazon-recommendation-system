import gzip
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from datasets import load_dataset


# Root dir = repo root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def parse(path: Path):
    """Yield one JSON object per line from a .jsonl.gz file."""
    with gzip.open(path, "rb") as g:
        for line in g:
            yield json.loads(line)


def get_df(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read a .jsonl.gz file into a pandas DataFrame."""
    records = {}
    for i, d in enumerate(parse(path)):
        if max_rows is not None and i >= max_rows:
            break
        records[i] = d
    return pd.DataFrame.from_dict(records, orient="index")


def load_amazon_fashion(
    max_rows_meta: Optional[int] = None,
    max_rows_reviews: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Amazon Fashion metadata and reviews.

    Files are expected at:
    - data/meta_Amazon_Fashion.jsonl.gz
    - data/Amazon_Fashion.jsonl.gz
    """
    meta_path = DATA_DIR / "meta_Amazon_Fashion.jsonl.gz"
    reviews_path = DATA_DIR / "Amazon_Fashion.jsonl.gz"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    if not reviews_path.exists():
        raise FileNotFoundError(f"Reviews file not found: {reviews_path}")

    meta_df = get_df(meta_path, max_rows_meta)
    reviews_df = get_df(reviews_path, max_rows_reviews)
    return meta_df, reviews_df

def load_preprocess_5core(meta_filename, reviews_filename):
    
    meta_path = DATA_DIR / meta_filename
    reviews_path = DATA_DIR / reviews_filename
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    if not reviews_path.exists():
        raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
    
    reviews_df = pd.read_csv(reviews_path, compression='gzip')
    meta_df = get_df(meta_path)
    
    interaction_df = build_interaction_df(meta_df, reviews_df)
    return interaction_df
    
def build_interaction_df(meta_df: pd.DataFrame, reviews_df: pd.DataFrame):
    """
    Convert reviews and meta into a compact interaction DataFrame.
    
    Returns:
        interaction_df: DataFrame with ['user_idx', 'product_idx', 'title', 'rating']
        user_to_idx: dict mapping user_id -> user_idx
        product_to_idx: dict mapping parent_asin -> product_idx
    """
    # Merge titles from meta
    merged = reviews_df.merge(
        meta_df[['parent_asin', 'title']].drop_duplicates(),
        on='parent_asin',
        how='left'
    )
    
    # Factorize user_id and product_id to integers
    merged['user_idx'], user_to_idx_arr = pd.factorize(merged['user_id'])
    merged['product_idx'], product_to_idx_arr = pd.factorize(merged['parent_asin'])

    
    # Keep only necessary columns
    interaction_df = merged[['user_idx', 'product_idx', 'title', 'rating', 'timestamp']]
    
    return interaction_df

def load_preprocess_split(train_filename, val_filename, test_filename, meta_filename):
    train_path = DATA_DIR / train_filename
    val_path = DATA_DIR / val_filename
    test_path = DATA_DIR / test_filename
    meta_path = DATA_DIR / meta_filename
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    train_df = pd.read_csv(train_path, compression='gzip')
    val_df = pd.read_csv(val_path, compression='gzip')
    test_df = pd.read_csv(test_path, compression='gzip')
    
    meta_df = get_df(meta_path)
    
    train_df, val_df, test_df = build_interaction_df_eval(train_df, val_df, test_df, meta_df)
    
    return train_df, val_df, test_df

def build_interaction_df_eval(train_df, val_df, test_df, meta_df):
    
        # Merge titles from meta
    def merge_meta(df):
        return df.merge(
            meta_df[['parent_asin', 'title']].drop_duplicates(),
            on='parent_asin',
            how='left'
        )
    
    # Factorize user_id and product_id to integers
    train_df = merge_meta(train_df)
    val_df = merge_meta(val_df)
    test_df = merge_meta(test_df)

    unique_products = meta_df['parent_asin'].unique()
    product_to_idx = {pid: i for i, pid in enumerate(unique_products)}
    
    unique_users = train_df['user_id'].unique()
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    
    def map_product_user_idx(df):
        df['product_idx'] = df['parent_asin'].map(product_to_idx)
        df['user_idx'] = df['user_id'].map(user_to_idx)
        return df[(df['product_idx'] >= 0) & (df['user_idx'] >= 0)]
    
    train_df = map_product_user_idx(train_df)
    val_df = map_product_user_idx(val_df)
    test_df = map_product_user_idx(test_df)
    
    # Keep only necessary columns
    keep_cols = ['user_idx', 'product_idx', 'title', 'rating', 'timestamp']
    train_df = train_df[keep_cols]
    val_df = val_df[keep_cols]
    test_df = test_df[keep_cols]
    
    # Keep only necessary columns
    return train_df, val_df, test_df
    