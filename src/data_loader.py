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

def load_preprocess_5core():
    
    meta_path = DATA_DIR / "meta_Office_Products.jsonl.gz"
    reviews_path = DATA_DIR / "Office_Products_5core.csv.gz"
    
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
    interaction_df = merged[['user_idx', 'product_idx', 'title', 'rating']]
    
    return interaction_df

    
    