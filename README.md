# Amazon Fashion Recommendation System

This repository implements three recommendation approaches on the Amazon Fashion dataset for the 02807 Computational Tools for Data Science course. Each method corresponds to a specific algorithmic topic from the course and is placed in the `src/` directory.

## Repository Structure


.
├── data/                     # LOCAL ONLY (Amazon Fashion dataset, ignored by git)
│   ├── Amazon_Fashion.jsonl.gz
│   └── meta_Amazon_Fashion.jsonl.gz
│
├── results/                  # Script outputs (examples + LSH pkl files)
│   ├── lsh_examples.txt
│   ├── tfidf_examples.txt
│   └── cluster_examples.txt
│
├── src/
│   ├── data_loader.py        # Shared loader for gz JSONL -> pandas
│   ├── recommender_lsh.py    # MinHash + LSH (item–item CF)
│   ├── recommender_tfidf.py  # TF-IDF content-based recommender
│   ├── recommender_kmeans.py # K-means clustering on TF-IDF
│   └── query_lsh.py          # Optional: fast querying of saved LSH model
│
├── requirements.txt
└── README.md

## Dataset Setup (Required Before Running Any Script)

This project uses the **Amazon Fashion review and metadata dataset**, which is **not included in this repository** due to its large size.

You must manually download the dataset **before running any code** using **one of the following options**:

### Option 1: Manual Download  
Download the following files from the official Amazon Reviews 2023 site:  
https://amazon-reviews-2023.github.io/index.html  

Required files:
- `Amazon_Fashion.jsonl.gz`
- `meta_Amazon_Fashion.jsonl.gz`
- `meta_Electronics.jsonl.gz
- `Electronics - 5core`
- optional - train/valid/test split for Electronics

After downloading, place both files into the local directory:

```text
data/
├── Amazon_Fashion.jsonl.gz
└── meta_Amazon_Fashion.jsonl.gz
```text

### Option 2: Shell script
Adjust directory in `download_data.sh` and run the script

## 1. Shared Data Loader

### `src/data_loader.py`

Loads the gzipped JSONL Amazon Fashion metadata and review files into Pandas DataFrames. All recommenders depend on this module for consistent dataset access.

Run order: called automatically by all scripts.

---

## 2. Item–Item Collaborative Filtering (MinHash + LSH)

### Script: `src/recommender_lsh.py`  
### Course topic: Similar Items + Locality Sensitive Hashing (LSH)  
### Classification: Counts as a “new algorithm/topic” in the project

Implements item–item collaborative filtering using MinHash to approximate Jaccard similarity between item user-sets, and LSH banding to efficiently retrieve candidate neighbours.

Pipeline:
1. Load reviews and metadata  
2. Build item-to-user mappings (ratings ≥ threshold)  
3. Compute MinHash signatures  
4. Build LSH bands  
5. Retrieve top-k similar items for example products  
6. Save signatures, bands, and index-to-title mappings to `results/`  
7. Export human-readable examples to `results/lsh_examples.txt`

Run:
python -m src.recommender_lsh

### Query without retraining

After training once:

python -m src.query_lsh <item_idx> -k 5
Example: python -m src.query_lsh <70994> -k 5

Uses saved `.pkl` files for instant recommendations.

---

## 3. Content-Based Recommendations (TF-IDF + Cosine Similarity)

### Script: `src/recommender_tfidf.py`  
### Course topic: Vector Representation + Similarity Search  
### Classification: Does not count as a new topic (standard baseline)

Creates a content-based recommender using TF-IDF vectors computed from product titles. Cosine similarity identifies the top-k closest textual neighbours.

Pipeline:
1. Load metadata  
2. Build TF-IDF matrix  
3. Sample example items  
4. Compute top-k cosine similarities  
5. Export readable examples to `results/tfidf_examples.txt`

Run:
python -m src.recommender_tfidf

---

## 4. K-Means Clustering on TF-IDF Vectors

### Script: `src/recommender_kmeans.py`  
### Course topic: Clustering  
### Classification: Counts as a “new algorithm/topic” in the project

Clusters all products in the TF-IDF space using K-means to identify semantic item groups (e.g., graphic tees, maxi dresses, yoga leggings, Halloween costumes).

Pipeline:
1. Load metadata  
2. Build TF-IDF matrix  
3. Run K-means with configurable K  
4. Output top descriptive terms + sample items for 10 clusters  
5. Export cluster summaries to `results/cluster_examples.txt`

Run:
python -m src.recommender_kmeans

---

## Summary of Included Algorithms

| Script                    | Methodology                         | Course Topic            | New Topic? |
|--------------------------|--------------------------------------|--------------------------|------------|
| recommender_lsh.py       | MinHash + LSH (item–item CF)         | Similarity + LSH         | Yes        |
| recommender_tfidf.py     | TF-IDF + Cosine Similarity           | Vector models            | No         |
| recommender_kmeans.py    | K-Means Clustering on TF-IDF         | Clustering               | No         |
| query_lsh.py             | LSH querying utility                 | Auxiliary                | –          |

These components together satisfy the project requirement of combining multiple computational tools and algorithms: one standard technique, two “new” topics (LSH and clustering), and clear separation of content-based vs. behaviour-based approaches.


