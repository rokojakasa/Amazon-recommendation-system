# Amazon Fashion Recommendation System

This repository implements three recommendation approaches on the Amazon Fashion dataset for the 02807 Computational Tools for Data Science course. Each method corresponds to a specific algorithmic topic from the course and is placed in the `src/` directory.

## Repository Structure

```text
.
├── data/                     # LOCAL ONLY (Amazon Fashion dataset, ignored by git)
│   ├── Amazon_Fashion.jsonl.gz
│   └── meta_Amazon_Fashion.jsonl.gz
│
├── results/                  # Script outputs (examples + LSH pkl files)
│   ├── lsh_examples.txt
│   ├── tfidf_examples.txt
│   ├── cf_BPR_examples.txt
│   └── cluster_examples.txt
│
├── src/
│   ├── data_loader.py        # Shared loader for gz JSONL -> pandas
│   ├── recommender_lsh.py    # MinHash + LSH (item–item CF)
│   ├── recommender_cfBPR.py    # Matrix Factorization + BPR (User–item CF)
│   ├── recommender_tfidf.py  # TF-IDF content-based recommender
│   ├── recommender_kmeans.py # K-means clustering on TF-IDF
│   ├── benchmark_lsh.py # benchmark comparing our lsh with standard library
│   ├── benchmark_lsh_withram.py # benchmark comparing our lsh with standard        │library+ram usage check
│   ├── benchmark_bpr.py # benchmark comparing our bpr with standard library
│   ├── benchmarkvisualization.py # creates plots with data that was put manually after collecting
│   ├── kmeans_elbow_mykmeans.py #
│   ├── recommender_mykmeans.py #
│   ├── visualization.py # 
│   └── query_lsh.py          # Optional: fast querying of saved LSH model
│
├── download_data.sh          # Shell script for downloading requested data
├── run_hpc.sh                # Script for running a python file on HPC cluster
├── run_gpuv100.lsf           # Script for running python file on HPC with GPU
└── README.md
```
## Dataset Setup (Required Before Running Any Script)

This project uses the **Amazon Fashion and Electronics review and metadata dataset**, which is **not included in this repository** due to its large size.

You must manually download the dataset **before running any code** using **one of the following options**:

### Option 1: Manual Download  
Download the following files from the official Amazon Reviews 2023 site:  
https://amazon-reviews-2023.github.io/index.html  

Required files:
- `Amazon_Fashion.jsonl.gz`
- `meta_Amazon_Fashion.jsonl.gz`
- `meta_Electronics.jsonl.gz`
- Electronics - 5core
- optional - train/valid/test split for Electronics

After downloading, place files into the local directory:

```text
data/
├── Amazon_Fashion.jsonl.gz
└── meta_Amazon_Fashion.jsonl.gz
.
.
.
```

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

After training once, use query_lsh.py to sample recommendations.
Uses saved `.pkl` files for instant recommendations.

---

## 3. User–Item Collaborative Filtering (Matrix Factorization + BPR)

### Script: `src/recommender_cf_bpr.py`  
### Course topic: Matrix Factorization + Ranking Optimization  

Implements a **user–item collaborative filtering recommender** using **matrix factorization optimized with Bayesian Personalized Ranking (BPR)**. Unlike the LSH-based item–item method, which recommends similar products based on shared customers, this approach directly learns **personalized user preferences** from historical interactions.

Each user and each item is represented as a low-dimensional latent vector learned from data. Instead of predicting explicit star ratings, the model is trained using **implicit feedback** and a ranking objective: for each user, items they interacted with should be ranked higher than randomly sampled non-interacted items. This makes the approach well-suited for sparse Amazon review data.

Pipeline:
1. Load user–item interaction data  
2. Construct implicit feedback matrix  
3. Train matrix factorization model with BPR loss  
4. Sample negative items during training  
5. Generate Top-K recommendations per user  
6. Evaluate using Recall@K  
7. Export example recommendations to `results/`

Run:
```bash
python -m src.recommender_cf_bpr
```
---

## 4. Content-Based Recommendations (TF-IDF + Cosine Similarity)

### Script: `src/recommender_tfidf.py`  
### Course topic: Vector Representation + Similarity Search  

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

## 5. K-Means Clustering on TF-IDF Vectors

### Script: `src/recommender_kmeans.py`  
### Course topic: Clustering  

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

## 6. Benchmarking

### Script: `src/benchmark_lsh.py` `src/benchmark_lsh_withram.py` `src/benchmark_bpr.py`

#### LSH Benchmark (Speed & Scalability)

Benchmarks the raw execution speed of the custom implementation against datasketch to identify Big-O complexity behavior and latency bottlenecks.

Pipeline:

Load and downsample 5-core interaction data to target size (e.g., 50k, 200k, 15M rows).

Build Custom LSH Index (measure wall-clock time).

Build datasketch Index (measure wall-clock time).

execute random queries to measure average Latency per item.

Output execution time ratios (e.g., "5.8x faster").

Run: python -m src.benchmark_lsh

#### LSH Benchmark (Memory & Accuracy)

Profiles resource consumption and algorithmic correctness by tracking peak memory usage and comparing retrieved candidates against a brute-force ground truth.

Pipeline:

Load data and enable tracemalloc memory tracing.

Build indices for both implementations (capture Peak RAM usage in MB).

For a set of query items, compute the Exact Jaccard top-5 neighbors (Ground Truth).

Query both LSH indices and check for intersections with the Ground Truth.

Compute and export Recall@5 percentages and Memory footprints.

Run: python -m src.benchmark_lsh_withram

#### Matrix Factorization (BPR) Benchmark

Benchmarks the training efficiency of a flexible custom PyTorch implementation against the optimized implicit library to quantify the overhead of Python-based sampling.

Pipeline:

Load pre-split training, validation, and test data.

Initialize models with identical embedding dimensions (d=128).

Train custom PyTorch model for 1 epoch on CPU (measure time).

Train implicit library model for 1 epoch on CPU (measure time).

Output speedup factor (e.g., "740x faster") to console.

Run: python -m src.benchmark_bpr

## Summary of Included Algorithms
| Script                    | Methodology                               | Course Topic                     | New Topic? |
|---------------------------|--------------------------------------------|----------------------------------|------------|
| recommender_lsh.py        | MinHash + LSH (item–item CF)               | Similarity + LSH                 | No        |
| recommender_cfBPR.py      | Matrix Factorization + BPR (user–item CF)  | Matrix Factorization + Ranking   |  Yes       |
| recommender_tfidf.py      | TF-IDF + Cosine Similarity                 | Vector models                    | No         |
| recommender_kmeans.py     | K-Means Clustering on TF-IDF               | Clustering                       | No         |
| query_lsh.py              | LSH querying utility                       | Auxiliary                        | –          |


