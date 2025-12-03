#!/bin/bash

# Base setup directory (relative to project root)
DATA_DIR="$(pwd)/data" # path to your data directory

# Create directory if it doesn't exist
mkdir -p "$DATA_DIR"

echo "Downloading Amazon Electronics dataset to:"
echo "$DATA_DIR"
echo "-------------------------------------------"

wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Electronics.train.csv.gz \
  -O "$DATA_DIR/Electronics_train.csv.gz"

wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Electronics.valid.csv.gz \
  -O "$DATA_DIR/Electronics_valid.csv.gz"

wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Electronics.test.csv.gz \
  -O "$DATA_DIR/Electronics_test.csv.gz"

wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Electronics.csv.gz \
  -O "$DATA_DIR/Electronics_5core.csv.gz"

wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz \
  -O "$DATA_DIR/meta_Electronics.jsonl.gz"

echo "-------------------------------------------"
echo "Download complete."
