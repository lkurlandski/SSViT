#!/bin/bash

# for t in "IterableBinaryDataset" "IterableBinaryDatasetBatchedLoader" "MapBinaryDataset" "MapBinaryDatasetBatchedLoader" "MapBinaryDatasetMemoryMapped"; do


for n in 0 1 2 4; do
 for t in IterableBinaryDataset IterableBinaryDatasetBatchedLoader MapBinaryDataset MapBinaryDatasetBatchedLoader MapBinaryDatasetMemoryMapped; do
   for b in 64; do

      python benchmarks/bmark_datasets.py \
        --type="$t" \
        --batch_size="$b" \
        --num_workers="$n" \
        --input=./data/ass \
        --num_samples=16384 \
	--outfile="./benchmarks/bmark_datasets.jsonl"

      echo ""

    done
  done
done
