# GameSense Recommender Evaluation Report

This document evaluates the Performance of Collaborative Filtering Matrix Factorization against baselines for the **GameSense** recommender engine.

## Dataset Statistics
- **Pre-filtering interactions:** 1005
- **Post-filtering interactions:** 1005
- **Unique Users:** 100
- **Unique Games:** 50

## Best Model Hyperparameters
- **Embedding Dimensions:** 16
- **L2 Regularization (Weight Decay):** 0.0001

## Metrics Comparison (K=10)

| Model | Hit@10 | Recall@10 |
|---|---|---|
| **Popularity Baseline** | 0.9000 | 0.1948 |
| **TF-IDF Content-Based Baseline** | 0.9000 | 0.1948 |
| **Collaborative Filtering (MF)** | 0.9000 | 0.1841 |

