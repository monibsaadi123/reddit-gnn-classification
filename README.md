# Reddit GNN Classification

This project implements and compares three Graph Neural Network (GNN) models—Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and GraphSAGE—for node classification on the Reddit dataset. The goal is to classify posts into their respective subreddits based on graph structure and text embeddings.

## Project Overview

We analyze a large Reddit interaction graph with:
- **232,965 nodes** (Reddit posts)
- **114,615,892 edges** (user-commented connections)
- **602 features per node** using 300-dimensional GloVe vectors
- **41 classes** (subreddits)

Each post is a node, and an edge connects posts commented on by the same user. The task is to predict the subreddit (class) of each post.

## Objectives

- Evaluate GCN, GAT, and GraphSAGE for node classification
- Measure and compare accuracy, loss, and generalization
- Identify performance bottlenecks and model suitability

## Dataset

- **Source**: Reddit (September 2014 snapshot)
- **Nodes**: 232,965 posts
- **Edges**: 114,615,892
- **Classes**: 41 subreddit labels
- **Features**: GloVe-based text embeddings (602 per node)

## GNN Models Implemented

### Graph Convolutional Network (GCN)
- Aggregates features from neighboring nodes using graph convolutions
- Best performance with **88.9% test accuracy**

### Graph Attention Network (GAT)
- Uses attention weights to weigh neighbor contributions
- Test accuracy: **88.5%**

### GraphSAGE
- Inductive model with neighbor sampling
- Test accuracy: **87.9%**

## Model Architecture & Training

- Framework: PyTorch
- Hidden dimension: 256
- Layers: 2
- Attention heads (GAT): 2
- Optimizer: Adam
- Batch size: 1024
- Dropout & ReLU activation

## Results Summary

| Model     | Test Accuracy | Training Accuracy | Validation Accuracy |
|-----------|----------------|--------------------|----------------------|
| GCN       | 88.9%          | High               | High                 |
| GAT       | 88.5%          | Lower              | Moderate             |
| GraphSAGE | 87.9%          | High               | Moderate             |

- **GCN** showed the best generalization.
- All models performed similarly due to strong node features and community structure.

## Key Insights

- Node embeddings (GloVe) and dense community connections contributed heavily to high performance.
- GraphSAGE is ideal for inductive tasks, while GCN performed best overall on this static dataset.
- Attention (GAT) helped in learning nuanced neighborhood relationships.

## Future Work

- Test models on dynamic or heterogeneous graphs
- Explore transformer-based GNNs for richer representations
- Apply on real-time or streaming graph datasets

## Authors

- Monib Saadi – edusaamon001@fh-kaernten.at  
- Aditya Babar – edubabadi001@fh-kaernten.ac.at  
FH Kärnten, Applied Data Science

## References

1. Hamilton, Ying, Leskovec – *Inductive Representation Learning on Large Graphs*
2. Kipf & Welling – *Semi-Supervised Classification with Graph Convolutional Networks*
3. Velickovic et al. – *Graph Attention Networks*
4. GNN reviews and tutorials from Medium, arXiv

