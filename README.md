# CC-LightGCN: A Framework for Niche Recommendation Systems

> Popularity bias causes GCN recommenders to ignore long-tail items. CC-LightGCN fixes this on MovieLens-100K, achieving **+226% Tail Percentage** at K=10 vs. the best baseline while staying competitive on accuracy.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![LightGCN](https://img.shields.io/badge/LightGCN-based-orange)
![InfoNCE](https://img.shields.io/badge/Loss-BPR%20%2B%20InfoNCE-purple)

---

## Project Summary

Most recommender systems have a popularity bias problem. A small set of "head" items dominate the training data, so GCN models like LightGCN end up amplifying that skew and rarely recommending niche "long-tail" content.

CC-LightGCN tackles this with two ideas working together: **counterfactual graph augmentation** (swap some popular-item edges with genre-similar tail items during training) and **multi-view contrastive learning** (align embeddings from real, noisy, and counterfactual graph views using InfoNCE loss).

Evaluated on MovieLens-100K against five baselines, CC-LightGCN achieves TP@10 of **0.0828** and TP@20 of **0.1008**, which is 226% and 298% better than the next-best model, while keeping HR and NDCG in a reasonable range.

---

## Results Progression

All models evaluated on MovieLens-100K. Metrics averaged over 10 repeated runs (mean ± std). Higher is better for all metrics. **Bold** = best per metric. Negative improvement means a baseline outperforms CC-LightGCN on that metric (accuracy vs. diversity tradeoff).

| Model | Type | HR@10 | HR@20 | NDCG@10 | NDCG@20 | TP@10 | TP@20 | Notes |
|---|---|---|---|---|---|---|---|---|
| NGCF (Wang et al., 2019) | Baseline GCN | 0.6978 ± 0.0118 | 0.8229 ± 0.0120 | 0.2182 ± 0.0035 | 0.2450 ± 0.0030 | 0.0000 | 0.0000 | Heavy non-linear layers; zero tail exposure |
| LightGCN (He et al., 2020) | Baseline GCN | 0.6734 ± 0.0038 | 0.8010 ± 0.0120 | 0.2142 ± 0.0037 | 0.2308 ± 0.0061 | 0.0000 | 0.0000 | Ultra-fast; purely linear but amplifies bias |
| SGL (Wang et al., 2021) | + Contrastive (random aug) | **0.7303 ± 0.0090** | **0.8466 ± 0.0088** | **0.2521 ± 0.0022** | **0.2691 ± 0.0040** | 0.0032 ± 0.0012 | 0.0032 ± 0.0011 | Best accuracy; augmentations are bias-agnostic |
| SimGCL (Yu et al., 2022) | + Contrastive (noise) | 0.4758 ± 0.0085 | 0.6416 ± 0.0212 | 0.1079 ± 0.0018 | 0.1288 ± 0.0032 | 0.0254 ± 0.0158 | 0.0253 ± 0.0126 | First meaningful tail exposure; accuracy drops |
| HMLET (Kim et al., 2023) | Hybrid linear/non-linear | 0.6709 ± 0.0050 | 0.7922 ± 0.0046 | 0.2039 ± 0.0006 | 0.2202 ± 0.0034 | 0.0000 | 0.0000 | Gating module adds complexity; no debiasing |
| **CC-LightGCN (Ours)** | Counterfactual + CL | 0.5806 ± 0.0189 | 0.7006 ± 0.0318 | 0.1586 ± 0.0053 | 0.1720 ± 0.0056 | **0.0828 ± 0.0143** | **0.1008 ± 0.0089** | **+226% TP@10, +298% TP@20 vs. best baseline** |

All prior GCN models score TP = 0 or near-zero. CC-LightGCN is the only model that consistently surfaces long-tail content, confirming that the counterfactual and contrastive combination is what drives the change.

---

## The Story

### Chapter 1: Two Views of Reality — What the Model Actually Learns

![CC-LightGCN Architecture](https://raw.githubusercontent.com/suparuek2405/CC-LightGCN/main/results/figures/Fig1.png)

A model trained only on real interactions will always inherit the popularity bias baked into that data. CC-LightGCN breaks this by encoding two parallel graph views at the same time: the real interaction graph and a counterfactual graph where some head-item edges are swapped with genre-similar tail items (e.g., "Star Wars" to "Gattaca" at cosine similarity above sim_thresh). A third view adds Gaussian noise to the real embeddings for robustness.

All three views are aligned through InfoNCE contrastive loss, while the real view is also supervised by BPR ranking loss. The total loss L = BPR + λ x InfoNCE forces the model to be both accurate and tail-aware. Without the counterfactual view, the contrastive signal stays bias-agnostic (like SGL or SimGCL). Without the contrastive loss, the counterfactual edges have no regularization effect. Both pieces are required.

---

### Chapter 2: Diversity Without Sacrificing Relevance — What CC-LightGCN Actually Recommends

![Inference Example](https://raw.githubusercontent.com/suparuek2405/CC-LightGCN/main/results/figures/Fig2_credit_stockimages.png)

Given User A's watch history (Star Wars, Resident Evil, Alien), the top-5 list from CC-LightGCN includes three popular titles (Star Trek, Batman, Men in Black) and two niche ones (Dumbo, Gattaca). The tail items are genre-coherent, not random. They appear at positions 4 and 5, meaning the model genuinely shifted their ranking scores during training, not just added them to a candidate pool.

This is exactly what Tail Percentage captures, and it is only possible because training exposed the model to counterfactual "what-if" worlds. The augmentation step, not just the contrastive loss, is doing the real work.

---

### Chapter 3: Compact Embeddings, Better Generalization

![Embedding Dimension Tuning](https://raw.githubusercontent.com/suparuek2405/CC-LightGCN/main/results/figures/Fig3.png)

Sweeping embed_size from 4 to 64 shows a clear peak at **embed_size = 8** for HR@10 and NDCG@10, then a steady decline — classic overfitting on a small dataset (943 users, 1,682 items). Tail Percentage tells a different story: it is highest at embed_size = 4, collapses at 8 to 16, then slowly recovers at 32 to 64.

Very small embeddings act as a strong regularizer that incidentally helps diversity. Mid-size embeddings over-specialize on head items. The final model uses embed_size = 8 to get the best accuracy, accepting the TP dip as a known tradeoff.

---

### Chapter 4: Temperature Controls How Hard the Model Competes

![SSL Temperature Tuning](https://raw.githubusercontent.com/suparuek2405/CC-LightGCN/main/results/figures/Fig4.png)

The InfoNCE temperature (ssl_temp) controls how sharply the model separates positive from negative pairs. Lower values push harder; higher values are more relaxed. Sweeping from 0.1 to 0.3 shows TP@10 peaks at **ssl_temp = 0.15** and degrades after that, while HR@10 keeps climbing to 0.3.

The two objectives are in real tension: a sharper contrastive loss improves tail diversity but hurts accuracy. The operating point at ssl_temp = 0.15 is a deliberate tradeoff. Small steps of 0.05 produce measurable swings across all three metrics, making this the most sensitive hyperparameter in the model.

---

## Key Technical Decisions & Lessons

**1. Bias-agnostic augmentation is not enough.**
SGL and SimGCL both use contrastive learning with random or noise-based augmentation. Neither gets meaningful tail exposure (TP near 0). Contrastive learning only debiases if the augmentation is explicitly bias-aware. Random dropout changes how often the model sees an item, not what it sees. Counterfactual substitution changes what it sees, which is the necessary step.

**2. Semantic similarity threshold (sim_thresh) matters a lot.**
Too low and head items get replaced with unrelated tail items, adding noise. Too high and almost nothing gets substituted, collapsing the counterfactual graph into the real one. A cosine similarity threshold around 0.5 to 0.6 preserved genre coherence while still touching enough interactions to make a difference.

**3. Small datasets punish large embeddings harder than you expect.**
With only 943 users and 1,682 items, embed_size = 8 is already sufficient. The HR gap between embed_size = 8 and 64 is about 8 percentage points. Model capacity needs to match dataset size, not benchmark conventions from larger datasets.

**4. The BPR + InfoNCE balance is the core engineering tension.**
The ssl_weight (λ) controls how much the diversity objective shapes training. Too high and accuracy collapses. Too low and the contrastive loss has no effect. The useful zone is roughly λ = 0.11 to 0.13, and the model is sensitive within that range.

**5. You cannot fully escape the accuracy-diversity tradeoff.**
CC-LightGCN pays a 20 to 37% cost in HR and NDCG compared to SGL. That is not a flaw in the method. It reflects a real tradeoff that is invisible when you only optimize for accuracy. Picking the right evaluation metric is just as important as picking the right model.

---

## Final Model

**How the architecture works**

All 1,682 movies are first split into "head" (popular) and "tail" (niche) groups by interaction count. For each user interaction with a head item, CC-LightGCN finds a genre-similar tail item (cosine similarity above sim_thresh) and creates a counterfactual edge, as if the user had watched that tail item instead.

The LightGCN encoder runs on both the real graph and the counterfactual graph, producing two sets of embeddings. A third set comes from adding small Gaussian noise to the real embeddings. InfoNCE loss aligns all three views. BPR loss on the real embeddings keeps ranking accurate. At inference time, only the real embeddings are used.

**Architecture Components**

| Component | Role | Key Parameter |
|---|---|---|
| LightGCN Encoder (real graph) | Learns user/item representations from observed interactions | K layers, embed_size |
| Head-tail substitution | Builds counterfactual graph by swapping head edges with tail edges | head_ratio, tail_ratio, cf_ratio, sim_thresh |
| LightGCN Encoder (counterfactual graph) | Encodes the "what-if" world | Shared weights with real encoder |
| Gaussian noise perturbation | Generates a third view for robustness | noise_std |
| BPR Loss | Ranking loss on real embeddings | (none) |
| InfoNCE Loss | Aligns real, noise, and counterfactual views | ssl_temp |
| Total Loss combiner | Weighted sum of BPR and InfoNCE | ssl_weight (λ) |

**Best Hyperparameters**

| Hyperparameter | Value | What it controls |
|---|---|---|
| embed_size | 8 | Embedding dimension for users and items |
| K layers | 3 | Depth of graph convolution propagation |
| learning_rate | 1e-3 | Adam optimizer step size |
| weight_decay | 1e-4 | L2 regularization to prevent overfitting |
| head_ratio | 0.2 | Top 20% most interacted items classified as "head" |
| tail_ratio | 0.4 | Bottom 40% least interacted items classified as "tail" |
| cf_ratio | 0.3 | Fraction of head interactions substituted per training epoch |
| sim_thresh | 0.5 | Minimum cosine similarity for a valid head-to-tail swap |
| noise_std | 0.1 | Standard deviation of Gaussian noise added to real embeddings |
| ssl_temp | 0.15 | InfoNCE softmax temperature (lower = sharper contrastive signal) |
| ssl_weight (λ) | 0.13 | Weight of InfoNCE in total loss L = BPR + λ x InfoNCE |
| epoch | 200 | Maximum training epochs |
| stopping_step | 20 | Early stopping patience (epochs without val improvement) |

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| PyTorch | 2.0+ | GCN encoder, loss functions, backpropagation |
| NumPy | 1.24+ | Matrix operations and graph adjacency construction |
| Pandas | 2.0+ | Data loading and preprocessing |
| Scikit-learn | 1.3+ | Cosine similarity computation for head-tail matching |
| Matplotlib | 3.7+ | Hyperparameter sweep visualization |
| PyYAML | 6.0+ | Config file parsing |

---

## Dataset

[MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) — GroupLens Research, University of Minnesota.

| Stat | Value | Notes |
|---|---|---|
| Users | 943 | Demographic info available (age, gender, occupation) |
| Movies | 1,682 | Genre labels available (Drama 31%, Comedy 30%, Action 17%, Sci-Fi 7%) |
| Interactions | 122,762 | Explicit ratings 1-5; binarized to implicit feedback for training |
| Sparsity | 93.7% | Fraction of unobserved user-item pairs |
| Popularity skew | Top 10% movies = 48% of all ratings | Confirms strong head bias requiring debiasing |
| Split | 80% train / 10% val / 10% test | Leave-1-out: 1 interaction per user held out for evaluation |

---

## References

1. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and powering graph convolution network for recommendation. *SIGIR 2020*. https://doi.org/10.1145/3397271.3401063

2. Wang, X., He, X., Wang, M., Feng, F., & Chua, T.-S. (2019). Neural graph collaborative filtering. *SIGIR 2019*. https://doi.org/10.1145/3331184.3331267

3. Wang, J., Jiang, J., An, W., Quan, Y., Zheng, K., & Chen, C. (2021). Self-supervised graph learning for recommendation. *SIGIR 2021*. https://doi.org/10.1145/3404835.3462862

4. Yu, J., Yin, H., Xia, X., Chen, T., Cui, L., & Nguyen, Q. V. H. (2022). Are graph augmentations necessary? Simple graph contrastive learning for recommendation. *SIGIR 2022*. https://doi.org/10.1145/3477495.3531937

5. Kim, J., Kim, H., & Kim, J. (2023). HMLET: Hybrid method of linear and non-linear collaborative filtering. *RecSys 2023*. https://doi.org/10.1145/3604915.3608822

6. Harper, F. M., & Konstan, J. A. (2016). The MovieLens datasets: History and context. *ACM TIIS, 5*(4), 19:1-19:19. https://doi.org/10.1145/2827872
