# Netflix Prize Recommender System (SVD vs. Collaborative Filtering)

An algorithmic engineering project designed to evaluate, validate, and compare Latent-Factor Matrix Factorization against Neighborhood-Based Collaborative Filtering using historical explicit transaction records from the **Netflix Prize Dataset**.

---

## Repository Architecture

* **`netflix_recsys (1).ipynb`**: The primary end-to-end engineering pipeline. It contains:
  * **Data Processing Pipeline**: Slicing, cleaning, and handling extreme matrix sparsity.
  * [cite_start]**Model Training Pipeline**: Batch configuration setups for SVD and Item-Based CF models.
  * [cite_start]**Evaluation Scripts**: Validation routines tracking RMSE, MAE, and MAP@10 metrics.

---

## Hyperparameter Boundary Configurations

The modeling mechanics within the pipeline were constrained to the following synchronized execution boundaries during cross-validation training:

### Model 1: Singular Value Decomposition (SVD) Core
* **Latent Space Dimensions (`n_factors`)**: 100 [cite: 33]
* **Training Epoch Iterations (`n_epochs`)**: 30 [cite: 33]
* **Target Learning Rate (`lr_all`)**: 0.005 [cite: 33]
* **Ridge Regularization Penalty (`reg_all`)**: 0.02 [cite: 33]
* **Active Training Boundaries**: 1,000,000 rating rows [cite: 33]

### Model 2: Item-Based Collaborative Filtering (Baseline)
* [cite_start]**Proximity Metric Framework**: Cosine Similarity [cite: 36]
* [cite_start]**Max Target Neighborhoods (`k neighbors`)**: 20 [cite: 36]
* **Orientation Paradigm**: Item-Centric (`user_based = False`) [cite: 36]
* [cite_start]**Active Training Boundaries**: 50,000 rating rows [cite: 36]

---

## Experimental Validation Benchmarks

Both architectures were cross-evaluated on a synchronized 80/20 train/test data split block. [cite_start]For the Mean Average Precision (MAP@10) ranking metric, a movie is strictly classified as **highly relevant** if its observed true ground-truth rating is $\ge$ 3.5 stars[cite: 44, 45, 46].

| Performance Evaluation Metric | Optimized Latent SVD | Item-Based Neighborhood CF | Performance Verdict |
| :--- | :---: | :---: | :---: |
| **RMSE (Prediction Accuracy) ↓** | **0.9686** | 1.1905 | [cite_start]**SVD Wins (+18.6% Error Reduction)** [cite: 46] |
| **MAE (Absolute Prediction Gap) ↓** | **0.8194** | 0.9133 | [cite_start]**SVD Wins (+10.2% Absolute Gain)** [cite: 46] |
| **MAP@10 (List Ranking Quality) ↑** | **0.5337** | 0.5165 | [cite_start]**SVD Wins (+3.3% Top-K Relevance)** [cite: 46] |

### Core Project Insights
1. [cite_start]**SVD handles matrix sparsity much more effectively** than neighborhood techniques. Neighborhood models degrade rapidly under extreme sparsity because overlapping co-rating columns are rare, whereas SVD finds smooth abstract patterns in hidden taste dimensions.
2. [cite_start]**Data scale is critical** for optimization. [cite_start]Increasing SVD tracking row counts from 100k to 1M directly dropped validation RMSE bounds from 1.0155 to 0.9686.

---

## 🛠️ Local Environment Reproduction Steps

Follow these basic commands to configure a local environment workspace to view and execute the pipeline notebook.

### 1. Clone the Repository Workspace
```bash
git clone [https://github.com/Priyanka-31-b/netflix-recsys.git](https://github.com/Priyanka-31-b/netflix-recsys.git)
cd netflix-recsys
