from surprise import SVD, KNNWithMeans

def get_optimized_svd():
    """Returns tuned SVD core parameters from hyperparameter sweeps."""
    return SVD(
        n_factors=100,
        n_epochs=30,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )

def get_baseline_item_cf():
    """Returns neighborhood baseline parameters using Cosine similarity."""
    sim_options = {
        'name': 'cosine',
        'user_based': False  
    }
    return KNNWithMeans(k=20, sim_options=sim_options)
