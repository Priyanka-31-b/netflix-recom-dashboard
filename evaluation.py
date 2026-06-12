import numpy as np
from surprise.accuracy import rmse, mae

def calculate_map_at_k(predictions, k=10, threshold=3.5):
    """Calculates Mean Average Precision with a strict relevance cut-off."""
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))

    user_precisions = []
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated score value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(1 for (_, true_r) in user_ratings[:k] if true_r >= threshold)
        if len(user_ratings[:k]) > 0:
            user_precisions.append(n_rel / len(user_ratings[:k]))
            
    return np.mean(user_precisions) if user_precisions else 0.0
