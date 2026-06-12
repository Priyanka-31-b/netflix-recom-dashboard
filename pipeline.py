import pandas as pd
from surprise import Dataset, Reader

def load_and_clean_data(file_path):
    """Loads dataset and prepares the data parsing schema."""
    df = pd.read_csv(file_path)
    
    df.columns = ['user_id', 'movie_id', 'rating']
    

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    return dataset, df
