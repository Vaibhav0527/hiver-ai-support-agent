import pandas as pd
import os

class DatasetLoader:
    def __init__(self, data_path: str = None):
        if data_path is None:
            # Check for raw dataset, fallback to sample
            if os.path.exists("data/raw/twcs.csv"):
                self.data_path = "data/raw/twcs.csv"
            elif os.path.exists("data/samples/sample_twcs.csv"):
                self.data_path = "data/samples/sample_twcs.csv"
            else:
                raise FileNotFoundError("Could not find twcs.csv or sample_twcs.csv")
        else:
            self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Loads the dataset from the CSV file into a Pandas DataFrame."""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path, dtype={'tweet_id': str, 'in_response_to_tweet_id': str, 'response_tweet_id': str})
        return df
