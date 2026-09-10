import os
import sys

def check_dataset():
    """
    Checks if the dataset is present, otherwise instructs the user on how to acquire it.
    """
    raw_path = "data/raw/twcs.csv"
    if os.path.exists(raw_path):
        print(f"Dataset already exists at {raw_path}")
        return

    print("--- DATASET ACQUISITION REQUIRED ---")
    print("The Kaggle 'Customer Support on Twitter' dataset (~900 MB) is required.")
    print("Official Source: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
    print("\nTo set it up:")
    print("1. Download the archive from Kaggle.")
    print("2. Extract the archive.")
    print("3. Move 'twcs.csv' into the 'data/raw/' directory of this project.")
    print("\nFor development purposes, a mock sample has been generated at 'data/samples/sample_twcs.csv'.")
    print("The pipeline will automatically fallback to the sample if the raw dataset is missing.")
    
if __name__ == "__main__":
    check_dataset()
