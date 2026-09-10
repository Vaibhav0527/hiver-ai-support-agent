import sys
import os

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import DatasetLoader
from src.data.cleaner import DataCleaner
from src.data.conversation import ConversationExtractor

def inspect():
    loader = DatasetLoader()
    raw_df = loader.load_data()
    
    print("\n--- DATASET INSPECTION ---")
    print(f"Number of rows: {len(raw_df)}")
    print(f"Columns: {list(raw_df.columns)}")
    
    print("\nMissing values:")
    print(raw_df.isnull().sum())
    
    # Clean the data
    df = DataCleaner.clean(raw_df)
    
    # Brands vs Customers
    df['is_brand'] = ~df['inbound']
    num_brands = df[df['is_brand']]['author_id'].nunique()
    print(f"\nNumber of unique brands: {num_brands}")
    
    # Extract threads
    threads = ConversationExtractor.extract_threads(df)
    print(f"Number of conversations/threads found: {len(threads)}")
    
    if threads:
        print("\n--- EXAMPLE CONVERSATION ---")
        example = threads[0]
        for tweet in example:
            sender = "BRAND" if tweet.get('is_brand') else "CUSTOMER"
            print(f"[{sender} - {tweet['author_id']}]: {tweet['text']}")

if __name__ == "__main__":
    inspect()
