import pandas as pd
import numpy as np
import os
import sys

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.loader import DatasetLoader
from src.data.cleaner import DataCleaner
from src.data.conversation import ConversationExtractor

def load_data():
    loader = DatasetLoader()
    df = loader.load_data()
    return DataCleaner.clean(df)

def run_eda():
    os.makedirs("reports", exist_ok=True)
    
    df = load_data()
    
    # Basic info
    total_tweets = len(df)
    
    # Unique brands: Any author_id that is NOT numeric is a brand in this dataset.
    # Actually, some might be numeric if they are customer IDs.
    df['is_brand'] = ~df['inbound']
    brands = df[df['is_brand']]['author_id'].unique()
    total_brands = len(brands)
    
    # Which brands have the most conversations?
    brand_counts = df[df['is_brand']]['author_id'].value_counts()
    brand_counts.head(20).to_csv("reports/top_brands_by_count.csv")
    
    # Multi-turn conversations
    # A multi-turn conversation happens when a tweet has an in_response_to_tweet_id 
    # that is itself in the dataset, leading to a chain.
    # To quickly count multi-turn, we can check how many tweets are both a response and get responded to
    has_response = df['response_tweet_id'].notna() & (df['response_tweet_id'] != '')
    is_response = df['in_response_to_tweet_id'].notna()
    multi_turn_tweets = df[has_response & is_response]
    num_multi_turn = len(multi_turn_tweets)
    
    # Message lengths
    df['message_length'] = df['text'].astype(str).str.len()
    msg_length_dist = df['message_length'].describe()
    msg_length_dist.to_csv("reports/message_length_distribution.csv")
    
    # Conversation Length Distribution
    threads = ConversationExtractor.extract_threads(df)
    thread_lengths = pd.Series([len(t) for t in threads])
    if not thread_lengths.empty:
        thread_lengths.value_counts().sort_index().to_csv("reports/conversation_length_distribution.csv")
    
    # Customer vs Brand messages
    customer_msgs = df['inbound'].sum()
    brand_msgs = len(df) - customer_msgs
    
    # Missing values and Basic Data Quality
    missing_vals = df.isnull().sum()
    missing_vals.to_frame(name="missing_count").to_csv("reports/data_quality_stats.csv")
    
    # Duplicates
    duplicates = df['text'].duplicated().sum()
    
    # Malformed/Noisy
    # E.g., text is empty or very short
    noisy_records = len(df[df['message_length'] < 2])
    
    # Imbalance
    top_5_percent = brand_counts.head(5).sum() / brand_msgs * 100 if brand_msgs > 0 else 0
    
    # Output markdown report
    with open("reports/eda.md", "w") as f:
        f.write("# Exploratory Data Analysis\n\n")
        f.write(f"1. **Total tweets:** {total_tweets:,}\n")
        f.write(f"2. **Unique brands:** {total_brands:,}\n")
        f.write(f"3. **Top brand:** {brand_counts.index[0] if len(brand_counts) > 0 else 'N/A'} ({brand_counts.iloc[0]:,} messages)\n")
        f.write(f"4. **Multi-turn messages (approx):** {num_multi_turn:,}\n")
        f.write(f"5. **Average message length:** {df['message_length'].mean():.1f} chars\n")
        f.write(f"6. **Message length range:** {df['message_length'].min()} to {df['message_length'].max()} chars\n")
        f.write(f"7. **Customer vs Brand:** {customer_msgs:,} customer, {brand_msgs:,} brand\n")
        f.write(f"8. **Missing values:** \n```\n{missing_vals}\n```\n")
        f.write(f"9. **Duplicate texts:** {duplicates:,}\n")
        f.write(f"10. **Malformed (length < 2):** {noisy_records:,}\n")
        f.write(f"11. **Brand imbalance:** Top 5 brands account for {top_5_percent:.1f}% of all brand messages.\n")
        
    print("EDA complete. Output written to reports/eda.md")

if __name__ == "__main__":
    run_eda()
