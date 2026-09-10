import pandas as pd

class ConversationExtractor:
    @staticmethod
    def extract_threads(df: pd.DataFrame, brand_filter: str = None) -> list:
        """
        Extracts multi-turn conversational threads from the dataset.
        Returns a list of conversations, where each conversation is a list of tweets.
        """
        if brand_filter:
            # We filter for tweets that involve the brand
            df = df[(df['author_id'] == brand_filter) | (df['text'].str.contains(brand_filter, case=False, na=False))]
            
        # Simplified thread extraction logic for the sample
        # In a real scenario, we'd use a graph or recursive lookup on in_response_to_tweet_id
        
        # Build a dictionary for fast lookup
        tweet_dict = df.set_index('tweet_id').to_dict('index')
        
        threads = []
        visited = set()
        
        for tweet_id, row in tweet_dict.items():
            if tweet_id in visited:
                continue
                
            # Try to build a thread starting from this tweet
            thread = [{'tweet_id': tweet_id, **row}]
            visited.add(tweet_id)
            
            # Look forward
            current_response = row.get('response_tweet_id')
            while current_response and current_response in tweet_dict and current_response not in visited:
                next_tweet = tweet_dict[current_response]
                thread.append({'tweet_id': current_response, **next_tweet})
                visited.add(current_response)
                current_response = next_tweet.get('response_tweet_id')
                
            if len(thread) > 1:
                threads.append(thread)
                
        return threads
