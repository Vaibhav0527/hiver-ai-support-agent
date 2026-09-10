import pandas as pd
import json

class ConversationBuilder:
    def __init__(self, brand_name: str):
        self.brand_name = brand_name

    def build_conversations(self, df: pd.DataFrame) -> list:
        # Filter for the target brand
        # A conversation is relevant if the brand is involved in it.
        # To get the full conversation, we first find all tweet IDs from/to the brand,
        # but to be safe and simple, we'll just reconstruct the whole graph and then filter.
        
        # Build dictionary for fast lookup
        tweet_dict = df.set_index('tweet_id').to_dict('index')
        
        # Build reverse lookup: which tweet responds to which
        # Because the schema has response_tweet_id as comma separated, we parse it
        # Actually in_response_to_tweet_id is a single value, making it easier to trace backwards.
        
        # Find root tweets: tweets that have no in_response_to_tweet_id
        # BUT a thread might start mid-dataset. So we find any tweet and trace it back to its root.
        
        parent_map = {}
        for tid, row in tweet_dict.items():
            parent_id = str(row.get('in_response_to_tweet_id')).strip()
            # Handle float nan strings or empty strings
            if parent_id and parent_id != 'nan' and parent_id != 'None':
                parent_map[tid] = parent_id

        def get_root(tid):
            seen = set()
            curr = tid
            while curr in parent_map and curr not in seen:
                seen.add(curr)
                if parent_map[curr] in tweet_dict:
                    curr = parent_map[curr]
                else:
                    break
            return curr

        # Group by root
        from collections import defaultdict
        threads_by_root = defaultdict(list)
        
        for tid in tweet_dict.keys():
            root = get_root(tid)
            threads_by_root[root].append(tid)
            
        conversations = []
        for root, tids in threads_by_root.items():
            # Get full tweet objects and sort chronologically
            thread_tweets = []
            for tid in tids:
                if tid in tweet_dict:
                    tweet = tweet_dict[tid]
                    tweet['tweet_id'] = tid
                    thread_tweets.append(tweet)
                    
            if not thread_tweets:
                continue
                
            # Sort by created_at
            # created_at might be string or datetime
            thread_tweets.sort(key=lambda x: str(x.get('created_at', '')))
            
            # Check if brand is involved
            is_brand_involved = any(t.get('author_id') == self.brand_name or 
                                    (isinstance(t.get('text'), str) and self.brand_name.lower() in t.get('text', '').lower()) 
                                    for t in thread_tweets)
            
            if not is_brand_involved:
                continue
                
            # Format normalized messages
            messages = []
            valid = True
            for t in thread_tweets:
                text = str(t.get('text', ''))
                if not text or len(text) < 2:
                    valid = False
                    break
                    
                role = "brand" if t.get('author_id') == self.brand_name else "customer"
                
                messages.append({
                    "role": role,
                    "text": text,
                    "timestamp": str(t.get('created_at', ''))
                })
                
            # Must have at least 1 customer and 1 brand message, and start with customer
            if not valid or len(messages) < 2:
                continue
                
            # Ensure chronological correctness for roles: 
            # Ideally starts with customer. If it starts with brand, it's a broken context fragment.
            if messages[0]['role'] != 'customer':
                continue
                
            conversations.append({
                "conversation_id": str(root),
                "messages": messages
            })
            
        return conversations

    def save_jsonl(self, conversations: list, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
