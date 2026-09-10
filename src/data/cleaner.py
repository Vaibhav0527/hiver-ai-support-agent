import pandas as pd

class DataCleaner:
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the raw twitter dataset by handling missing values
        and standardizing column formats.
        """
        # Convert created_at to datetime
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            
        # Ensure text is string
        df['text'] = df['text'].astype(str)
        
        # Fill missing response references with empty string
        for col in ['in_response_to_tweet_id', 'response_tweet_id']:
            if col in df.columns:
                df[col] = df[col].fillna('')
                
        return df
