import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.conversation_builder import ConversationBuilder

def test_conversation_builder():
    # Mock data
    df = pd.DataFrame({
        'tweet_id': ['1', '2', '3', '4', '5'],
        'author_id': ['123', 'AppleSupport', '123', 'AppleSupport', '456'],
        'in_response_to_tweet_id': ['', '1', '2', '3', ''],
        'text': ['Help my phone', 'What iOS?', 'iOS 11', 'DM us', 'Not apple related'],
        'created_at': ['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05']
    })
    
    builder = ConversationBuilder(brand_name="AppleSupport")
    conversations = builder.build_conversations(df)
    
    assert len(conversations) == 1
    
    conv = conversations[0]
    assert conv['conversation_id'] == '1'
    assert len(conv['messages']) == 4
    
    # Check chronological order and roles
    assert conv['messages'][0]['role'] == 'customer'
    assert conv['messages'][1]['role'] == 'brand'
    assert conv['messages'][2]['role'] == 'customer'
    assert conv['messages'][3]['role'] == 'brand'

def test_excludes_brand_initiated():
    df = pd.DataFrame({
        'tweet_id': ['1', '2'],
        'author_id': ['AppleSupport', '123'],
        'in_response_to_tweet_id': ['', '1'],
        'text': ['Hey we have a new product!', 'cool'],
        'created_at': ['2017-01-01', '2017-01-02']
    })
    builder = ConversationBuilder(brand_name="AppleSupport")
    assert len(builder.build_conversations(df)) == 0
