import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.loader import DatasetLoader
from src.data.cleaner import DataCleaner
from src.data.conversation_builder import ConversationBuilder

def load_config():
    with open("configs/brand.yaml", "r") as f:
        return yaml.safe_load(f)

def run():
    print("Loading configuration...")
    config = load_config()
    brand_name = config['brand_name']
    
    print(f"Loading data to build conversations for {brand_name}...")
    df = DataCleaner.clean(DatasetLoader().load_data())
    
    builder = ConversationBuilder(brand_name)
    conversations = builder.build_conversations(df)
    
    # Filter based on config
    min_length = config['data_filtering_rules'].get('min_thread_length', 2)
    filtered = [c for c in conversations if len(c['messages']) >= min_length]
    
    out_path = f"data/processed/{brand_name}_conversations.jsonl"
    builder.save_jsonl(filtered, out_path)
    
    print(f"Successfully saved {len(filtered)} conversations to {out_path}.")
    
    # Generate Statistics
    if not filtered:
        print("No conversations met the criteria.")
        return
        
    num_convs = len(filtered)
    lengths = [len(c['messages']) for c in filtered]
    avg_turns = sum(lengths) / num_convs
    lengths.sort()
    median_turns = lengths[num_convs // 2]
    max_turns = lengths[-1]
    
    total_cust = sum(1 for c in filtered for m in c['messages'] if m['role'] == 'customer')
    total_brand = sum(1 for c in filtered for m in c['messages'] if m['role'] == 'brand')
    ratio = total_cust / total_brand if total_brand > 0 else 0
    
    print("\n--- CONVERSATION STATISTICS ---")
    print(f"Number of conversations: {num_convs}")
    print(f"Average turns: {avg_turns:.2f}")
    print(f"Median turns: {median_turns}")
    print(f"Maximum turns: {max_turns}")
    print(f"Customer messages: {total_cust}")
    print(f"Brand messages: {total_brand}")
    print(f"Customer/Brand ratio: {ratio:.2f}")

if __name__ == "__main__":
    run()
