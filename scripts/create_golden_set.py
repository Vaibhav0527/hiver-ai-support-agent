import json
import random
import os

def load_conversations(path):
    conversations = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))
    return conversations

def simple_heuristic_classifier(text):
    text = text.lower()
    if any(w in text for w in ['battery', 'charge', 'power', 'die']):
        return "battery_power_issue"
    elif any(w in text for w in ['crack', 'drop', 'break', 'shatter', 'water']):
        return "physical_hardware_damage"
    elif any(w in text for w in ['password', 'id', 'lock', 'forgot']):
        return "account_id_access"
    elif any(w in text for w in ['charge', 'bill', 'refund', 'subscription']):
        return "billing_subscription"
    elif any(w in text for w in ['lost', 'stolen', 'find']):
        return "device_lost_stolen"
    elif any(w in text for w in ['ship', 'order', 'trade', 'store']):
        return "purchase_shipping_inquiry"
    elif any(w in text for w in ['how do i', 'where is', 'backup']):
        return "feature_how_to"
    elif any(w in text for w in ['crash', 'freeze', 'update', 'wifi', 'bluetooth']):
        return "software_bug_glitch"
    else:
        return "general_complaint"

def run():
    in_path = "data/processed/AppleSupport_conversations.jsonl"
    out_path = "data/golden/golden_set.jsonl"
    
    print(f"Loading conversations from {in_path}...")
    conversations = load_conversations(in_path)
    
    if not conversations:
        print(f"No conversations found in {in_path}. Cannot create golden set.")
        return
        
    # We want 150-250 for the final dataset, but since we only have a tiny sample,
    # we'll just take whatever is available up to 250.
    target_size = min(len(conversations), 250)
    
    # Shuffle for random sampling
    random.seed(42)
    sampled = random.sample(conversations, target_size)
    
    golden_set = []
    class_distribution = {}
    
    for i, conv in enumerate(sampled):
        # We will label the FIRST customer message as the entry point
        # The preceding context is empty, the message is the customer's query
        customer_msg = None
        for msg in conv['messages']:
            if msg['role'] == 'customer':
                customer_msg = msg['text']
                break
                
        if not customer_msg:
            continue
            
        # Provide a heuristic draft label to save human annotator time
        draft_intent = simple_heuristic_classifier(customer_msg)
        
        # Track distribution of drafts
        class_distribution[draft_intent] = class_distribution.get(draft_intent, 0) + 1
        
        golden_entry = {
            "id": f"eval_{i}",
            "conversation_id": conv['conversation_id'],
            "customer_message": customer_msg,
            "conversation_context": [],  # For the first turn, context is empty
            "intent": draft_intent, # DRAFT LABEL ONLY
            "expected_reply_characteristics": ["DRAFT: Needs Human Review"],
            "should_escalate": draft_intent in ["physical_hardware_damage", "general_complaint", "billing_subscription"],
            "escalation_reason": "DRAFT: Needs Human Review",
            "source": "heuristic_draft_pending_human_review"
        }
        golden_set.append(golden_entry)
        
    print(f"Saving {len(golden_set)} draft entries to {out_path}...")
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in golden_set:
            f.write(json.dumps(entry) + '\n')
            
    print("\n--- DRAFT CLASS DISTRIBUTION ---")
    for k, v in class_distribution.items():
        print(f"{k}: {v}")
        
    print("\nIMPORTANT: The golden set currently contains heuristic drafts.")
    print(f"A human MUST review {out_path} following configs/labeling_guidelines.md")
    print("Change 'source' to 'human_labeled' once verified.")

if __name__ == "__main__":
    run()
