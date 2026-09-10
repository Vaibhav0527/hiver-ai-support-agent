"""
scripts/build_index.py

Builds the TF-IDF retrieval index from training conversations.

LEAKAGE PREVENTION:
  - Golden set conversation_ids are loaded FIRST.
  - Only conversations NOT in the golden set are indexed.
  - This mirrors exactly the same split used by all other training scripts.

Output:
  models/retrieval_index.pkl
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.index import ConversationIndex

# ---------------------------------------------------------------------------
# Synthetic fallback conversations (mirrors train_tfidf.py synthetic data)
# ---------------------------------------------------------------------------
SYNTHETIC_CONVERSATIONS = [
    {
        "conversation_id": "synth_001",
        "messages": [
            {"role": "customer", "text": "My iPhone battery drains really fast after iOS update", "timestamp": ""},
            {"role": "brand",    "text": "We can help! Which iPhone model and iOS version are you on?", "timestamp": ""},
            {"role": "customer", "text": "iPhone 13, iOS 17.1", "timestamp": ""},
            {"role": "brand",    "text": "Try Settings > Battery > Battery Health. If below 80%, visit a Genius Bar.", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_002",
        "messages": [
            {"role": "customer", "text": "My phone won't charge at all, tried different cables", "timestamp": ""},
            {"role": "brand",    "text": "Sorry to hear this! Let's troubleshoot. Try a soft reset: hold Side + Volume Down.", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_003",
        "messages": [
            {"role": "customer", "text": "Safari keeps crashing every time I open a new tab on iOS 17", "timestamp": ""},
            {"role": "brand",    "text": "Let's fix that! First, try clearing Safari history: Settings > Safari > Clear History.", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_004",
        "messages": [
            {"role": "customer", "text": "WiFi disconnects randomly, happens every few minutes", "timestamp": ""},
            {"role": "brand",    "text": "Go to Settings > General > Transfer or Reset iPhone > Reset > Reset Network Settings.", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_005",
        "messages": [
            {"role": "customer", "text": "I dropped my iPhone and the screen is completely shattered", "timestamp": ""},
            {"role": "brand",    "text": "We're sorry to hear that! Please visit an Apple Store or Apple Authorized Service Provider for a screen repair.", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_006",
        "messages": [
            {"role": "customer", "text": "Forgot my Apple ID password and can't log in", "timestamp": ""},
            {"role": "brand",    "text": "Visit iforgot.apple.com to reset your Apple ID password. Need more help? DM us.", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_007",
        "messages": [
            {"role": "customer", "text": "I got charged $9.99 I don't recognize on my card", "timestamp": ""},
            {"role": "brand",    "text": "Check your purchase history at reportaproblem.apple.com. If unrecognized, contact us via DM.", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_008",
        "messages": [
            {"role": "customer", "text": "My iPhone was stolen last night, how do I lock it remotely?", "timestamp": ""},
            {"role": "brand",    "text": "Go to icloud.com/find, sign in, select your device, and choose Lost Mode to lock it immediately.", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_009",
        "messages": [
            {"role": "customer", "text": "How do I take a screenshot on the new iPhone 15?", "timestamp": ""},
            {"role": "brand",    "text": "Press the Side button and Volume Up button at the same time. The screenshot saves to Photos!", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_010",
        "messages": [
            {"role": "customer", "text": "Apple products are the worst, switching to Android", "timestamp": ""},
            {"role": "brand",    "text": "We're sorry to hear you're frustrated. Could you DM us so we can understand what went wrong?", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_011",
        "messages": [
            {"role": "customer", "text": "Battery health says 79%, phone keeps dying mid-day", "timestamp": ""},
            {"role": "brand",    "text": "At 79% health, your battery is considered worn. We recommend a battery replacement. Book via Apple Store app.", "timestamp": ""},
        ],
    },
    {
        "conversation_id": "synth_012",
        "messages": [
            {"role": "customer", "text": "Bluetooth keeps disconnecting from my AirPods every few minutes", "timestamp": ""},
            {"role": "brand",    "text": "Try unpairing and re-pairing your AirPods. Go to Settings > Bluetooth, tap the i next to AirPods > Forget.", "timestamp": ""},
        ],
    },
]


def load_jsonl(path: str) -> list:
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def run():
    os.makedirs("models", exist_ok=True)
    print("=" * 60)
    print("  Building Retrieval Index")
    print("=" * 60)

    # Load real data
    real_convs = load_jsonl("data/processed/AppleSupport_conversations.jsonl")
    golden     = load_jsonl("data/golden/golden_set.jsonl")
    golden_ids = {item["conversation_id"] for item in golden}

    print(f"  Real conversations available : {len(real_convs)}")
    print(f"  Golden set IDs to exclude    : {len(golden_ids)}")

    train_convs = [c for c in real_convs if c["conversation_id"] not in golden_ids]
    print(f"  Eligible training conversations: {len(train_convs)}")

    USE_SYNTHETIC = len(train_convs) < 3
    if USE_SYNTHETIC:
        print("\n  [INFO] Too few real conversations — using synthetic demo data.\n")
        conversations = SYNTHETIC_CONVERSATIONS
        mode = "synthetic_demo"
    else:
        conversations = train_convs
        mode = "real_data"

    # Build and save index
    index = ConversationIndex(max_features=15_000, ngram_range=(1, 2))
    index.build(conversations, excluded_ids=golden_ids)

    out_path = "models/retrieval_index.pkl"
    index.save(out_path)

    print(f"\n  Mode           : {mode}")
    print(f"  Documents indexed: {len(index)}")
    print(f"  Index path     : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()
