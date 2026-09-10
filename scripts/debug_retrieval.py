"""
scripts/debug_retrieval.py

Interactive debug script for manually inspecting retrieval quality.

Shows: query → top-k retrieved examples with similarity scores.

Usage:
  python scripts/debug_retrieval.py
  python scripts/debug_retrieval.py --k 5
  python scripts/debug_retrieval.py --interactive    # type your own queries
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.index import ConversationIndex
from src.retrieval.retriever import ConversationRetriever

INDEX_PATH = "models/retrieval_index.pkl"

DEMO_QUERIES = [
    ("My iPhone battery drains super fast after the latest update",   None),
    ("Safari keeps freezing and crashing on my iPad",                 None),
    ("I forgot my Apple ID password, can't log in",                   None),
    ("My phone screen cracked after I dropped it",                    None),
    ("I was charged $9.99 on my card but didn't buy anything",        None),
    ("How do I back up my iPhone to iCloud?",                         None),
    ("My AirPods keep disconnecting from Bluetooth",                  None),
    ("iPhone was stolen, need to lock it remotely",                   None),
]

SEPARATOR = "-" * 65


def display_result(i: int, ex, query: str):
    print(f"\n  [{i}] Similarity: {ex.similarity_score:.4f}")
    print(f"       Conv ID  : {ex.conversation_id}")
    print(f"       Intent   : {ex.intent or 'N/A'}")
    print(f"       Outcome  : {ex.conversation_outcome}")
    print(f"\n       Customer : {ex.customer_message}")
    if ex.historical_brand_response:
        print(f"       Brand    : {ex.historical_brand_response}")
    if ex.conversation_context:
        print(f"       Context  : {len(ex.conversation_context)} prior message(s)")


def run_demo(retriever: ConversationRetriever, k: int):
    print("\n" + SEPARATOR)
    print("  DEMO QUERIES")
    print(SEPARATOR)

    for query, intent_filter in DEMO_QUERIES:
        print(f"\n  QUERY : {query}")
        if intent_filter:
            print(f"  FILTER: intent={intent_filter}")
        print(SEPARATOR)

        results = retriever.retrieve(query, k=k, intent_filter=intent_filter)
        if not results:
            print("  (no results)")
        else:
            for i, ex in enumerate(results, 1):
                display_result(i, ex, query)
        print()


def run_interactive(retriever: ConversationRetriever, k: int):
    print("\nInteractive mode. Type a customer message. Enter 'q' to quit.\n")
    while True:
        query = input("Customer message> ").strip()
        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:
            continue

        intent_filter = input("Filter by intent? (leave blank for none)> ").strip() or None
        results = retriever.retrieve(query, k=k, intent_filter=intent_filter)

        print(f"\n  Top {len(results)} result(s):")
        for i, ex in enumerate(results, 1):
            display_result(i, ex, query)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3, help="Number of results to retrieve")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode")
    args = parser.parse_args()

    if not os.path.exists(INDEX_PATH):
        print(f"[ERROR] Index not found at {INDEX_PATH}")
        print("Run 'python scripts/build_index.py' first.")
        sys.exit(1)

    index     = ConversationIndex.load(INDEX_PATH)
    retriever = ConversationRetriever(index, default_k=args.k)

    print(f"\n  Loaded index: {len(index)} documents")
    print(f"  Top-k       : {args.k}")

    if args.interactive:
        run_interactive(retriever, args.k)
    else:
        run_demo(retriever, args.k)


if __name__ == "__main__":
    main()
