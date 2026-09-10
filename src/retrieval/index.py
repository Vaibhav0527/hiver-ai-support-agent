"""
src/retrieval/index.py

Builds and persists a TF-IDF vector index over training conversations.

Design rationale:
  - TF-IDF cosine similarity is chosen over embedding APIs because:
      * Zero latency (fully local, no network calls)
      * Completely reproducible (deterministic vectoriser)
      * Sufficient for keyword-rich tech-support queries
      * No API key required — index can be built offline
  - The index stores the full conversation metadata alongside the vectors
    so retrievals are self-contained (no second DB lookup needed).
  - Only TRAINING conversations are indexed. Golden/test conversation_ids
    are excluded at index-build time to prevent data leakage.

Index format (saved as .pkl):
  {
    "vectors":   scipy sparse matrix (n_docs, vocab),
    "vectorizer": fitted TfidfVectorizer,
    "documents": list[dict]   # full metadata per indexed doc
  }
"""

import json
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class ConversationIndex:
    """
    Manages building and persisting the TF-IDF retrieval index.
    """

    def __init__(
        self,
        max_features: int = 15_000,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            lowercase=True,
        )
        self.vectors = None       # scipy sparse matrix
        self.documents = []       # list[dict] parallel to rows in self.vectors

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(
        self,
        conversations: list,
        excluded_ids: Optional[set] = None,
    ) -> "ConversationIndex":
        """
        Fit the vectoriser and build the index from a list of conversations.

        Args:
            conversations: list of dicts with 'conversation_id' and 'messages'.
            excluded_ids:  set of conversation_ids to exclude (golden set).

        Returns:
            self (for chaining)
        """
        excluded_ids = excluded_ids or set()
        self.documents = []
        texts = []

        for conv in conversations:
            if conv["conversation_id"] in excluded_ids:
                continue

            messages = conv.get("messages", [])
            if not messages:
                continue

            # Find the first customer message (the "query anchor")
            first_customer = next(
                (m["text"] for m in messages if m["role"] == "customer"), None
            )
            if not first_customer:
                continue

            # Find the first brand reply
            first_brand = next(
                (m["text"] for m in messages if m["role"] == "brand"), None
            )

            # Build a rich text representation for indexing
            # (all customer messages concatenated so retrieval finds multi-turn context)
            all_customer_text = " ".join(
                m["text"] for m in messages if m["role"] == "customer"
            )

            doc = {
                "conversation_id": conv["conversation_id"],
                "customer_message": first_customer,
                "conversation_context": messages[:-1] if len(messages) > 1 else [],
                "historical_brand_response": first_brand or "",
                "intent": None,          # populated by caller if labels exist
                "conversation_outcome": "resolved" if first_brand else "unanswered",
                "index_text": all_customer_text,
            }
            self.documents.append(doc)
            texts.append(all_customer_text)

        if not texts:
            raise ValueError("No documents to index after applying exclusions.")

        self.vectors = self.vectorizer.fit_transform(texts)
        print(f"[Index] Built index: {len(self.documents)} docs, vocab={self.vectors.shape[1]}")
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "vectorizer": self.vectorizer,
            "vectors": self.vectors,
            "documents": self.documents,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        size_kb = os.path.getsize(path) / 1024
        print(f"[Index] Saved to {path} ({size_kb:.1f} KB)")

    @classmethod
    def load(cls, path: str) -> "ConversationIndex":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        idx = cls.__new__(cls)
        idx.vectorizer = payload["vectorizer"]
        idx.vectors    = payload["vectors"]
        idx.documents  = payload["documents"]
        print(f"[Index] Loaded {len(idx.documents)} docs from {path}")
        return idx

    def __len__(self) -> int:
        return len(self.documents)
