"""
src/retrieval/retriever.py

Retrieves top-k similar conversations from the pre-built index.

Each result is a RetrievedExample dataclass containing:
  - customer_message
  - conversation_context
  - historical_brand_response
  - intent
  - conversation_outcome
  - similarity_score      (cosine, 0.0-1.0)
  - conversation_id

Retrieval uses cosine similarity between the query TF-IDF vector and
the indexed document vectors.

Optional intent-filtering: if intent is supplied, only documents whose
recorded intent matches are considered before ranking by similarity.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.index import ConversationIndex


@dataclass
class RetrievedExample:
    conversation_id: str
    customer_message: str
    conversation_context: list
    historical_brand_response: str
    intent: Optional[str]
    conversation_outcome: str
    similarity_score: float


class ConversationRetriever:
    """
    Retrieves top-k similar historical conversations for a query message.
    """

    def __init__(self, index: ConversationIndex, default_k: int = 3):
        self.index = index
        self.default_k = default_k

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        intent_filter: Optional[str] = None,
    ) -> list[RetrievedExample]:
        """
        Retrieve the top-k most similar training conversations for a query.

        Args:
            query:          The customer's message text.
            k:              Number of results to return. Defaults to self.default_k.
            intent_filter:  If given, restrict candidates to this intent label.

        Returns:
            List of RetrievedExample sorted by descending similarity score.
        """
        k = k or self.default_k

        if not query or not query.strip():
            return []

        # Vectorise the query using the fitted vectoriser
        q_vec = self.index.vectorizer.transform([query.strip()])

        # If intent filtering is requested, build a mask
        docs = self.index.documents
        if intent_filter:
            mask = [i for i, d in enumerate(docs) if d.get("intent") == intent_filter]
            if not mask:
                # Fall back to searching everything
                mask = list(range(len(docs)))
        else:
            mask = list(range(len(docs)))

        # Compute cosine similarities against masked subset
        candidate_vectors = self.index.vectors[mask]
        scores = cosine_similarity(q_vec, candidate_vectors).flatten()

        # Pick top-k indices within the candidate subset
        top_local = np.argsort(scores)[::-1][:k]

        results = []
        for local_idx in top_local:
            global_idx = mask[local_idx]
            doc  = docs[global_idx]
            score = float(scores[local_idx])
            results.append(
                RetrievedExample(
                    conversation_id=doc["conversation_id"],
                    customer_message=doc["customer_message"],
                    conversation_context=doc["conversation_context"],
                    historical_brand_response=doc["historical_brand_response"],
                    intent=doc.get("intent"),
                    conversation_outcome=doc.get("conversation_outcome", "unknown"),
                    similarity_score=round(score, 4),
                )
            )

        return results
