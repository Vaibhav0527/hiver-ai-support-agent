"""
tfidf_classifier.py

TF-IDF + Logistic Regression intent classifier.

Design rationale:
  - TF-IDF captures keyword/phrase importance without needing embeddings.
  - Logistic Regression is fast, interpretable, and robust on small datasets.
  - Together they form the canonical simple-NLP baseline that should beat
    majority voting while still being orders of magnitude faster than any LLM.
"""

import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TFIDFIntentClassifier:
    """
    A scikit-learn Pipeline wrapping TF-IDF vectorisation and
    Logistic Regression. Follows a fit/predict interface to match
    our other classifiers.
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        max_iter: int = 1_000,
        random_state: int = 42,
    ):
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        strip_accents="unicode",
                        lowercase=True,
                        sublinear_tf=True,  # log(1+tf) dampens high-freq terms
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        random_state=random_state,
                        class_weight="balanced",  # compensates for class imbalance
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self.classes_ = None

    def fit(self, X_train: list, y_train: list) -> "TFIDFIntentClassifier":
        self.pipeline.fit(X_train, y_train)
        self.classes_ = list(self.pipeline.classes_)
        return self

    def predict(self, X: list) -> list:
        return list(self.pipeline.predict(X))

    def predict_proba(self, X: list) -> list:
        return self.pipeline.predict_proba(X).tolist()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str) -> "TFIDFIntentClassifier":
        with open(path, "rb") as fh:
            return pickle.load(fh)
