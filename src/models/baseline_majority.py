import collections

class MajorityBaselineClassifier:
    """
    A simple baseline classifier that always predicts the majority class
    observed during training.
    """
    def __init__(self):
        self.majority_class = None

    def fit(self, y_train):
        """
        Calculates the majority class from the training labels.
        """
        if not y_train:
            raise ValueError("Training labels cannot be empty")
        counter = collections.Counter(y_train)
        self.majority_class = counter.most_common(1)[0][0]
        return self

    def predict(self, X_test):
        """
        Predicts the majority class for all samples in X_test.
        """
        if self.majority_class is None:
            raise ValueError("Classifier must be fitted before calling predict.")
        return [self.majority_class] * len(X_test)
