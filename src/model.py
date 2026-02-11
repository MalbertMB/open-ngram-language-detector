import numpy as np

class MultinomialNB:
    """
    Custom implementation of a Multinomial Naive Bayes classifier
    with Laplace smoothing.
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Laplace smoothing parameter.
        """
        self.alpha = alpha
        self.classes_ = None
        self.log_priors_ = None      # log P(c)
        self.log_conditionals_ = None # log P(w|c)

    def fit(self, X, y):
        """
        Trains the model.
        
        Args:
            X (sparse matrix): TF-IDF or count features (n_samples, n_features).
            y (array-like): Target labels.
        """
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.log_priors_ = np.zeros(n_classes, dtype=np.float64)
        self.log_conditionals_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes_):
            # Mask for current class
            c_mask = (y == c)
            
            # Calculate Prior: P(c)
            # Count documents of class c / total documents
            count_docs_c = np.sum(c_mask)
            self.log_priors_[idx] = np.log(count_docs_c / len(y))

            # Calculate Conditionals: P(w|c) with Laplace Smoothing
            # Sum feature counts for this class
            feature_counts_c = np.array(X[c_mask, :].sum(axis=0)).flatten()
            total_count_c = feature_counts_c.sum()

            numerator = np.log(feature_counts_c + self.alpha)
            denominator = np.log(total_count_c + n_features * self.alpha)
            
            self.log_conditionals_[idx, :] = numerator - denominator

        return self

    def predict(self, X):
        """
        Predicts class labels for samples in X.
        
        Args:
            X (sparse matrix): Test features.
        
        Returns:
            np.array: Predicted labels.
        """
        # Calculate log posterior: log P(c) + sum(log P(w|c))
        # Matrix multiplication handles the sum over features
        scores = X @ self.log_conditionals_.T
        scores += self.log_priors_

        # Select class with max score
        best_indices = np.argmax(scores, axis=1)
        return self.classes_[best_indices]