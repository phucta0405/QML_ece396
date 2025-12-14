import numpy as np
from sklearn import svm


def fit_svm_classifier(X: np.ndarray, Y: np.ndarray, kernel: str = 'sigmoid', C: float = 1.0, gamma: str = 'scale') -> svm.SVC:
    """Train a SVM classifier on the given dataset.

    Args:
        X: Input features, shape (N_samples, N_features)
        Y: Target labels, shape (N_samples,)
        kernel: SVM kernel type ('linear', 'rbf', etc.)
        C: Regularization parameter
        gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.

    Returns:
        Trained sklearn SVC model.
    """
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(X, Y)
    return clf


def predict_svm_classifier(clf: svm.SVC, X: np.ndarray) -> np.ndarray:
    """Predict class labels using a trained SVM classifier.

    Args:
        clf: Trained sklearn SVC model.
        X: Input features, shape (N_samples, N_features)

    Returns:
        Predicted class labels, shape (N_samples,)
    """
    return clf.predict(X)
