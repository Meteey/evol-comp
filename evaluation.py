
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_indices: np.ndarray,
    classifier_factory: Callable,
    label: str,
) -> Dict:

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    selected_indices = np.asarray(selected_indices)

    X_tr = X_train[:, selected_indices]
    X_te = X_test[:, selected_indices]

    clf = classifier_factory()
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_te)[:, 1]
    else:
        y_score = clf.decision_function(X_te)

    return {
        "method": label,
        "selected_feature_count": int(selected_indices.size),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
