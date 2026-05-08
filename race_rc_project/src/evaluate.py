"""evaluate.py — Metric computation utilities."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, r2_score


def compute_metrics(y_true, y_pred, y_proba=None, n_options=4):
    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'macro_f1':  f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        n = len(y_true) // n_options
        correct = sum(
            np.argmax(y_proba[i*n_options:(i+1)*n_options]) == np.argmax(y_true[i*n_options:(i+1)*n_options])
            for i in range(n)
            if y_true[i*n_options:(i+1)*n_options].sum() > 0
        )
        metrics['exact_match'] = correct / max(n, 1)
    return metrics
