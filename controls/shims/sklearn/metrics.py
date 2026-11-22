"""roc_auc_score for binary labels, by the rank-sum formula with average ranks for ties.
This equals scikit-learn's value, including under ties."""
import numpy as np
from scipy.stats import rankdata


def roc_auc_score(y_true, y_score):
    y = np.asarray(y_true).astype(bool)
    r = rankdata(np.asarray(y_score, dtype=np.float64))
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("only one class present")
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
