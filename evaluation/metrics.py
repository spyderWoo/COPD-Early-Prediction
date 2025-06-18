# evaluate/metrics.py
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def compute_metrics(y_true, y_pred, y_prob):
    return {
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred)
    }
