"""Evaluation metrics: Macro-F1 + accuracy for all tasks, MCC added
for acceptability — exactly as stated in the proposal."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def build_compute_metrics(report_mcc: bool):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        out = {
            "macro_f1": f1_score(labels, preds, average="macro"),
            "accuracy": accuracy_score(labels, preds),
        }
        if report_mcc:
            out["mcc"] = matthews_corrcoef(labels, preds)
        return out

    return compute_metrics
