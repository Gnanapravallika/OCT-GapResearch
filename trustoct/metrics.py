"""
trustoct.metrics
Standard metrics table for the ablation (EXP001 vs EXP003) — this is the
single most-referenced table in the paper, so get its computation airtight and
identical across experiments.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, confusion_matrix, classification_report,
)


@torch.no_grad()
def get_predictions(model, loader, device):
    """Runs the model over a loader once, returns (y_true, y_pred, y_prob, paths)."""
    model.eval()
    all_labels, all_preds, all_probs, all_paths = [], [], [], []
    for images, labels, paths in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)
        all_paths.extend(paths)
    return (np.concatenate(all_labels), np.concatenate(all_preds),
            np.concatenate(all_probs), all_paths)


def specificity_per_class(y_true, y_pred, num_classes):
    """Specificity isn't in sklearn directly — derive it from the confusion matrix
    per class (TN / (TN+FP)), then macro-average."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specs = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs)), specs


def compute_metrics(y_true, y_pred, y_prob, num_classes=4, class_names=None):
    """Returns a flat dict of the metrics you'll put in the ablation table:
    accuracy, macro precision/recall/F1, balanced accuracy, specificity, MCC,
    Cohen's kappa, macro (one-vs-rest) ROC-AUC."""
    macro_spec, per_class_spec = specificity_per_class(y_true, y_pred, num_classes)

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")  # can happen if a class is absent from a small eval split

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity_macro": macro_spec,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc_macro": auc,
    }
    return metrics


def build_ablation_table(results_dict, num_classes=4):
    """results_dict: {exp_name: (y_true, y_pred, y_prob)} -> pandas DataFrame with
    one row per experiment, ready to paste into the paper (Table 2 in most OCT
    ablation papers)."""
    rows = []
    for exp_name, (y_true, y_pred, y_prob) in results_dict.items():
        m = compute_metrics(y_true, y_pred, y_prob, num_classes)
        m["experiment"] = exp_name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("experiment")
    col_order = ["accuracy", "precision_macro", "recall_macro", "specificity_macro",
                 "f1_macro", "balanced_accuracy", "mcc", "cohen_kappa", "roc_auc_macro"]
    return df[col_order].round(4)


def print_classwise_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
