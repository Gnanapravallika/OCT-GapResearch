"""
trustoct.external_validation
Tests the trained reference model's generalization on an INDEPENDENT OCT
dataset it never saw during training or model selection.

Why this matters for the paper: 96–97% accuracy on Kermany's own test split is
expected (it's a large, relatively clean, single-source dataset) and reviewers
increasingly ask "does this hold up on a different scanner/population?" without
it, "trustworthy" only covers in-distribution behavior. Even a modest external
set is stronger evidence than none.

Suggested independent datasets (pick one you can access):
  - OCTID (Optical Coherence Tomography Image Database) — public, different
    acquisition source from Kermany.
  - Duke OCT datasets (Srinivasan et al. / Farsiu Duke AMD dataset) — different
    patient population and device.
  - Any local hospital/clinic OCT scans you have ethical clearance to use.

This module makes NO assumption about label schema matching exactly — you'll
likely need a small mapping step (external_label -> {CNV, DME, DRUSEN, NORMAL})
since class taxonomies differ slightly across public OCT datasets. Do that
mapping explicitly and document it in the paper's dataset section; don't
silently drop mismatched classes without saying so.
"""
import os
import torch
from torch.utils.data import DataLoader

from trustoct.data import OCTDataset, build_transforms, CLASSES
from trustoct.metrics import get_predictions, compute_metrics, print_classwise_report
from trustoct.calibration import calibration_report
from trustoct.utils import get_device


def index_external_folder(root_dir, class_folder_map):
    """
    root_dir: path to the external dataset root, expected to contain one
              subfolder per class (folder names may differ from Kermany's).
    class_folder_map: dict mapping external folder name -> Kermany-schema
              class name, e.g. {"AMD": "DRUSEN", "Normal": "NORMAL", ...}.
              Any external class NOT in this map is skipped (and reported),
              rather than silently mis-assigned.
    """
    filepaths, labels, skipped = [], [], []
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name not in class_folder_map:
            skipped.append(folder_name)
            continue
        mapped_class = class_folder_map[folder_name]
        class_idx = CLASSES.index(mapped_class)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(folder_path, fname))
                labels.append(class_idx)

    if skipped:
        print(f"NOTE: skipped external folders with no class mapping: {skipped}. "
              f"Add them to class_folder_map if they should be included.")
    print(f"External validation set indexed: {len(filepaths)} images across "
          f"{len(set(labels))} mapped classes.")
    return filepaths, labels


def run_external_validation(model, root_dir, class_folder_map, image_size=224,
                             batch_size=32, num_classes=4, save_calibration_plot=None):
    """
    Evaluates an already-trained model (e.g. your best EXP003 checkpoint) on
    the external dataset. Returns metrics + calibration report, computed
    identically to the in-distribution test evaluation so the two are directly
    comparable in the paper (put them side by side in one table — that
    comparison IS the generalization result).
    """
    filepaths, labels = index_external_folder(root_dir, class_folder_map)
    if len(filepaths) == 0:
        raise RuntimeError("No external images matched the provided class_folder_map — check paths/mapping.")

    transform = build_transforms(image_size=image_size, train=False, use_clahe=True)
    ext_ds = OCTDataset(filepaths, labels, transform=transform)
    ext_loader = DataLoader(ext_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = get_device()
    model = model.to(device)
    y_true, y_pred, y_prob, paths = get_predictions(model, ext_loader, device)

    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    cal_report, bin_stats = calibration_report(y_true, y_prob, num_classes=num_classes)

    print("=== External Validation Results ===")
    print_classwise_report(y_true, y_pred, CLASSES)
    print(f"Accuracy: {metrics['accuracy']:.4f} | F1 (macro): {metrics['f1_macro']:.4f} | "
          f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
    print(f"ECE: {cal_report['ece']:.4f} | Brier: {cal_report['brier_score']:.4f} | "
          f"avg confidence: {cal_report['avg_confidence']:.4f} vs avg accuracy: {cal_report['avg_accuracy']:.4f}")

    if save_calibration_plot:
        from trustoct.calibration import plot_reliability_diagram
        plot_reliability_diagram({"external_validation": bin_stats},
                                  title="Reliability — External Validation Set",
                                  save_path=save_calibration_plot)

    return {
        "metrics": metrics,
        "calibration": cal_report,
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "paths": paths,
    }
