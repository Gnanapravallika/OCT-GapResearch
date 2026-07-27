"""
TrustOCT — a trustworthy-AI evaluation framework for OCT retinal disease
classification (CNV / DME / DRUSEN / NORMAL), built around ResNet50 + MSF + CBAM.

Not just "another model" — the contribution is the evaluation methodology:
accuracy is necessary but not sufficient, so this package also measures
calibration (ECE, Brier), explanation faithfulness (LayerCAM + Deletion/
Insertion AOPC), and robustness to acquisition-noise perturbations.
"""

from trustoct.model import (
    build_model, ResNetMSFCBAM, TrustOCTNet, EXPERIMENTS,
    build_resnet50, build_resnet50_msf_cbam,
)
from trustoct.data import (
    OCTDataset, build_transforms, index_kermany_folder, stratified_split,
    patient_grouped_stratified_split, assert_no_patient_leakage, extract_patient_id,
    download_kermany_dataset, CLASSES,
)
from trustoct.train import fit, compute_class_weights
from trustoct.metrics import get_predictions, compute_metrics, build_ablation_table
from trustoct.calibration import calibration_report, plot_reliability_diagram
from trustoct.explainability import LayerCAM, faithfulness_report, plot_cam_grid
from trustoct.robustness import evaluate_under_perturbation
from trustoct.utils import set_seed, get_device, count_parameters
from trustoct.multiseed import run_multiseed_ablation, run_single_seed, format_mean_std_table
from trustoct.external_validation import run_external_validation, index_external_folder

__all__ = [
    "build_model", "ResNetMSFCBAM", "TrustOCTNet", "EXPERIMENTS",
    "build_resnet50", "build_resnet50_msf_cbam",
    "OCTDataset", "build_transforms", "index_kermany_folder", "stratified_split",
    "patient_grouped_stratified_split", "assert_no_patient_leakage", "extract_patient_id",
    "download_kermany_dataset", "CLASSES",
    "fit", "compute_class_weights",
    "get_predictions", "compute_metrics", "build_ablation_table",
    "calibration_report", "plot_reliability_diagram",
    "LayerCAM", "faithfulness_report", "plot_cam_grid",
    "evaluate_under_perturbation",
    "set_seed", "get_device", "count_parameters",
    "run_multiseed_ablation", "run_single_seed", "format_mean_std_table",
    "run_external_validation", "index_external_folder",
]
