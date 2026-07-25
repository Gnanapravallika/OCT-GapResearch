"""
trustoct.calibration
Phase 3 core differentiator, part A: is the model's confidence trustworthy?
Most OCT papers report accuracy only; a model can be accurate yet badly
overconfident, which matters clinically (a 99%-confident wrong prediction is more
dangerous than a 55%-confident wrong one). We quantify this with:
  - Expected Calibration Error (ECE)
  - Brier score (multiclass)
  - Reliability diagrams (plotted, for EXP001 vs EXP003)
"""
import numpy as np
import matplotlib.pyplot as plt


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """ECE = sum over bins of (|bin|/N) * |accuracy(bin) - confidence(bin)|,
    using max predicted probability (top-1 confidence) per sample, standard
    definition from Guo et al. 2017 ('On Calibration of Modern Neural Networks')."""
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else \
                 (confidences >= lo) & (confidences <= hi)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_stats.append((lo, hi, acc_in_bin, conf_in_bin, prop_in_bin))
        else:
            bin_stats.append((lo, hi, np.nan, np.nan, 0.0))
    return float(ece), bin_stats


def brier_score_multiclass(y_true, y_prob, num_classes):
    """Multiclass Brier score: mean squared error between predicted probability
    vector and one-hot true label, averaged over samples. Lower is better; a
    perfectly calibrated + accurate model scores 0."""
    y_onehot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def plot_reliability_diagram(bin_stats_dict, title="Reliability Diagram", save_path=None):
    """bin_stats_dict: {exp_name: bin_stats} where bin_stats comes from
    expected_calibration_error(). Plots one line per experiment against the
    perfect-calibration diagonal — this is the figure that visually backs up
    your ECE numbers in the paper."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    for exp_name, bin_stats in bin_stats_dict.items():
        confs = [b[3] for b in bin_stats if not np.isnan(b[3])]
        accs = [b[2] for b in bin_stats if not np.isnan(b[2])]
        ax.plot(confs, accs, marker="o", label=exp_name)

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def calibration_report(y_true, y_prob, num_classes, n_bins=15):
    """
    Beyond ECE/Brier, also reports average confidence and average correctness
    (accuracy) as plain scalars — simple numbers, but they let a reader
    interpret the DIRECTION of miscalibration at a glance: if avg_confidence >
    avg_accuracy the model is overconfident (the more clinically concerning
    direction); if avg_confidence < avg_accuracy it's underconfident.
    """
    ece, bin_stats = expected_calibration_error(y_true, y_prob, n_bins)
    brier = brier_score_multiclass(y_true, y_prob, num_classes)

    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    avg_confidence = float(confidences.mean())
    avg_accuracy = float(accuracies.mean())

    return {
        "ece": ece,
        "brier_score": brier,
        "avg_confidence": avg_confidence,
        "avg_accuracy": avg_accuracy,
        "confidence_minus_accuracy": avg_confidence - avg_accuracy,  # >0 = overconfident
    }, bin_stats
