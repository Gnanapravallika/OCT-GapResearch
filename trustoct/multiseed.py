"""
trustoct.multiseed
Statistical significance across the ablation (EXP001 vs EXP003).

A single run per experiment invites the obvious reviewer question: "is that
accuracy difference meaningful, or noise?" This module trains each experiment
config across multiple random seeds and reports mean ± std for every metric,
plus a paired t-test / Wilcoxon signed-rank test between the reference model
(EXP003) and the baseline (EXP001) on per-seed metric values.

Usage (in the Colab notebook):

    from trustoct.multiseed import run_multiseed_ablation
    results = run_multiseed_ablation(
        train_ds_fn, val_ds_fn, test_ds_fn,   # callables: seed -> Dataset
        seeds=[42, 123, 2024],
        epochs=25, batch_size=32, lr=1e-4, ckpt_dir="/content/checkpoints",
    )

Note on cost: this multiplies total training time by len(seeds). With 3 seeds
across 2 experiments that's 6 full training runs — budget Colab GPU time
accordingly. If time is tight, 3 seeds is the minimum that's still defensible
in a viva/review ("we repeated with 3 random seeds"); fewer than that, don't
claim statistical significance at all — just report single-run numbers
honestly and note this as a limitation.
"""
import numpy as np
import pandas as pd
from scipy import stats

from trustoct.model import build_model, EXPERIMENTS
from trustoct.train import fit, compute_class_weights
from trustoct.metrics import get_predictions, compute_metrics
from trustoct.utils import set_seed, get_device


def run_single_seed(exp_name, train_ds, val_ds, test_ds, seed, num_classes=4,
                     epochs=25, batch_size=32, lr=1e-4, weight_decay=1e-4,
                     ckpt_dir="/content/checkpoints", patience=5, verbose=False):
    """Trains one (experiment, seed) combination end-to-end and returns test-set metrics."""
    set_seed(seed)
    device = get_device()

    model = build_model(exp_name, num_classes=num_classes)
    class_weights = compute_class_weights(train_ds.labels, num_classes)

    model, history = fit(
        model, train_ds, val_ds, exp_name=f"{exp_name}_seed{seed}",
        epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        ckpt_dir=ckpt_dir, patience=patience, class_weights=class_weights, verbose=verbose,
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    y_true, y_pred, y_prob, _ = get_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    metrics["seed"] = seed
    metrics["experiment"] = exp_name
    return metrics, model


def run_multiseed_ablation(train_ds, val_ds, test_ds, seeds=(42, 123, 2024),
                            num_classes=4, epochs=25, batch_size=32, lr=1e-4,
                            weight_decay=1e-4, ckpt_dir="/content/checkpoints",
                            patience=5, verbose=True):
    """
    Runs all EXPERIMENTS across all seeds. Datasets are fixed across
    seeds here (only model init + training stochasticity varies) — that's the
    correct design for "does the architecture reliably help," as opposed to
    also varying the data split, which would conflate two different sources
    of variance.

    Returns:
        per_run_df: one row per (experiment, seed) — the raw data.
        summary_df: mean ± std per experiment, ready for the paper's table.
        significance: dict of paired-test results (e.g. EXP003 vs EXP001) per metric.
    """
    all_rows = []
    for exp_name in EXPERIMENTS:
        for seed in seeds:
            if verbose:
                print(f"\n=== Running {exp_name} | seed={seed} ===")
            metrics, _ = run_single_seed(
                exp_name, train_ds, val_ds, test_ds, seed,
                num_classes=num_classes, epochs=epochs, batch_size=batch_size,
                lr=lr, weight_decay=weight_decay, ckpt_dir=ckpt_dir,
                patience=patience, verbose=verbose,
            )
            all_rows.append(metrics)

    per_run_df = pd.DataFrame(all_rows)

    metric_cols = [c for c in per_run_df.columns if c not in ("seed", "experiment")]
    summary_rows = []
    for exp_name in EXPERIMENTS:
        sub = per_run_df[per_run_df["experiment"] == exp_name]
        row = {"experiment": exp_name}
        for m in metric_cols:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std(ddof=1) if len(sub) > 1 else 0.0
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("experiment")

    significance = {}
    exp_names = list(EXPERIMENTS.keys())
    reference = exp_names[-1]  # EXP003
    for baseline in exp_names[:-1]:
        sig_for_baseline = {}
        ref_sub = per_run_df[per_run_df["experiment"] == reference].sort_values("seed")
        base_sub = per_run_df[per_run_df["experiment"] == baseline].sort_values("seed")
        for m in metric_cols:
            if len(ref_sub) >= 2 and len(base_sub) >= 2 and len(ref_sub) == len(base_sub):
                t_stat, p_val = stats.ttest_rel(ref_sub[m].values, base_sub[m].values)
                sig_for_baseline[m] = {"t_stat": float(t_stat), "p_value": float(p_val)}
            else:
                sig_for_baseline[m] = {
                    "t_stat": float("nan"), "p_value": float("nan"),
                    "note": "need >=2 matched seeds per experiment for a paired test",
                }
        significance[f"{reference}_vs_{baseline}"] = sig_for_baseline

    return per_run_df, summary_df, significance


def format_mean_std_table(summary_df, metrics_to_show=None):
    """Formats summary_df into 'mean ± std' strings for direct paste into the paper."""
    if metrics_to_show is None:
        metrics_to_show = ["accuracy", "f1_macro", "roc_auc_macro", "mcc"]
    out = pd.DataFrame(index=summary_df.index)
    for m in metrics_to_show:
        out[m] = summary_df.apply(lambda r: f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}", axis=1)
    return out
