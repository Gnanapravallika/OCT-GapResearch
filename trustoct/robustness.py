"""
trustoct.robustness
Phase 4 (supporting, keep lean): does accuracy hold up under realistic OCT
acquisition noise? 4 perturbation types x a few severities, one summary table.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def add_gaussian_noise(img_tensor, severity):
    sigma = [0.02, 0.05, 0.08, 0.12][severity - 1]
    noise = torch.randn_like(img_tensor) * sigma
    return torch.clamp(img_tensor + noise, -3, 3)  # normalized-space clamp


def add_gaussian_blur(img_tensor, severity):
    ksize = [3, 5, 7, 9][severity - 1]
    sigma = ksize / 6.0
    channels = img_tensor.shape[1]
    coords = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(img_tensor.device)
    kernel_1d = g.view(1, 1, 1, ksize)
    kernel_1d_t = g.view(1, 1, ksize, 1)
    kernel_1d = kernel_1d.repeat(channels, 1, 1, 1)
    kernel_1d_t = kernel_1d_t.repeat(channels, 1, 1, 1)
    pad = ksize // 2
    x = F.conv2d(img_tensor, kernel_1d, padding=(0, pad), groups=channels)
    x = F.conv2d(x, kernel_1d_t, padding=(pad, 0), groups=channels)
    return x


def adjust_brightness(img_tensor, severity, mean, std):
    """Adjust brightness in *unnormalized* pixel space then re-normalize, so the
    perturbation magnitude is physically meaningful (fraction of pixel range)."""
    delta = [0.06, 0.12, 0.20, 0.30][severity - 1]
    mean_t = torch.tensor(mean, device=img_tensor.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=img_tensor.device).view(1, -1, 1, 1)
    pixel = img_tensor * std_t + mean_t
    pixel = torch.clamp(pixel + delta, 0, 1)
    return (pixel - mean_t) / std_t


def adjust_contrast(img_tensor, severity, mean, std):
    factor = [0.9, 0.75, 0.6, 0.45][severity - 1]
    mean_t = torch.tensor(mean, device=img_tensor.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=img_tensor.device).view(1, -1, 1, 1)
    pixel = img_tensor * std_t + mean_t
    gray_mean = pixel.mean(dim=[2, 3], keepdim=True)
    pixel = torch.clamp((pixel - gray_mean) * factor + gray_mean, 0, 1)
    return (pixel - mean_t) / std_t


PERTURBATIONS = {
    "gaussian_noise": add_gaussian_noise,
    "gaussian_blur": add_gaussian_blur,
}
NORM_SPACE_PERTURBATIONS = {
    "brightness": adjust_brightness,
    "contrast": adjust_contrast,
}


@torch.no_grad()
def evaluate_under_perturbation(model, loader, device, mean, std, severities=(1, 2, 3)):
    """Runs the test set through each perturbation type/severity and records
    accuracy + macro-F1 drop relative to clean performance. One row per
    (perturbation, severity) in the returned list -> becomes your Phase-4 table."""
    model.eval()
    results = []

    def run_pass(perturb_fn, name, needs_norm_stats):
        for sev in severities:
            all_labels, all_preds = [], []
            for images, labels, _ in loader:
                images = images.to(device)
                if needs_norm_stats:
                    images_p = perturb_fn(images, sev, mean, std)
                else:
                    images_p = perturb_fn(images, sev)
                logits = model(images_p)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.numpy())
            if len(all_labels) == 0:
                print(f"Warning: loader yielded 0 samples for perturbation '{name}' severity {sev}.")
                continue
            y_true = np.concatenate(all_labels)
            y_pred = np.concatenate(all_preds)
            results.append({
                "perturbation": name,
                "severity": sev,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            })

    for name, fn in PERTURBATIONS.items():
        run_pass(fn, name, needs_norm_stats=False)
    for name, fn in NORM_SPACE_PERTURBATIONS.items():
        run_pass(fn, name, needs_norm_stats=True)

    return results
