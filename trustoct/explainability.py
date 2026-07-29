"""
trustoct.explainability
Phase 3 core differentiator, part B: are the model's explanations faithful, not
just pretty? Most OCT papers stop at a qualitative Grad-CAM picture. We add:
  - LayerCAM (Jiang et al. 2021) — finer-grained than Grad-CAM since it uses
    positive per-pixel gradient*activation rather than a single global-average
    weight per channel, which matters for small OCT lesions (e.g. early drusen).
  - Deletion/Insertion AOPC (Area Over Perturbation Curve, Samek et al. 2017) —
    a *quantitative* faithfulness score: does removing the pixels LayerCAM marks
    "important" actually hurt the prediction? This is the number that turns
    "look, a heatmap" into a scientific claim.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class LayerCAM:
    """Hooks a target layer's forward activations and backward gradients, then
    computes: CAM = ReLU(sum_k ReLU(grad_k) * activation_k), upsampled to input
    resolution. Call `generate(image_tensor, class_idx)` for a single image.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx=None):
        """image_tensor: [1, C, H, W], already normalized, on the correct device.
        Returns (cam [H, W] in [0,1], predicted_class_idx, predicted_prob)."""
        self.model.eval()
        image_tensor = image_tensor.clone().requires_grad_(True)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # LayerCAM weighting: positive gradients only, elementwise (not GAP-pooled
        # like Grad-CAM), which preserves finer spatial detail.
        weights = F.relu(self.gradients)
        weighted_activations = weights * self.activations
        cam = F.relu(weighted_activations.sum(dim=1, keepdim=True))  # [1,1,h,w]

        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear",
                             align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx, float(probs[0, class_idx].item())


def overlay_cam_on_image(image_np, cam, alpha=0.45, colormap="jet"):
    """image_np: [H, W, 3] float in [0,1]. cam: [H, W] float in [0,1].
    Returns an RGB overlay for the figure in your paper."""
    heatmap = cm.get_cmap(colormap)(cam)[:, :, :3]
    overlay = (1 - alpha) * image_np + alpha * heatmap
    return np.clip(overlay, 0, 1)


# ---------------------------------------------------------------------------
# Deletion / Insertion AOPC — quantitative faithfulness
# ---------------------------------------------------------------------------
@torch.no_grad()
def _predict_prob(model, image_tensor, class_idx):
    logits = model(image_tensor)
    prob = torch.softmax(logits, dim=1)[0, class_idx].item()
    return prob


def deletion_insertion_curves(model, image_tensor, cam, class_idx, device,
                               num_steps=20, baseline_value=0.0):
    """Ranks pixels by CAM importance (descending), then:
      - Deletion: progressively replaces the *most* important pixels with
        `baseline_value` (mean-normalized black) and re-scores -> a faithful CAM
        should cause a STEEP drop (low AOPC = good deletion behavior, i.e. area
        UNDER the curve is small).
      - Insertion: starts from a blank image and progressively reveals the *most*
        important pixels first -> a faithful CAM should cause a STEEP rise
        (high AOPC = good insertion behavior).
    Returns (deletion_scores, insertion_scores, del_aopc, ins_aopc) where AOPC is
    computed as area-over/under the respective curve using the trapezoidal rule.
    """
    model.eval()
    img = image_tensor.clone().to(device)  # [1, C, H, W]
    C, H, W = img.shape[1], img.shape[2], img.shape[3]

    flat_order = np.argsort(-cam.flatten())  # descending importance
    total_pixels = H * W
    step_size = max(total_pixels // num_steps, 1)

    # --- Deletion ---
    del_img = img.clone()
    deletion_scores = [_predict_prob(model, del_img, class_idx)]
    mask_flat = np.ones(total_pixels, dtype=bool)
    for step in range(1, num_steps + 1):
        idx_to_remove = flat_order[(step - 1) * step_size: step * step_size]
        mask_flat[idx_to_remove] = False
        mask_2d = torch.tensor(mask_flat.reshape(H, W), device=device, dtype=torch.float32)
        del_img = image_tensor.to(device) * mask_2d + baseline_value * (1 - mask_2d)
        deletion_scores.append(_predict_prob(model, del_img, class_idx))

    # --- Insertion ---
    ins_img_base = torch.full_like(img, baseline_value)
    insertion_scores = [_predict_prob(model, ins_img_base, class_idx)]
    mask_flat = np.zeros(total_pixels, dtype=bool)
    for step in range(1, num_steps + 1):
        idx_to_add = flat_order[(step - 1) * step_size: step * step_size]
        mask_flat[idx_to_add] = True
        mask_2d = torch.tensor(mask_flat.reshape(H, W), device=device, dtype=torch.float32)
        ins_img = image_tensor.to(device) * mask_2d + baseline_value * (1 - mask_2d)
        insertion_scores.append(_predict_prob(model, ins_img, class_idx))

    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz
    x_axis = np.linspace(0, 1, num_steps + 1)
    del_aopc = float(_trapz(deletion_scores, x_axis))   # want this LOW
    ins_aopc = float(_trapz(insertion_scores, x_axis))  # want this HIGH

    return deletion_scores, insertion_scores, del_aopc, ins_aopc


def faithfulness_report(model, cam_engine, loader, device, num_samples=50, num_steps=20):
    """Runs deletion/insertion AOPC over a random subset of the test set (full test
    set is expensive: num_samples x num_steps x 2 forward passes each). Returns
    mean deletion AOPC (lower=better) and mean insertion AOPC (higher=better) —
    the two numbers that go in your Phase-3 table next to the ECE/Brier numbers."""
    del_aopcs, ins_aopcs = [], []
    seen = 0
    for images, labels, _ in loader:
        for i in range(images.size(0)):
            if seen >= num_samples:
                break
            img = images[i:i + 1].to(device)
            cam, pred_idx, _ = cam_engine.generate(img)
            _, _, del_aopc, ins_aopc = deletion_insertion_curves(
                model, img, cam, pred_idx, device, num_steps=num_steps)
            del_aopcs.append(del_aopc)
            ins_aopcs.append(ins_aopc)
            seen += 1
        if seen >= num_samples:
            break
    return {
        "mean_deletion_aopc": float(np.mean(del_aopcs)),
        "mean_insertion_aopc": float(np.mean(ins_aopcs)),
        "n_samples": seen,
    }


def plot_cam_grid(images_np, cams, titles, save_path=None):
    """Small qualitative grid: original | heatmap | overlay, for 4-6 examples
    (your Phase-4 failure-analysis figure can reuse this too)."""
    n = len(images_np)
    if n == 0:
        print("Warning: plot_cam_grid received 0 images. Skipping plot creation.")
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No samples available", ha='center', va='center', fontsize=12)
        ax.axis("off")
        if save_path:
            fig.savefig(save_path, dpi=200)
        return fig
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(images_np[i]); axes[i, 0].set_title(f"{titles[i]} - input"); axes[i, 0].axis("off")
        axes[i, 1].imshow(cams[i], cmap="jet"); axes[i, 1].set_title("LayerCAM"); axes[i, 1].axis("off")
        overlay = overlay_cam_on_image(images_np[i], cams[i])
        axes[i, 2].imshow(overlay); axes[i, 2].set_title("Overlay"); axes[i, 2].axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig
