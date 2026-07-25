"""
trustoct.train
Training loop shared by all three experiments (EXP001/002/003), so the only
difference between runs is the model architecture — everything else (optimizer,
schedule, augmentation, loss, seed) is held fixed for a fair ablation.
"""
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trustoct.utils import AverageMeter, save_checkpoint, get_device


def compute_class_weights(labels, num_classes):
    """Inverse-frequency class weights, used in the loss to counter Kermany's
    class imbalance (NORMAL is undersampled relative to CNV/DME/DRUSEN)."""
    counts = torch.zeros(num_classes)
    for lb in labels:
        counts[lb] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
    return loss_meter.avg, acc_meter.avg


def fit(model, train_ds, val_ds, exp_name, epochs=25, batch_size=32, lr=1e-4,
        weight_decay=1e-4, num_workers=2, patience=5, ckpt_dir="/content/checkpoints",
        class_weights=None, overfit_gap_threshold=0.15, overfit_patience=3,
        min_epochs=5, verbose=True):
    """Full training run with TWO independent early-stopping triggers, plus
    best-checkpoint saving. Returns the trained model (best weights loaded) and
    a history dict (including a per-epoch overfit_gap) for the loss/accuracy
    curves you'll want in the thesis.

    Early-stop triggers (either one alone can end training):

    1. **Val-loss plateau** (`patience`): stops if val_loss hasn't improved for
       `patience` consecutive epochs. Catches the case where the model has
       simply stopped getting better.

    2. **Overfitting gap** (`overfit_gap_threshold`, `overfit_patience`): stops
       if `train_acc - val_acc` exceeds `overfit_gap_threshold` for
       `overfit_patience` consecutive epochs. This catches the case a pure
       val-loss patience check MISSES — val_loss can still be (slowly)
       improving even while the train/val gap widens, especially with a
       pretrained backbone that memorizes quickly. `min_epochs` guards against
       triggering this before the model has had a chance to warm up.

    In both cases the model reverts to the BEST checkpoint by val_loss (not the
    epoch training stopped at), so an overfitting-triggered stop still returns
    the best generalizing weights seen so far, not an already-overfit model.
    """
    device = get_device()
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    overfit_streak = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "overfit_gap": []}
    stop_reason = "completed all epochs"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        gap = tr_acc - val_acc  # positive & growing -> overfitting signature

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["overfit_gap"].append(gap)

        elapsed = time.time() - t0
        if verbose:
            flag = "  <- overfit gap high" if gap > overfit_gap_threshold else ""
            print(f"[{exp_name}] epoch {epoch:02d}/{epochs} | "
                  f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
                  f"gap {gap:+.4f} | {elapsed:.1f}s{flag}")

        # --- checkpoint on best val_loss ---
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss,
                             f"{ckpt_dir}/{exp_name}_best.pt")
        else:
            epochs_no_improve += 1

        # --- overfitting-gap trigger ---
        if epoch >= min_epochs and gap > overfit_gap_threshold:
            overfit_streak += 1
        else:
            overfit_streak = 0

        # --- check both stopping conditions ---
        if epochs_no_improve >= patience:
            stop_reason = (f"val_loss plateaued for {patience} epochs "
                            f"(best val_loss={best_val_loss:.4f} at epoch {best_epoch})")
            break
        if overfit_streak >= overfit_patience:
            stop_reason = (f"train/val accuracy gap exceeded {overfit_gap_threshold:.2f} "
                            f"for {overfit_patience} consecutive epochs "
                            f"(gap={gap:.3f} at epoch {epoch}) — reverting to best "
                            f"checkpoint from epoch {best_epoch}")
            break

    if verbose:
        print(f"[{exp_name}] Stopped: {stop_reason}")
        print(f"[{exp_name}] Restoring best weights from epoch {best_epoch} "
              f"(val_loss={best_val_loss:.4f}).")

    model.load_state_dict(best_state)
    history["stop_reason"] = stop_reason
    history["best_epoch"] = best_epoch
    return model, history
