"""
trustoct.data
Kermany/Mendeley OCT2017 dataset loading for Colab.

Dataset: "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for
Classification" (Kermany et al., 2018) — CNV / DME / DRUSEN / NORMAL, ~84,000 train
images + the original test/val split.

On Colab, the easiest reliable path is Kaggle via kagglehub:
    kaggle dataset: paultimothymooney/kermany2018
"""
import os
import re
import glob
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["NORMAL", "CNV", "DME", "DRUSEN"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def download_kermany_dataset(dest_dir="/content/data"):
    """Downloads OCT2017 dataset on Colab using kagglehub.
    Requires kaggle.json uploaded / KAGGLE credentials set, OR kagglehub's
    interactive auth. Run this once per Colab session.
    """
    import kagglehub
    path = kagglehub.dataset_download("paultimothymooney/kermany2018")
    print(f"Dataset downloaded to: {path}")
    return path


def apply_clahe(img_np: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization — standard OCT preprocessing
    step to boost layer contrast before feeding into a network pretrained on natural
    images. Operates on single-channel (grayscale) OCT B-scans."""
    if img_np.ndim == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    out = clahe.apply(img_np.astype(np.uint8))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


class CLAHETransform:
    """torchvision-compatible transform wrapping apply_clahe, operates on PIL Image."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img.convert("L"))
        out = apply_clahe(arr, self.clip_limit, self.tile_grid_size)
        return Image.fromarray(out)


def build_transforms(image_size=224, train=True, use_clahe=True):
    ops = []
    if use_clahe:
        ops.append(CLAHETransform())
    if train:
        ops += [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    else:
        ops += [transforms.Resize((image_size, image_size))]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class OCTDataset(Dataset):
    """Generic ImageFolder-style dataset for CNV/DME/DRUSEN/NORMAL, built from an
    explicit list of file paths so we control the exact train/val/test split
    (important: Kermany's own 'test' folder has only 8 images/class — too small for
    a stable test metric, so we re-split the ~84k 'train' folder ourselves)."""

    def __init__(self, filepaths, labels, transform=None):
        assert len(filepaths) == len(labels)
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


def index_kermany_folder(root_train_dir):
    """Scans root_train_dir (or searches recursively) for OCT2017 class folders
    (NORMAL, CNV, DME, DRUSEN) and returns (filepaths, labels)."""
    def scan_dir(d):
        fps, lbs = [], []
        if not os.path.isdir(d):
            return fps, lbs
        subdirs = {entry.upper(): entry for entry in os.listdir(d)}
        for cls in CLASSES:
            if cls in subdirs:
                cls_dir = os.path.join(d, subdirs[cls])
                files = sorted(glob.glob(os.path.join(cls_dir, "*.jpeg")) +
                                glob.glob(os.path.join(cls_dir, "*.jpg")) +
                                glob.glob(os.path.join(cls_dir, "*.png")) +
                                glob.glob(os.path.join(cls_dir, "*.JPEG")) +
                                glob.glob(os.path.join(cls_dir, "*.JPG")) +
                                glob.glob(os.path.join(cls_dir, "*.PNG")))
                fps += files
                lbs += [CLASS_TO_IDX[cls]] * len(files)
        return fps, lbs

    filepaths, labels = scan_dir(root_train_dir)

    # If nothing found directly, search recursively up to 3 levels deep
    if len(filepaths) == 0:
        parent_dir = os.path.dirname(root_train_dir) if root_train_dir else ""
        search_roots = [root_train_dir, parent_dir, os.path.dirname(parent_dir)]
        for root in search_roots:
            if root and os.path.exists(root):
                for dirpath, dirnames, _ in os.walk(root):
                    dirnames_upper = [d.upper() for d in dirnames]
                    if any(c in dirnames_upper for c in CLASSES):
                        fps, lbs = scan_dir(dirpath)
                        if len(fps) > 0:
                            filepaths, labels = fps, lbs
                            break
            if len(filepaths) > 0:
                break

    if len(filepaths) == 0:
        raise RuntimeError(
            f"No OCT images found in '{root_train_dir}'. "
            f"Expected subfolders for classes {CLASSES} containing .jpeg/.png images."
        )

    print(f"Total images found: {len(filepaths)}")
    return filepaths, labels


_PATIENT_ID_PATTERN = re.compile(r"^([A-Za-z]+)-(\d+)-\d+\.\w+$")


def extract_patient_id(filepath):
    """Kermany filenames encode a patient ID: e.g. 'CNV-1016042-1.jpeg' ->
    class=CNV, patient_id=1016042, image_index=1. A single patient contributes
    MULTIPLE B-scans, so splitting by image (not patient) lets the same
    patient's scans leak across train/val/test — inflating reported accuracy,
    since adjacent B-scans from one eye are highly correlated. This extracts
    the patient ID so splitting can be done at the patient level instead.

    Falls back to 'parent_dir/basename' (i.e. treats every image as its own
    'patient', namespaced by its containing folder so fallback IDs can't
    collide across classes) if the filename doesn't match the expected
    pattern — this keeps the pipeline from crashing on an unexpected naming
    scheme, but prints a one-time warning since it means leakage protection
    is NOT actually active for those files.
    """
    basename = os.path.basename(filepath)
    m = _PATIENT_ID_PATTERN.match(basename)
    if m:
        return m.group(2)  # numeric patient ID
    parent_dir = os.path.basename(os.path.dirname(filepath))
    return f"{parent_dir}/{basename}"  # unrecognized pattern -> no real grouping, but namespaced


_warned_ungrouped = False


def patient_grouped_stratified_split(filepaths, labels, val_frac=0.10, test_frac=0.10,
                                      seed=42, max_per_class=None):
    """Stratified split by class AND grouped by patient, so no patient's images
    appear in more than one of train/val/test. This is the split you should
    use for any number you intend to report or publish — a plain per-image
    split (see `stratified_split` below, kept only for quick experimentation)
    silently inflates test accuracy via patient leakage.

    Algorithm per class: group filepaths by patient ID, shuffle the patient
    groups (not the individual images), then greedily assign whole patient
    groups to test -> val -> train until each split's image-count target is
    reached. Patients (not images) are the unit being split, so the final
    image counts will approximate but not exactly hit val_frac/test_frac
    (patients have different numbers of scans).

    max_per_class: caps images per class by keeping whole patient groups
    (never splits a patient's images across the cap boundary) up to
    approximately max_per_class images.
    """
    global _warned_ungrouped
    rng = random.Random(seed)

    # class -> patient_id -> [filepaths]
    by_class_patient = {i: {} for i in range(len(CLASSES))}
    ungrouped_count = 0
    for fp, lb in zip(filepaths, labels):
        pid = extract_patient_id(fp)
        basename = os.path.basename(fp)
        if not _PATIENT_ID_PATTERN.match(basename):
            ungrouped_count += 1
        by_class_patient[lb].setdefault(pid, []).append(fp)

    if ungrouped_count > 0 and not _warned_ungrouped:
        print(f"WARNING: {ungrouped_count} filenames didn't match the expected "
              f"Kermany 'CLASS-patientID-index.ext' pattern and could not be "
              f"grouped by patient — leakage protection is NOT active for them. "
              f"Inspect a few filenames if this number is large.")
        _warned_ungrouped = True

    train_fp, train_lb = [], []
    val_fp, val_lb = [], []
    test_fp, test_lb = [], []
    patient_counts = {"train": 0, "val": 0, "test": 0}

    for cls_idx, patient_dict in by_class_patient.items():
        patient_ids = list(patient_dict.keys())
        rng.shuffle(patient_ids)

        if max_per_class is not None:
            capped_ids, running_total = [], 0
            for pid in patient_ids:
                if running_total >= max_per_class:
                    break
                capped_ids.append(pid)
                running_total += len(patient_dict[pid])
            patient_ids = capped_ids

        total_images = sum(len(patient_dict[pid]) for pid in patient_ids)
        target_val = int(total_images * val_frac)
        target_test = int(total_images * test_frac)

        val_ids, test_ids, train_ids = [], [], []
        running = 0
        for pid in patient_ids:
            n_imgs = len(patient_dict[pid])
            if running < target_test:
                test_ids.append(pid)
            elif running < target_test + target_val:
                val_ids.append(pid)
            else:
                train_ids.append(pid)
            running += n_imgs

        for pid in train_ids:
            train_fp += patient_dict[pid]; train_lb += [cls_idx] * len(patient_dict[pid])
        for pid in val_ids:
            val_fp += patient_dict[pid]; val_lb += [cls_idx] * len(patient_dict[pid])
        for pid in test_ids:
            test_fp += patient_dict[pid]; test_lb += [cls_idx] * len(patient_dict[pid])

        patient_counts["train"] += len(train_ids)
        patient_counts["val"] += len(val_ids)
        patient_counts["test"] += len(test_ids)

    print(f"Split sizes (images) -> train: {len(train_fp)}, val: {len(val_fp)}, test: {len(test_fp)}")
    print(f"Split sizes (patients) -> train: {patient_counts['train']}, "
          f"val: {patient_counts['val']}, test: {patient_counts['test']}")

    assert_no_patient_leakage(train_fp, val_fp, test_fp)

    return (train_fp, train_lb), (val_fp, val_lb), (test_fp, test_lb)


def assert_no_patient_leakage(train_fp, val_fp, test_fp):
    """Hard check: raises if any patient ID appears in more than one split.
    Run this after ANY split you intend to report numbers from — it's cheap
    and it's exactly the check a careful reviewer will ask whether you did."""
    train_ids = {extract_patient_id(fp) for fp in train_fp}
    val_ids = {extract_patient_id(fp) for fp in val_fp}
    test_ids = {extract_patient_id(fp) for fp in test_fp}

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Patient leakage detected! "
            f"train/val overlap: {len(overlap_train_val)} patients, "
            f"train/test overlap: {len(overlap_train_test)} patients, "
            f"val/test overlap: {len(overlap_val_test)} patients."
        )
    print("Patient-leakage check passed: no patient appears in more than one split.")


def stratified_split(filepaths, labels, val_frac=0.10, test_frac=0.10, seed=42,
                      max_per_class=None):
    """DEPRECATED for reporting numbers — splits by IMAGE, not patient, so the
    same patient's B-scans can land in both train and test (leakage inflates
    test accuracy). Kept only for fast, throwaway smoke-testing of the
    pipeline itself. Use `patient_grouped_stratified_split` for anything you
    intend to put in a table or a paper.
    """
    rng = random.Random(seed)
    by_class = {i: [] for i in range(len(CLASSES))}
    for fp, lb in zip(filepaths, labels):
        by_class[lb].append(fp)

    train_fp, train_lb = [], []
    val_fp, val_lb = [], []
    test_fp, test_lb = [], []

    for cls_idx, files in by_class.items():
        rng.shuffle(files)
        if max_per_class is not None:
            files = files[:max_per_class]
        n = len(files)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        val_files = files[:n_val]
        test_files = files[n_val:n_val + n_test]
        train_files = files[n_val + n_test:]

        train_fp += train_files; train_lb += [cls_idx] * len(train_files)
        val_fp += val_files;     val_lb += [cls_idx] * len(val_files)
        test_fp += test_files;   test_lb += [cls_idx] * len(test_files)

    print(f"Split sizes -> train: {len(train_fp)}, val: {len(val_fp)}, test: {len(test_fp)}")
    print("NOTE: this is an image-level split (not patient-grouped) — do not "
          "report numbers from this split in a paper/thesis. Use "
          "patient_grouped_stratified_split instead.")
    return (train_fp, train_lb), (val_fp, val_lb), (test_fp, test_lb)
