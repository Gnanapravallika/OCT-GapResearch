# TrustOCT / OCT-GapResearch

A trustworthy-AI **evaluation framework** for Optical Coherence Tomography (OCT) retinal disease classification (**CNV** / **DME** / **DRUSEN** / **NORMAL**) on the Kermany/Mendeley OCT2017 dataset.

**TrustOCT is an evaluation framework.** Beyond classification accuracy, it requires reporting calibration (ECE, Brier score), explanation faithfulness (LayerCAM + Deletion/Insertion AOPC), acquisition noise robustness, and external generalization. `ResNetMSFCBAM` (ResNet50 + Multi-Scale Feature fusion + CBAM) serves as the reference model demonstrating the framework.

---

## ⚡ Quickstart Options for Google Colab

### Option 1: 3-Notebook Modular Workflow (Recommended for Colab GPU limits)

To avoid free Google Colab GPU disconnects and usage limits, training and evaluation are split into 3 modular notebooks connected via Google Drive:

1. **Account 1 — `TrustOCT_Account1_Baseline_EXP001.ipynb`**
   - Mounts Google Drive (`/content/drive/MyDrive/TrustOCT_Results`).
   - Downloads dataset and trains **EXP001 Baseline ResNet50**.
   - Saves model weights, predictions (`exp001_predictions.npz`), and history to Google Drive.

2. **Account 2 — `TrustOCT_Account2_Reference_EXP003.ipynb`**
   - Mounts Google Drive.
   - Downloads dataset and trains **EXP003 Reference Model (ResNet50 + MSF + CBAM)**.
   - Saves model weights (`EXP003_resnet50_msf_cbam_best.pt`), predictions (`exp003_predictions.npz`), and history to Google Drive.

3. **Account 3 — `TrustOCT_Account3_Full_Evaluation.ipynb` (NEW — Zero GPU Training Required)**
   - Mounts Google Drive and loads EXP001 & EXP003 results from Drive.
   - Runs in **just 1–2 minutes** with zero risk of timing out!
   - Generates the full **Ablation Table**, **McNemar Test & Bootstrap CIs**, **ECE & Reliability Diagrams**, **LayerCAM AOPC**, **Noise Robustness Metrics**, **Compute Cost**, **Failure Case Gallery**, **External Validation**, and saves `trustoct_results.zip` to Google Drive.

---

### Option 2: Single-Notebook Workflow

- **`TrustOCT_Kermany_Colab.ipynb`**
  - Runs the entire pipeline top-to-bottom in a single Colab GPU session.

---

## 📂 Repository Structure

```
OCT-GapResearch/
│── README.md                                 # Framework & setup documentation
│── TrustOCT_Kermany_Colab.ipynb              # Combined single-session notebook
│── TrustOCT_Account1_Baseline_EXP001.ipynb   # Dedicated notebook for Account 1 (EXP001 Training)
│── TrustOCT_Account2_Reference_EXP003.ipynb  # Dedicated notebook for Account 2 (EXP003 Training)
│── TrustOCT_Account3_Full_Evaluation.ipynb   # Dedicated notebook for Account 3 (Full Evaluation - Fast!)
└── trustoct/                                 # TrustOCT Python package
    ├── __init__.py                           # Public API exports
    ├── data.py                               # CLAHE preprocessing, Kermany dataset indexing & patient-grouped splits
    ├── modules.py                             # Architectural blocks: CBAM attention & MSF multi-scale fusion
    ├── model.py                               # Unified ResNetMSFCBAM model & factories
    ├── train.py                               # Shared training loop with dual early-stopping mechanisms
    ├── metrics.py                             # Multi-metric evaluation (Accuracy, F1, Specificity, MCC, Kappa, ROC-AUC)
    ├── calibration.py                         # ECE, Brier score, and reliability diagram plotting
    ├── explainability.py                      # LayerCAM generation & quantitative Deletion/Insertion AOPC
    ├── robustness.py                          # Perturbation evaluation (Gaussian noise/blur, brightness/contrast)
    ├── multiseed.py                           # Multi-seed ablation runner with paired statistical significance tests
    ├── external_validation.py                 # Independent dataset validation framework
    └── utils.py                               # Seeding, checkpointing, and metrics tracking utilities
```
