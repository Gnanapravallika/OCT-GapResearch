"""
trustoct.model
ResNet50 (+ optional MSF, + optional CBAM) classifier for OCT.

IMPORTANT NAMING NOTE (read this before writing the paper):
TrustOCT is the FRAMEWORK — the evaluation methodology spanning metrics,
calibration, explainability faithfulness, and robustness (see trustoct/__init__.py).
The class below, `ResNetMSFCBAM`, is just ONE reference model used to
demonstrate that framework. Keeping these conceptually separate is what makes
the contribution "a framework" rather than "a CNN with a fancy name" — say so
explicitly in the paper's contribution statement.

Three experiment configs share this single class, controlled by two flags,
so the ablation (EXP001 -> EXP002 -> EXP003) is a true controlled comparison
(same backbone, same head, same training recipe):
    EXP001: use_msf=False, use_cbam=False   (plain ResNet50 baseline)
    EXP002: use_msf=True,  use_cbam=False   (+MSF)
    EXP003: use_msf=True,  use_cbam=True    (+MSF+CBAM)  <- the reference model

MSF here is used as a fusion mechanism to demonstrate the framework, not
presented as a novel architectural contribution in its own right — the paper's
novelty claim should rest on the evaluation methodology, not the backbone.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

from trustoct.modules import CBAM, MSFModule


class ResNetMSFCBAM(nn.Module):
    def __init__(self, num_classes=4, use_msf=True, use_cbam=True,
                 pretrained=True, msf_out_channels=256, cbam_reduction=16):
        super().__init__()
        self.use_msf = use_msf
        self.use_cbam = use_cbam

        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)

        # Split backbone into stages so we can tap intermediate feature maps.
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256 ch
        self.layer2 = backbone.layer2   # 512 ch
        self.layer3 = backbone.layer3   # 1024 ch
        self.layer4 = backbone.layer4   # 2048 ch

        if self.use_msf:
            self.msf = MSFModule(in_channels_list=[512, 1024, 2048], out_channels=msf_out_channels)
            head_in_channels = msf_out_channels
        else:
            head_in_channels = 2048

        if self.use_cbam:
            self.cbam = CBAM(head_in_channels, reduction_ratio=cbam_reduction)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(head_in_channels, num_classes)

        # Keep a handle to the last conv feature map for Grad-CAM/LayerCAM hooks.
        self._last_features = None

    def forward(self, x, return_features=False):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        if self.use_msf:
            feat = self.msf([c2, c3, c4])
        else:
            feat = c4

        if self.use_cbam:
            feat = self.cbam(feat)

        self._last_features = feat  # used by explainability hooks

        pooled = self.gap(feat).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if return_features:
            return logits, feat
        return logits

    def get_target_layer(self):
        """Returns the module whose output activations/gradients LayerCAM should use.
        This is the final fused+attended feature map -> most semantically meaningful
        for a class-discriminative heatmap."""
        if self.use_cbam:
            return self.cbam
        if self.use_msf:
            return self.msf
        return self.layer4


EXPERIMENTS = {
    "EXP001_baseline_resnet50": dict(use_msf=False, use_cbam=False),
    "EXP003_resnet50_msf_cbam": dict(use_msf=True, use_cbam=True),
}


def build_model(exp_name, num_classes=4, pretrained=True):
    """Generic factory, kept for programmatic/looped use (e.g. multiseed.py)."""
    assert exp_name in EXPERIMENTS, f"Unknown exp_name. Choose from {list(EXPERIMENTS.keys())}"
    cfg = EXPERIMENTS[exp_name]
    return ResNetMSFCBAM(num_classes=num_classes, pretrained=pretrained, **cfg)


# --- Explicit named factories -------------------------------------------------
# Reviewer feedback: prefer explicit constructors over scattering boolean flags
# through calling code. Use these directly in notebook/paper code listings;
# build_model() above stays available for anywhere you need to loop over
# EXPERIMENTS programmatically (e.g. the multiseed runner).

def build_resnet50(num_classes=4, pretrained=True):
    """EXP001: plain ResNet50 baseline, no MSF, no CBAM."""
    return ResNetMSFCBAM(num_classes=num_classes, use_msf=False, use_cbam=False, pretrained=pretrained)


def build_resnet50_msf_cbam(num_classes=4, pretrained=True):
    """EXP003: ResNet50 + MSF + CBAM — the TrustOCT reference model."""
    return ResNetMSFCBAM(num_classes=num_classes, use_msf=True, use_cbam=True, pretrained=pretrained)


# Backward-compatible alias — remove before final submission once all
# notebook cells / saved-script references are confirmed migrated to the new name.
TrustOCTNet = ResNetMSFCBAM
