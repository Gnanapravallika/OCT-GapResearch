"""
trustoct.modules
Architectural building blocks: CBAM (Convolutional Block Attention Module,
Woo et al. 2018) and MSF (Multi-Scale Feature fusion module).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CBAM
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Squeezes spatial dims via avg+max pool, learns a shared MLP over both,
    sums, sigmoids -> per-channel weight. Tells the network 'which feature maps
    matter', e.g. layer-thickness channels vs texture channels."""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        hidden = max(in_channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        attn = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """Squeezes channel dim via avg+max pool, convolves the 2-channel map with a
    7x7 kernel, sigmoids -> per-pixel weight. Tells the network 'where to look',
    e.g. the retinal layer boundary region rather than background."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(concat))
        return x * attn


class CBAM(nn.Module):
    """Sequential channel-then-spatial attention, as in the original paper.
    Drop-in module: output has identical shape to input."""

    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ---------------------------------------------------------------------------
# MSF - Multi-Scale Feature fusion
# ---------------------------------------------------------------------------
class MSFModule(nn.Module):
    """Fuses feature maps from multiple ResNet stages (e.g. layer2, layer3, layer4)
    at a common spatial resolution and channel width, via 1x1 projection + upsample
    + concatenation + 3x3 fusion conv. Motivation for OCT specifically: pathology
    (fluid pockets in DME, neovascular membranes in CNV, drusen deposits) appears at
    very different physical scales in a B-scan, so a single-resolution feature map
    from only the last ResNet stage can miss small/early-stage lesions.
    """

    def __init__(self, in_channels_list, out_channels=256):
        """in_channels_list: channel counts of the feature maps to fuse, ordered from
        shallow -> deep, e.g. [512, 1024, 2048] for ResNet50 layer2/3/4."""
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for c in in_channels_list
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_maps):
        """feature_maps: list of tensors [B, C_i, H_i, W_i], shallow->deep.
        All are projected to out_channels and upsampled to the shallowest map's
        spatial size before fusion."""
        target_size = feature_maps[0].shape[-2:]
        projected = []
        for feat, proj in zip(feature_maps, self.projections):
            p = proj(feat)
            if p.shape[-2:] != target_size:
                p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            projected.append(p)
        fused = torch.cat(projected, dim=1)
        return self.fuse_conv(fused)
