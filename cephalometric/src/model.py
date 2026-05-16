"""
HRNet-W32 backbone (timm) with a lightweight 29-channel heatmap head.
Optionally adds a CVM stage classification head for multi-task learning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

N_LANDMARKS = 29
N_CVM_CLASSES = 6


class HeatmapHead(nn.Module):
    """
    Head that maps HRNet's stride-4 features to N heatmaps at the
    requested output stride. If output stride < 4, features are
    upsampled (bilinear) before the final conv.
    """

    def __init__(
        self,
        in_channels: int,
        n_landmarks: int = N_LANDMARKS,
        upsample_factor: int = 1,
    ):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_landmarks, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_factor > 1:
            x = F.interpolate(
                x, scale_factor=self.upsample_factor, mode="bilinear", align_corners=False
            )
        return self.conv(x)


class LightFPN(nn.Module):
    """
    Lightweight top-down FPN: merges all backbone stages to the highest-resolution
    feature map via 1×1 lateral projections + bilinear upsampling.
    Input: list of feature tensors ordered high-res → low-res.
    Output: single merged tensor at the highest resolution.
    """

    def __init__(self, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, bias=False)
            for c in in_channels_list
        ])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        projected = [lat(f) for lat, f in zip(self.laterals, features)]
        out = projected[-1]
        for p in reversed(projected[:-1]):
            out = F.interpolate(out, size=p.shape[-2:], mode="bilinear", align_corners=False)
            out = out + p
        return out


class CVMHead(nn.Module):
    """Auxiliary CVM stage classification head (6 classes)."""

    def __init__(self, in_channels: int, n_classes: int = N_CVM_CLASSES):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


class CephalometricModel(nn.Module):
    """
    HRNet-W32 + heatmap head. Optionally includes CVM classification head.

    Args:
        pretrained:    Load ImageNet pretrained weights via timm.
        use_cvm_head:  Include auxiliary CVM stage classification head.
        heatmap_stride: Stride of the output heatmap relative to input.
                        HRNet's highest-res output is stride-4 by default.
    """

    def __init__(
        self,
        pretrained: bool = True,
        use_cvm_head: bool = False,
        heatmap_stride: int = 4,
    ):
        super().__init__()
        self.heatmap_stride = heatmap_stride
        self.use_cvm_head = use_cvm_head

        if not _TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm")

        # HRNet-W32 in timm returns a list of feature maps from multiple branches.
        # We use features_only=True to get the multi-scale outputs.
        self.backbone = timm.create_model(
            "hrnet_w32",
            pretrained=pretrained,
            features_only=True,
        )

        # Probe actual output sizes — more reliable than feat_info across timm versions.
        # Skip stride-2 stem features; the HRNet high-res branch starts at stride 4.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            feats = self.backbone(dummy)

        feat_strides = [256 // f.shape[-1] for f in feats]
        valid_strides = [(s, i) for i, s in enumerate(feat_strides) if s >= 4]
        self._hi_res_idx = min(valid_strides, key=lambda x: x[0])[1]
        self._backbone_stride = feat_strides[self._hi_res_idx]
        high_res_channels = feats[self._hi_res_idx].shape[1]
        self._deep_idx = feat_strides.index(max(feat_strides))

        if heatmap_stride > self._backbone_stride:
            raise ValueError(
                f"heatmap_stride={heatmap_stride} > backbone stride={self._backbone_stride}; "
                "cannot produce heatmap at requested stride."
            )
        if self._backbone_stride % heatmap_stride != 0:
            raise ValueError(
                f"heatmap_stride={heatmap_stride} must divide backbone stride={self._backbone_stride}."
            )
        upsample_factor = self._backbone_stride // heatmap_stride

        # FPN: merge all stages (hi-res → deepest) to the hi-res resolution.
        # This gives the heatmap head context from deep layers, which is critical
        # for geometrically difficult landmarks (Go, Po, Co, Ar, R).
        self._fpn_indices = list(range(self._hi_res_idx, self._deep_idx + 1))
        fpn_in_channels = [feats[i].shape[1] for i in self._fpn_indices]
        self.fpn = LightFPN(fpn_in_channels, out_channels=high_res_channels)

        self.heatmap_head = HeatmapHead(
            high_res_channels, N_LANDMARKS, upsample_factor=upsample_factor
        )

        if use_cvm_head:
            deep_channels = feats[self._deep_idx].shape[1]
            self.cvm_head = CVMHead(deep_channels, N_CVM_CLASSES)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)

        fpn_feats = [features[i] for i in self._fpn_indices]
        merged = self.fpn(fpn_feats)
        heatmaps = self.heatmap_head(merged)

        out = {"heatmaps": heatmaps}

        if self.use_cvm_head:
            out["cvm_logits"] = self.cvm_head(features[self._deep_idx])

        return out


# ---------------------------------------------------------------------------
# Soft-argmax coordinate extraction
# ---------------------------------------------------------------------------

def soft_argmax(heatmaps: torch.Tensor, temperature: float = 100.0) -> torch.Tensor:
    """
    Extract (x, y) coordinates from heatmaps via soft-argmax.

    Args:
        heatmaps: (B, N, H, W) float tensor
        temperature: softmax sharpness — higher = closer to argmax
    Returns:
        coords: (B, N, 2) float tensor in heatmap pixel space (x, y)
    """
    B, N, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, N, -1)
    weights = torch.softmax(flat * temperature, dim=-1)

    xs = torch.arange(W, dtype=torch.float32, device=heatmaps.device)
    ys = torch.arange(H, dtype=torch.float32, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(1, 1, -1)
    grid_y = grid_y.reshape(1, 1, -1)

    pred_x = (weights * grid_x).sum(-1)  # (B, N)
    pred_y = (weights * grid_y).sum(-1)  # (B, N)

    return torch.stack([pred_x, pred_y], dim=-1)  # (B, N, 2)


def heatmap_coords_to_input_space(
    coords_hm: torch.Tensor, stride: int
) -> torch.Tensor:
    """Scale heatmap-space coordinates back to input image space."""
    return coords_hm * stride


# ---------------------------------------------------------------------------
# EMA wrapper
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_to(self, model: nn.Module):
        """Copy EMA weights into model for evaluation."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]):
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def backup_weights(self, model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
