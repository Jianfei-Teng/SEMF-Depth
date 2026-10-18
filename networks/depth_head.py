"""Released SMF feature fusion and spatial routing.

The auxiliary prediction uses F_dec and detached F_edge. Routing combines
detached RGB-frequency and dense feature responses, followed by dilation,
an image-wise quantile threshold, and exponential weighting.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):
    """Weight-standardized convolution used in the released implementation."""

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DepthHeadWithFusion(nn.Module):
    """Predict auxiliary inverse depth and a continuous routing mask."""

    def __init__(self, fdec_channels, fedge_channels,
                 scales=range(4), target_sparsity=0.08):
        super().__init__()
        self.scales = scales
        self.fdec_channels = fdec_channels
        self.target_sparsity = target_sparsity
        total_in = fdec_channels + fedge_channels

        self.fusion_conv = nn.Sequential(
            StdConv2d(total_in, 128, 3, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.SiLU(inplace=True),
            StdConv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(32, 64),
            nn.SiLU(inplace=True),
        )

        self.disp_convs = nn.ModuleDict()
        for s in scales:
            if s == 0:
                self.disp_convs[f"dispconv{s}"] = nn.Sequential(
                    StdConv2d(64, 32, 3, padding=1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(32, 1, 1, bias=True),
                )
            else:
                self.disp_convs[f"dispconv{s}"] = nn.Conv2d(64, 1, 1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """Zero the raw auxiliary-channel slice, as in the released code."""
        with torch.no_grad():
            self.fusion_conv[0].weight[:, self.fdec_channels:, :, :].zero_()
        print(f"  [DepthHead] Fusion weights for auxiliary heads (after ch {self.fdec_channels}) zero-initialized.")

    def _compute_soft_mask(self, E_high, E_dense):
        E_high_det = torch.nan_to_num(E_high.detach(), nan=0.0)
        E_dense_det = E_dense.detach()
        W_raw = torch.max(E_high_det * E_dense_det, torch.pow(E_dense_det, 2.0))
        W_dilated = F.max_pool2d(W_raw, kernel_size=3, stride=1, padding=1)

        B = W_dilated.shape[0]
        W_flat = W_dilated.view(B, -1)
        q_threshold = torch.quantile(
            W_flat, 1.0 - self.target_sparsity, dim=1, keepdim=True
        )
        dynamic_threshold = q_threshold.view(B, 1, 1, 1)
        gamma = 5.0
        W_trigger = F.relu(W_dilated - dynamic_threshold)
        W_soft_mask = 1.0 - torch.exp(-gamma * W_trigger)
        return torch.clamp(W_soft_mask, min=0.0, max=1.0), dynamic_threshold

    def forward(self, F_dec, F_edge, E_high, E_dense):
        outputs = {}
        F_cat = torch.cat([F_dec, F_edge.detach()], dim=1)
        F_fused = self.fusion_conv(F_cat)

        for s in self.scales:
            if s == 0:
                disp = self.sigmoid(self.disp_convs[f"dispconv{s}"](F_fused))
            else:
                F_down = F.interpolate(
                    F_fused, scale_factor=1.0 / (2 ** s),
                    mode="bilinear", align_corners=False
                )
                disp = self.sigmoid(self.disp_convs[f"dispconv{s}"](F_down))
            outputs[("disp", s)] = disp

        W_soft_mask, q_threshold = self._compute_soft_mask(E_high, E_dense)
        outputs["W_soft_mask"] = W_soft_mask
        outputs["q_threshold"] = q_threshold
        outputs["E_high_raw"] = E_high.detach()
        return outputs
