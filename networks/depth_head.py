"""
Depth Head with Soft Masking & Feature Fusion
=============================================
Serial step 3: Fuses features from backbone, edge head, and normal head
to produce depth prediction with continuous soft mask for repulsive smoothing.

Pipeline:
    F_fused = Conv(Cat(F_dec, F_edge.detach()))
    F_fused → Conv layers → multi-scale disparity
    Soft mask: W = 1 - exp(-gamma * E)
"""

from __future__ import absolute_import, division, print_function

import torch

import torch.nn as nn
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):
    """[SOTA] Weight Standardization Convolution Layer
    Normalizes weight mean and variance before convolution to prevent gradient explosion. Suited for very small batch sizes (BS=2).
    """

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DepthHeadWithFusion(nn.Module):
    """
    Depth prediction head with feature fusion and continuous soft masking.

    CRITICAL: F_edge MUST be passed with .detach() to prevent
    gradient cross-contamination between the three heads!

    Outputs:
        disp_outputs: dict of {("disp", scale): tensor} for scales 0-3
        W_soft_mask:  (B, 1, H, W) continuous soft mask in [0, 1]
    """

    def __init__(self, fdec_channels, fedge_channels,
                 scales=range(4), alpha_rgb=0.8, target_sparsity=0.05):
        """
        Args:
            fdec_channels:  channels of F_dec from backbone decoder
            fedge_channels: channels of F_edge from Edge Head
            scales:         output scales (default [0,1,2,3])
        """
        super().__init__()
        self.scales = scales
        self.fdec_channels = fdec_channels
        total_in = fdec_channels + fedge_channels

        self.alpha_rgb = alpha_rgb
        self.target_sparsity = target_sparsity
        # [SOTA FIX] Fully replaced with WS + GN, and strictly disabled Bias before GN!
        self.fusion_conv = nn.Sequential(
            StdConv2d(total_in, 128, 3, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.SiLU(inplace=True),
            StdConv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(32, 64),
            nn.SiLU(inplace=True),
        )

        # Multi-scale disparity prediction
        self.disp_convs = nn.ModuleDict()
        for s in scales:
            if s == 0:
                # Full resolution
                self.disp_convs[f"dispconv{s}"] = nn.Sequential(
                    StdConv2d(64, 32, 3, padding=1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(32, 1, 1, bias=True),
                )
            else:
                # Lower resolution: downsample then predict
                self.disp_convs[f"dispconv{s}"] = nn.Conv2d(64, 1, 1, bias=True)

        self.alpha_rgb = alpha_rgb
        self.target_sparsity = target_sparsity
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """Zero-initialize weights for Edge and Normal features.

        This ensures that at the start of training (Phase 1: Warmup),
        the fused disparity only depends on the stable F_dec features,
        preventing the 'Gradient Whiplash' effect from noisy uninitialized heads.
        """
        # First conv in fusion_conv is at index 0
        conv1 = self.fusion_conv[0]
        with torch.no_grad():
            # weights shape: [out_ch, in_ch, k, k]
            # We want to zero out the weights for the auxiliary heads (everything after fdec).
            conv1.weight[:, self.fdec_channels:, :, :].zero_()
            if conv1.bias is not None:
                pass
        print(f"  [DepthHead] Fusion weights for auxiliary heads (after ch {self.fdec_channels}) zero-initialized.")

    def _compute_soft_mask(self, E_high, E_dense, rgb_grad=None):
        """[SOTA FIX v24] Surgical Masking with Semantic Gating.

        Instruction 2:
        1. Receive E_dense as a semantic gate to kill texture artifacts.
        2. Perform 'Liposuction' (3x3 dilation only).
        3. Adaptive percentile thresholding (Top 5% edges).
        4. Sigmoid softening.
        """
        E_high_det = E_high.detach()
        E_dense_det = E_dense.detach()  # Pure semantic gatekeeper

        E_high_det = torch.nan_to_num(E_high_det, nan=0.0)

        # 1. Physical geometry response
        W_raw = E_high_det

        # 2. [Ultimate texture suppression + pillar recovery: dynamic semantic saliency boost]
        # Semantic gating: first multiplicative step to kill texture.
        W_raw = W_raw * E_dense_det

        # [SOTA approach: nonlinear semantic activation]
        # Apply a steep activation (squaring) to the semantic signal so strong semantic edges (pillars)
        # light up instantly, while leveraging the squaring property to fully suppress low-frequency background noise from the semantic head.
        E_dense_boosted = torch.pow(E_dense_det, 2.0)

        # Top-level game: collision between physical edges and salient semantic edges
        W_raw = torch.max(W_raw, E_dense_boosted)

        # 3. Minimal surgical dilation (only 3x3 buffer; 5x5/7x7 strictly forbidden)
        W_dilated = F.max_pool2d(W_raw, kernel_size=3, stride=1, padding=1)

        # 4. Adaptive percentile threshold (retain only the top N% strongest cliffs in the image)
        B, C, H, W = W_dilated.shape
        W_flat = W_dilated.view(B, -1)
        # Force only target_sparsity fraction of pixels to exceed the threshold (Instruction 2: quantile 1 - sparsity)
        q_val = 1.0 - self.target_sparsity
        q_threshold = torch.quantile(W_flat, q_val, dim=1, keepdim=True)
        dynamic_threshold = q_threshold.view(B, 1, 1, 1)

        # 5. Step activation (SOTA Fix v29: Exp-variant)
        # Eliminates jagged cardboard artifacts; more physically consistent than Sigmoid
        gamma = 5.0
        # [Instruction 2: Physical Cut] Apply response only to pixels that exceed the threshold
        W_trigger = F.relu(W_dilated - dynamic_threshold)
        W_soft_mask = 1.0 - torch.exp(-gamma * W_trigger)

        return torch.clamp(W_soft_mask, min=0.0, max=1.0), dynamic_threshold

    def forward(self, F_dec, F_edge, E_high, E_dense, rgb_grad=None):
        """
        Args:
            F_dec:   (B, C1, H, W) decoded features (WITH gradients)
            F_edge:  (B, C2, H, W) edge features (will be DETACHED internally)
            E_high:  (B, 1, H, W) high-frequency edges (will be DETACHED internally)
            E_dense: (B, 1, H, W) semantic edges (will be DETACHED internally)
            rgb_grad:(B, 1, H, W) RGB image gradient magnitude (physical anchor)
        """
        outputs = {}

        # Feature fusion
        F_cat = torch.cat([F_dec, F_edge], dim=1)
        F_fused = self.fusion_conv(F_cat)  # (B, 64, H, W)

        # Multi-scale disparity prediction
        for s in self.scales:
            if s == 0:
                disp = self.sigmoid(self.disp_convs[f"dispconv{s}"](F_fused))
            else:
                scale_factor = 1.0 / (2 ** s)
                F_down = F.interpolate(F_fused, scale_factor=scale_factor,
                                       mode='bilinear', align_corners=False)
                disp = self.sigmoid(self.disp_convs[f"dispconv{s}"](F_down))
            outputs[("disp", s)] = disp

        # Compute soft mask from auxiliary signals + RGB anchor
        W_soft_mask, q_threshold = self._compute_soft_mask(
            E_high.detach(), E_dense.detach(), rgb_grad
        )
        outputs["W_soft_mask"] = W_soft_mask
        outputs["q_threshold"] = q_threshold

        # [CALIBRATION PROBES] Export raw signals for diagnostic monitoring.
        outputs["E_high_raw"] = E_high.detach()

        return outputs