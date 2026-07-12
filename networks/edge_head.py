"""
FFT Edge Head v2 — Dual-Stream with Learnable Texture Gate
==========================================================
[FIX v7] Root cause: old version applied FFT to E_dense (learned feature map),
whose frequency content bears NO relation to original image spatial frequencies.
This is like frequency-filtering scrambled noise.

New architecture (based on IJCAI 2025 FAD / ECCV 2024 HS-FPN):

    Stream A (Physics):  RGB → grayscale → FFT high-pass → E_rgb_high
                         Captures ALL high-frequency content (texture + geometry)

    Stream B (Semantics): F_dec → Conv → E_feat (semantic edge awareness)
                         Knows WHERE object boundaries are

    Fusion:  Gate = σ(Conv(Cat(E_rgb_high, E_feat)))
             E_high = E_rgb_high × Gate
             → Only keeps edges where BOTH physics AND semantics agree

This eliminates texture-on-bricks false positives while preserving
eaves/columns/lattice geometry edges.

Outputs:
    E_dense: (B, 1, H, W) raw semantic edge heatmap [0, 1]
    E_high:  (B, 1, H, W) physics+semantics fused edge [0, 1]
    F_edge:  (B, mid_channels, H, W) intermediate edge feature for depth_head
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):
    """[SOTA] Weight Standardization Convolution Layer"""
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class FFTEdgeHead(nn.Module):
    """
    Dual-Stream FFT Edge Head with Learnable Texture Gate.

    Stream A: Direct RGB FFT high-pass (physics-based, non-learnable)
    Stream B: Semantic feature edge (learnable, depth-aware)
    Gate: Learned fusion that suppresses texture edges

    Args:
        in_channels: channels of F_dec from backbone
        mid_channels: intermediate feature dimension (default 32)
        fft_radius: radius for Butterworth high-pass filter (default 30)
    """

    def __init__(self, in_channels, mid_channels=32, fft_radius=30):
        super().__init__()
        self.fft_radius = fft_radius
        self.mid_channels = mid_channels

        # ===== Stream B: Semantic feature extraction =====
        self.reduce = StdConv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.GroupNorm(8, mid_channels)
        self.relu = nn.SiLU(inplace=True)

        # Semantic edge prediction (E_dense)
        self.edge_conv = nn.Conv2d(mid_channels, 1, 1, bias=True)

        # ===== Texture Gate: learns to suppress texture, pass geometry =====
        # Input: Cat(E_rgb_high[1ch], E_feat_semantic[mid_channels])
        # Output: Gate[1ch] in [0,1] — 0=texture, 1=geometry
        self.gate_conv = nn.Sequential(
            StdConv2d(1 + mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1, bias=True),
        )

        # Initialize gate bias slightly positive so early training passes edges through
        with torch.no_grad():
            self.gate_conv[-1].bias.fill_(1.0)

        # Cached high-pass mask (lazy init on first forward)
        self._cached_hp_mask = None
        self._cached_hp_shape = (0, 0)

    def _get_highpass_mask(self, H, W, device, dtype, radius):
        """Get cached Butterworth High-pass Filter with specific radius."""
        cache_key = (H, W, radius)
        if hasattr(self, '_hp_cache') and cache_key in self._hp_cache:
            return self._hp_cache[cache_key].to(device=device, dtype=dtype)

        if not hasattr(self, '_hp_cache'):
            self._hp_cache = {}

        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device, dtype=dtype) - cy
        x = torch.arange(W, device=device, dtype=dtype) - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx ** 2 + yy ** 2) + 1e-8

        n = 2  # Butterworth order
        mask = 1.0 / (1.0 + (radius / dist) ** (2 * n))
        self._hp_cache[cache_key] = mask
        return mask

    def _fft_highpass_rgb(self, rgb_input):
        """[Stream A] Apply FFT high-pass directly to RGB image.

        This operates on the ORIGINAL image, so frequency content correctly
        corresponds to spatial frequencies (eaves = sharp step edges = broadband
        high freq; brick texture = periodic narrow-band high freq).

        Args:
            rgb_input: (B, 3, H, W) original RGB image in [0, 1]

        Returns:
            E_rgb_high: (B, 1, H, W) high-frequency magnitude, normalized to [0, 1]
        """
        # Convert to grayscale for single-channel FFT
        # Using luminance weights (ITU-R BT.601)
        gray = 0.299 * rgb_input[:, 0:1] + 0.587 * rgb_input[:, 1:2] + 0.114 * rgb_input[:, 2:3]
        # gray: (B, 1, H, W)

        B, C, H, W = gray.shape

        # 2D FFT on grayscale
        F_fft = torch.fft.fft2(gray)
        F_fft = torch.fft.fftshift(F_fft, dim=(-2, -1))

        # Apply Butterworth high-pass filter
        hp_mask = self._get_highpass_mask(H, W, gray.device, gray.dtype, self.fft_radius)
        hp_mask = hp_mask.view(1, 1, H, W)
        F_fft_high = F_fft * hp_mask

        # Inverse FFT
        F_fft_high = torch.fft.ifftshift(F_fft_high, dim=(-2, -1))
        E_rgb_high = torch.fft.ifft2(F_fft_high)
        E_rgb_high = torch.abs(E_rgb_high)

        # Kill border artifacts (FFT boundary discontinuity)
        pad = 5
        E_rgb_high[:, :, :pad, :] = 0
        E_rgb_high[:, :, -pad:, :] = 0
        E_rgb_high[:, :, :, :pad] = 0
        E_rgb_high[:, :, :, -pad:] = 0

        # Log1p + safe normalization
        E_rgb_high = torch.log1p(E_rgb_high)
        e_max = E_rgb_high.amax(dim=(-2, -1), keepdim=True)
        safe_max = torch.clamp(e_max, min=0.1)
        E_rgb_high = E_rgb_high / safe_max

        return E_rgb_high  # (B, 1, H, W) in [0, 1]

    def forward(self, F_dec, rgb_input=None):
        """
        Args:
            F_dec:      (B, C, H, W) decoded feature from backbone
            rgb_input:  (B, 3, H, W) original RGB image (required for Stream A)

        Returns:
            E_dense: (B, 1, H, W) raw semantic edge heatmap [0, 1]
            E_high:  (B, 1, H, W) physics+semantics fused edge [0, 1]
            F_edge:  (B, mid_channels, H, W) intermediate edge feature
        """
        # ===== Stream B: Semantic edge features =====
        F_edge = self.relu(self.bn1(self.reduce(F_dec)))  # (B, mid, H, W)
        E_dense = torch.sigmoid(self.edge_conv(F_edge))   # (B, 1, H, W)

        # ===== Stream A: RGB FFT high-pass =====
        if rgb_input is not None:
            # Resize RGB to match feature resolution if needed
            if rgb_input.shape[-2:] != F_dec.shape[-2:]:
                rgb_resized = F.interpolate(
                    rgb_input, size=F_dec.shape[-2:],
                    mode='bilinear', align_corners=False)
            else:
                rgb_resized = rgb_input

            with torch.no_grad():  # FFT branch is non-learnable (physics)
                E_high = self._fft_highpass_rgb(rgb_resized)  # (B, 1, H, W)

            # Normalize output to [0, 1]
            e_max = E_high.amax(dim=(-2, -1), keepdim=True)
            E_high = E_high / torch.clamp(e_max, min=1e-6)
        else:
            # Fallback: use old behavior (FFT on E_dense) if no RGB provided
            E_high = E_dense

        return E_dense, E_high, F_edge
