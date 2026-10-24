"""Fixed decoder-feature projection for SEE and RGB-frequency response for SMF."""

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


class FFTEdgeHead(nn.Module):
    """Return projected features and the RGB response used by no-dense SEMF."""

    def __init__(self, in_channels, mid_channels=32, fft_radius=30):
        super().__init__()
        self.fft_radius = fft_radius
        self.mid_channels = mid_channels

        self.reduce = StdConv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.GroupNorm(8, mid_channels)
        self.relu = nn.SiLU(inplace=True)
        # Dormant checkpoint slots, kept in the server's initialization order.
        self.edge_conv = nn.Conv2d(mid_channels, 1, 1, bias=True)
        self.gate_conv = nn.Sequential(
            StdConv2d(1 + mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1, bias=True),
        )
        with torch.no_grad():
            self.gate_conv[-1].bias.fill_(1.0)
        self._hp_cache = {}

    def _get_highpass_mask(self, H, W, device, dtype, radius):
        cache_key = (H, W, radius)
        if cache_key in self._hp_cache:
            return self._hp_cache[cache_key].to(device=device, dtype=dtype)

        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device, dtype=dtype) - cy
        x = torch.arange(W, device=device, dtype=dtype) - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(xx ** 2 + yy ** 2) + 1e-8
        n = 2
        mask = 1.0 / (1.0 + (radius / dist) ** (2 * n))
        self._hp_cache[cache_key] = mask
        return mask

    def _fft_highpass_rgb(self, rgb_input):
        """Return the magnitude of the filtered grayscale image."""
        gray = 0.299 * rgb_input[:, 0:1] + 0.587 * rgb_input[:, 1:2] + 0.114 * rgb_input[:, 2:3]
        B, C, H, W = gray.shape

        F_fft = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
        hp_mask = self._get_highpass_mask(H, W, gray.device, gray.dtype, self.fft_radius)
        F_fft_high = F_fft * hp_mask.view(1, 1, H, W)
        F_fft_high = torch.fft.ifftshift(F_fft_high, dim=(-2, -1))
        E_rgb_high = torch.abs(torch.fft.ifft2(F_fft_high))

        pad = 5
        E_rgb_high[:, :, :pad, :] = 0
        E_rgb_high[:, :, -pad:, :] = 0
        E_rgb_high[:, :, :, :pad] = 0
        E_rgb_high[:, :, :, -pad:] = 0

        E_rgb_high = torch.log1p(E_rgb_high)
        e_max = E_rgb_high.amax(dim=(-2, -1), keepdim=True)
        return E_rgb_high / torch.clamp(e_max, min=0.1)

    def forward(self, F_dec, rgb_input):
        """Return an API placeholder, E_rgb and fixed projected features."""
        if rgb_input is None:
            raise ValueError("SMF requires the original RGB image.")
        with torch.no_grad():
            F_edge = self.relu(self.bn1(self.reduce(F_dec.detach())))
            if rgb_input.shape[-2:] != F_dec.shape[-2:]:
                rgb_input = F.interpolate(rgb_input, size=F_dec.shape[-2:],
                                          mode="bilinear", align_corners=False)
            E_rgb = self._fft_highpass_rgb(rgb_input)
            E_rgb = E_rgb / E_rgb.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return torch.zeros_like(E_rgb), E_rgb, F_edge
