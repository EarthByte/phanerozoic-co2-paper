#!/usr/bin/env python3
# NeuroLEM model definition, as used for every NeuroLEM field of this study.
# The network is rebuilt from this file and the trained weights of the data archive
# (surrogate_model/best.pt, https://doi.org/10.5281/zenodo.23909273): the checkpoint stores in_ch, depth and base,
# and loads into GlobalUNet with strict=True. The scripts of this repository import it from scripts/;
# in the NeuroLEM package the same file is named salles_unet.py. Training and the rest of NeuroLEM are released separately.
"""
paleomap_unet.py — global UNet that emulates goSPL erosion on the PALEOMAP palaeo-DEMs.

Learns the goSPL erosion operator  (zpaleo, rain[, temp]) -> erate  on the regular lat/lon grid of the
published goSPL simulations (Salles et al., 2023) (0.1 deg native, 1801x3601; any coarsened
multiple works). Design points that make it *global* rather than a generic UNet:

  * **Longitude is periodic** — every conv pads circularly in W (the dateline
    seam) and by replication in H (the poles). A generic UNet would learn a
    false discontinuity at +/-180 deg.
  * **Resolution-agnostic** — works at any (H, W) that the dataset pads to a
    multiple of 2**depth; the trainer crops predictions back and the land mask
    makes the padding inert.
  * **Single-channel signed output** (erate); the loss is applied on land only
    (where goSPL defines erate), so ocean never contributes.

This is the model only; training, data handling and normalisation belong to the NeuroLEM package.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicPad2d(nn.Module):
    """Circular pad in longitude (W), replicate in latitude (H)."""
    def __init__(self, pad: int):
        super().__init__()
        self.p = pad

    def forward(self, x):
        if self.p == 0:
            return x
        x = F.pad(x, (self.p, self.p, 0, 0), mode="circular")     # W wrap
        x = F.pad(x, (0, 0, self.p, self.p), mode="replicate")    # H poles
        return x


class ConvBlock(nn.Module):
    """(periodic 3x3 conv -> GroupNorm -> SiLU) x 2."""
    def __init__(self, cin, cout, groups=8):
        super().__init__()
        g = min(groups, cout)
        while cout % g:
            g -= 1
        self.net = nn.Sequential(
            PeriodicPad2d(1), nn.Conv2d(cin, cout, 3), nn.GroupNorm(g, cout), nn.SiLU(),
            PeriodicPad2d(1), nn.Conv2d(cout, cout, 3), nn.GroupNorm(g, cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class GlobalUNet(nn.Module):
    """Global lat/lon UNet. in_ch -> out_ch, `depth` down/up levels, `base` width."""
    def __init__(self, in_ch: int, out_ch: int = 1, base: int = 48, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth + 1)]     # e.g. 48,96,192,384,768
        self.inc = ConvBlock(in_ch, chs[0])
        self.down = nn.ModuleList([ConvBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up_conv = nn.ModuleList(
            [ConvBlock(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        skips = []
        x = self.inc(x)
        for d in self.down:
            skips.append(x)
            x = d(self.pool(x))
        for i, uc in enumerate(self.up_conv):
            skip = skips[-(i + 1)]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = uc(torch.cat([x, skip], dim=1))
        return self.head(x)

    @property
    def size_multiple(self) -> int:
        """Inputs' H and W must be a multiple of this (pad in the dataset)."""
        return 2 ** self.depth


if __name__ == "__main__":
    # quick shape check
    m = GlobalUNet(in_ch=2, depth=4, base=16)
    x = torch.randn(1, 2, 16 * 28, 16 * 57)          # multiple of 16
    y = m(x)
    print("in", tuple(x.shape), "-> out", tuple(y.shape),
          "| params", sum(p.numel() for p in m.parameters()) / 1e6, "M")
