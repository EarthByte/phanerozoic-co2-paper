#!/usr/bin/env python3
"""Extended Data Fig. 1: how erosion and weathering are computed, step by step."""
import _nature_style as ns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _load import FIG

fig = plt.figure(figsize=(ns.DOUBLE, 2.95)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
def box(x, y, w, h, head, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec='#444444', lw=0.6))
    ax.text(x + w / 2, y + h - 3.0, head, ha='center', va='top', fontsize=ns.BASE, fontweight='bold')
    ax.text(x + w / 2, y + h - 11.5, body, ha='center', va='top', fontsize=ns.SMALL, linespacing=1.3)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.8, color='#444444'))
IN, MOD, OUT_ = '#E8F0F8', '#FCEFD9', '#E6F2E6'
box(2, 56, 22, 40, 'Palaeoelevation', 'PALEOMAP elevation model,\nadjusted with crustal thickness\nfrom igneous-rock chemistry\n(109 ages, 0.1° grid)', IN)
box(2, 6, 22, 40, 'Palaeoclimate', 'HadCM3L simulations of the\nsame palaeogeographies:\nprecipitation, runoff ratio,\nsurface-air temperature', IN)
box(30, 56, 30, 40, 'Landscape evolution', 'goSPL routes rain downslope into rivers;\nthe stream power law erodes each river bed\nin proportion to discharge and slope;\nhillslopes diffuse; sediment is deposited\nin basins and offshore. NeuroLEM, a neural\nnetwork trained on goSPL, reproduces\nthe erosion map at low cost', MOD)
box(66, 56, 32, 40, 'Erosion rate in every cell', 'one global factor scales all rates to\nmodern sediment delivery to the oceans;\nmodern mean 0.065 mm yr$^{-1}$\n(cosmogenic nuclides: 0.063 ± 0.015 mm yr$^{-1}$)', OUT_)
box(30, 6, 24, 40, 'Weathering laws', 'fitted to stream chemistry in Taiwan,\nSichuan and New Zealand:\nsilicate sink and sulfide source as\nfunctions of erosion rate, runoff\nand temperature; rock organic\ncarbon as an extension', MOD)
box(60, 6, 17, 40, 'Fluxes in every cell', 'silicate CO$_2$ sink,\nsulfide CO$_2$ source,\nnet drawdown and\nits derivative with\nrespect to erosion', OUT_)
box(83, 6, 15, 40, 'Global results', 'sinks and sources;\nerosion sensitivity;\nland beyond the\nlocal optimum;\nkinetic share;\nweathering intensity', OUT_)
arrow(24.6, 76, 29.4, 76); arrow(24.6, 40, 29.6, 55.2); arrow(60.6, 76, 65.4, 76); arrow(72, 55.4, 50, 46.8)
arrow(24.6, 26, 29.4, 26); arrow(54.6, 26, 59.4, 26); arrow(77.6, 26, 82.4, 26)
fig.savefig(FIG / 'ed1_workflow.png', dpi=400); fig.savefig(FIG / 'ed1_workflow.pdf')
