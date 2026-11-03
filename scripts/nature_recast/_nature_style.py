"""Nature figure style, reusing the automatic text-collision audit of `_epsl_style`.

Nature artwork: 89 mm (single column) or 183 mm (double column) wide, at most
170 mm deep (allowing for the legend beneath), sans-serif lettering of 5-7 pt at
final size, and bold lower-case 8 pt panel letters. Figures carry no titles; the
descriptive text belongs in the legend. Figures are authored at printed size, so
nominal point size is printed point size, and every `savefig` is audited for text
overlaps, clipped text, text over artwork and undersized lettering.
"""
import _config                     # noqa: F401  (adds the parent scripts folder to the path)
import _epsl_style as _e          # installs the audited savefig wrapper
import matplotlib.pyplot as plt

MM = 1.0 / 25.4
SINGLE = 89 * MM
DOUBLE = 183 * MM
MAX_DEPTH = 170 * MM

BASE = 7.0
SMALL = 6.0
PANEL = 8.0

_e.MIN_PT = 5.0
_e.MIN_PT_SUB = 5.0

plt.rcParams.update({
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
    "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.fontsize": SMALL,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

# colour-blind-safe palette (Okabe & Ito)
C = dict(adj="#D55E00", ref="#0072B2", sil="#009E73", sulf="#CC79A7", net="#000000",
         oc="#E69F00", grey="#7F7F7F", light="#BBBBBB", sky="#56B4E9", band="#F0E442")

PERIODS = [("Cm", 538.8, 485.4), ("O", 485.4, 443.8), ("S", 443.8, 419.2), ("D", 419.2, 358.9),
           ("C", 358.9, 298.9), ("P", 298.9, 251.9), ("Tr", 251.9, 201.4), ("J", 201.4, 145.0),
           ("K", 145.0, 66.0), ("Pg", 66.0, 23.03), ("N", 23.03, 0.0)]


def panel_letter(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL, fontweight="bold", va="bottom", ha="left")


def timescale(ax, height=0.06):
    """Geological period strip along the bottom of a time-series axis (age increases to the left)."""
    import matplotlib.transforms as mt
    tr = mt.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (name, a0, a1) in enumerate(PERIODS):
        ax.add_patch(plt.Rectangle((a1, 0), a0 - a1, height, transform=tr, facecolor="#E8E8E8" if i % 2 else "#FFFFFF",
                                   edgecolor="#777777", lw=0.3, clip_on=True, zorder=4))
        if a0 - a1 > 22:
            ax.text(0.5 * (a0 + a1), height / 2, name, transform=tr, ha="center", va="center", fontsize=5.0, zorder=5)


def equal_gaps(fig, axes):
    """Space a row of panels so that the white gaps between them are equal: the gap is measured from the right edge of one
    panel to the leftmost ink of the next (its tick labels, axis label and panel letter), not between axes boxes, which
    differ when the tick labels differ in width. Widths and the outer edges are kept."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); W = fig.bbox.width
    pos = [a.get_position() for a in axes]
    L = [p.x0 - a.get_tightbbox(r).x0 / W for a, p in zip(axes, pos)]          # ink to the left of each axes box
    R = [max(0.0, a.get_tightbbox(r).x1 / W - p.x1) for a, p in zip(axes, pos)]
    w = pos[0].width; n = len(axes)
    g = (pos[-1].x1 + R[-1] - pos[0].x0 - n * w - sum(R[:-1]) - R[-1] - sum(L[1:])) / (n - 1)
    x = pos[0].x0
    for i, (a, p) in enumerate(zip(axes, pos)):
        a.set_position([x, p.y0, w, p.height])
        if i + 1 < n:
            x = x + w + R[i] + g + L[i + 1]
    return g
