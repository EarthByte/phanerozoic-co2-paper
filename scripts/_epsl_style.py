"""EPSL / Elsevier figure style, with automatic text-collision auditing.

Elsevier artwork sizing specifies printed widths of 90 mm (single column),
140 mm (1.5 column) and 190 mm (double column), and requires lettering with a
FINISHED, PRINTED size of at least 7 pt (6 pt for sub/superscripts), with about
10 pt appropriate for prominent elements.

The failure mode this module exists to prevent: a figure authored at 15.2 in
wide and placed at 190 mm is reduced by 0.49, so nominal 8 pt lettering prints
at 3.9 pt. Authoring at the printed width makes the reduction factor 1.0, so
nominal point size IS printed point size.

Authoring at printed size has a second-order cost: lettering keeps its absolute
size while the canvas shrinks, so labels that used to fit can now collide or run
off the canvas. Eyeballing every panel does not scale and does not survive the
next re-run, so `savefig` is wrapped here to audit the figure automatically and
report, on stderr:

  * text-vs-text overlaps (the same defect that stacked the three row labels of
    Fig. 1 into an unreadable column),
  * text running outside the canvas (which silently clipped two of Fig. 8's
    colourbar labels), and
  * any lettering below the 7 pt floor.

Set EPSL_AUDIT=0 to silence, or EPSL_AUDIT=strict to raise instead of warn.

Import this module before creating any figure.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

MM = 1.0 / 25.4
SINGLE = 90 * MM      # 3.543 in
ONEHALF = 140 * MM    # 5.512 in
DOUBLE = 190 * MM     # 7.480 in

BASE = 8.0            # body lettering, comfortably above the 7 pt floor
SMALL = 7.0           # ticks and legends; the floor itself
LARGE = 10.0          # panel letters and the few prominent labels

MIN_PT = 7.0          # Elsevier floor for normal text
MIN_PT_SUB = 6.0      # Elsevier floor for sub/superscripts
OVERLAP_FRAC = 0.18   # ignore slight kerning touches; flag real collisions
INK_TOL = 40          # per-channel 0-255 deviation counting as artwork
                      # rather than background; above a 25%-alpha
                      # gridline on white, below any real mark
INK_FRAC = 0.10       # share of a label's box that may sit on artwork
UNIFORM_FRAC = 0.90   # if this much of the area behind a label is one
                      # flat colour it is a badge, panel or map fill --
                      # deliberate and legible, not a collision. The
                      # known blind spot: a label over a large uniform
                      # fill is never reported.

plt.rcParams.update({
    "font.family": "sans-serif",
    # DejaVu first for full glyph coverage: Helvetica and Arial lack the
    # arrows used in the workflow diagram and rendered them as tofu boxes.
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": BASE,
    "axes.titlesize": LARGE,
    "axes.labelsize": BASE,
    "xtick.labelsize": SMALL,
    "ytick.labelsize": SMALL,
    "legend.fontsize": SMALL,
    "figure.titlesize": LARGE,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    # NOT "tight": a tight bbox silently widens the saved figure past the
    # printed width, so the typesetter scales it back down and the lettering
    # shrinks again -- the exact failure authoring at printed size prevents.
    "savefig.bbox": None,
    "pdf.fonttype": 42,     # embed as TrueType, not Type 3
    "ps.fonttype": 42,
})


def size(width_in, old_w, old_h, min_h=2.0):
    """Aspect-preserving figsize at an EPSL printed width."""
    h = old_h * (width_in / old_w)
    return (width_in, max(h, min_h))


def _texts(fig):
    """Every non-empty Text artist in the figure, with its owning axes."""
    out = []
    for t in fig.texts:
        out.append((t, None))
    for ax in fig.axes:
        for t in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            out.append((t, ax))
        # Only ticks inside the view limits are actually drawn; matplotlib keeps
        # label artists for out-of-range ticks, and counting them produced false
        # "off-canvas" reports.
        if not getattr(ax, "axison", True):
            continue          # axis("off"): tick labels exist but are not drawn
        for locs, labels, lim in ((ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()),
                                  (ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for loc, t in zip(locs, labels):
                if lo - 1e-9 <= loc <= hi + 1e-9:
                    out.append((t, ax))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                out.append((t, ax))
    keep = []
    for t, ax in out:
        try:
            if t is None or not t.get_visible():
                continue
            s = t.get_text()
            if s is None or not str(s).strip():
                continue
            keep.append((t, ax, str(s)))
        except Exception:
            continue
    return keep


def _extent(t, renderer):
    try:
        bb = t.get_window_extent(renderer=renderer)
        if bb.width <= 0 or bb.height <= 0:
            return None
        return bb
    except Exception:
        return None


def _overlap_frac(a, b):
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    if dx <= 0 or dy <= 0:
        return 0.0
    inter = dx * dy
    return inter / min(a.width * a.height, b.width * b.height)


def _text_over_artwork(fig, name, items):
    """How much of each label's box is drawn over something else.

    Re-renders the figure with all text hidden and measures the ink underneath
    each label. Returns a list of problem strings.
    """
    try:
        import numpy as np
    except Exception:
        return []

    engine = "unset"
    hidden = []
    buf = None
    try:
        # Freeze the layout: hiding the text must not let a layout engine
        # re-solve and move the artwork out from under the boxes we measured.
        try:
            engine = fig.get_layout_engine()
            fig.set_layout_engine("none")
        except Exception:
            engine = "unset"
        for t_, _ax, _s, _bb in items:
            if t_.get_visible():
                hidden.append(t_)
                t_.set_visible(False)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.int16)
    except Exception:
        buf = None
    finally:
        for t_ in hidden:
            t_.set_visible(True)
        if engine != "unset":
            try:
                fig.set_layout_engine(engine)
            except Exception:
                pass
        try:
            fig.canvas.draw()
        except Exception:
            pass

    if buf is None:
        return []

    import matplotlib.colors as mcolors
    height, width = buf.shape[:2]

    def rgb255(color):
        r, g, b = mcolors.to_rgb(color)
        return np.array([r * 255, g * 255, b * 255], dtype=np.int16)

    fig_bg = rgb255(fig.get_facecolor())
    out = []
    for t_, ax, s, bb in items:
        try:
            if t_.get_bbox_patch() is not None:      # deliberately boxed
                continue
            # inset by a few per cent so an adjacent spine at the very edge of
            # the text box is not mistaken for artwork underneath it
            dx, dy = 0.06 * bb.width, 0.06 * bb.height
            x0 = int(max(0, round(bb.x0 + dx)))
            x1 = int(min(width, round(bb.x1 - dx)))
            y0 = int(max(0, round(height - bb.y1 + dy)))
            y1 = int(min(height, round(height - bb.y0 - dy)))
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            patch = buf[y0:y1, x0:x1]
            bg = fig_bg
            if ax is not None:
                try:
                    bg = rgb255(ax.get_facecolor())
                except Exception:
                    pass
            # a pixel is artwork if it differs from BOTH the axes background and
            # the figure background: panels sit on the figure, and text can
            # straddle the two
            dev_ax = np.abs(patch - bg).max(axis=2)
            dev_fig = np.abs(patch - fig_bg).max(axis=2)
            frac = float(((dev_ax > INK_TOL) & (dev_fig > INK_TOL)).mean())
            # A single flat colour behind the text is a badge, a panel fill or a
            # filled map region: deliberate, and the text stays legible. Only
            # structure -- a curve, a gradient, a neighbouring panel -- is a
            # collision.
            flat = np.median(patch.reshape(-1, 3), axis=0)
            uniform = float((np.abs(patch - flat).max(axis=2) <= INK_TOL).mean())
            if frac > INK_FRAC and uniform < UNIFORM_FRAC:
                out.append(f"{name}: {100*frac:.0f}% of the area behind "
                           f"{s[:34]!r} is artwork, not background "
                           f"(label printed over other content)")
        except Exception:
            continue
    return out


def audit(fig, name="figure", verbose=True, tight=False):
    """Report text collisions, off-canvas text and undersized lettering.

    Returns a list of human-readable problem strings (empty means clean).
    """
    problems = []
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception as exc:                       # pragma: no cover
        return [f"{name}: could not render for audit ({exc})"]

    items = []
    for t, ax, s in _texts(fig):
        bb = _extent(t, renderer)
        if bb is not None:
            items.append((t, ax, s, bb))

    # 1. lettering below the Elsevier floor (nominal == printed at these widths)
    for t, ax, s, bb in items:
        try:
            pt = float(t.get_fontsize())
        except Exception:
            continue
        if pt < MIN_PT_SUB:
            problems.append(f"{name}: {pt:.1f} pt (below the {MIN_PT_SUB:g} pt "
                            f"sub/superscript floor): {s[:44]!r}")
        elif pt < MIN_PT:
            problems.append(f"{name}: {pt:.1f} pt (below the {MIN_PT:g} pt "
                            f"floor): {s[:44]!r}")

    # 2. text outside the canvas -> silently clipped in the saved file
    fw, fh = fig.canvas.get_width_height()
    for t, ax, s, bb in items:
        if tight:
            break            # a tight bbox expands the canvas to include these
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fw + 1 or bb.y1 > fh + 1:
            over = max(-bb.x0, -bb.y0, bb.x1 - fw, bb.y1 - fh)
            problems.append(f"{name}: text runs {over:.0f} px off-canvas "
                            f"(will be clipped): {s[:44]!r}")

    # 3. text overflowing the box it sits in (schematics draw text inside
    #    patches; a label wider than its box reads as a layout error even
    #    though no two text objects touch)
    boxes = []
    for ax in fig.axes:
        for patch in ax.patches:
            if not isinstance(patch, FancyBboxPatch):
                continue        # shaded bands are not boxes
            try:
                bb = patch.get_window_extent(renderer=renderer)
                if bb.width > 1 and bb.height > 1:
                    boxes.append(bb)
            except Exception:
                continue

    def _inter(a, b):
        dx = min(a.x1, b.x1) - max(a.x0, b.x0)
        dy = min(a.y1, b.y1) - max(a.y0, b.y0)
        return dx * dy if dx > 0 and dy > 0 else 0.0

    for t, ax, s_, bb in items:
        # Own the text to the box it overlaps most. Centre-containment fails on
        # exactly the case that matters: a label so wide its centre falls
        # outside the box would be judged to belong to no box and pass.
        area = bb.width * bb.height
        best, best_area = None, 0.0
        for box in boxes:
            a = _inter(bb, box)
            if a > best_area:
                best, best_area = box, a
        if best is None or best_area < 0.2 * area:
            continue                      # free-standing text, not in a box
        over = max(best.x0 - bb.x0, bb.x1 - best.x1,
                   best.y0 - bb.y0, bb.y1 - best.y1)
        if over > 2:
            problems.append(f"{name}: text overflows its box by {over:.0f} px: "
                            f"{s_[:40]!r}")

    # 4. text-vs-text collisions
    for i in range(len(items)):
        ti, axi, si, bi = items[i]
        for j in range(i + 1, len(items)):
            tj, axj, sj, bj = items[j]
            f = _overlap_frac(bi, bj)
            # A twin axis carries a second copy of the shared tick labels at the
            # same place; identical strings on top of each other are that, not a
            # collision.
            if si == sj and f > 0.9:
                continue
            if f > OVERLAP_FRAC:
                problems.append(f"{name}: text overlap {100*f:.0f}%: "
                                f"{si[:30]!r} vs {sj[:30]!r}")

    # 6. an axis label longer than its axis:
    #    the x label no wider than the axes, the y label (rotated) no taller; axes drawn without a frame (maps) and
    #    colour bars are skipped, colour-bar labels being measured against the bar instead
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:
            continue
        try:
            abb = ax.get_window_extent(renderer=renderer)
        except Exception:
            continue
        for lab, along in ((ax.xaxis.label, 'x'), (ax.yaxis.label, 'y')):
            if not lab.get_visible() or not lab.get_text().strip():
                continue
            lbb = lab.get_window_extent(renderer=renderer)
            size, room = (lbb.width, abb.width) if along == 'x' else (lbb.height, abb.height)
            if size > 1.02 * room:
                problems.append(f"{name}: {along} label {100 * size / room:.0f}% of its axis: {lab.get_text()[:44]!r}")

    # 7. a right-hand axis with tick labels but no axis line
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:
            continue
        right_labels = ax.yaxis.get_label_position() == 'right' and any(
            t.label2.get_visible() and t.label2.get_text() for t in ax.yaxis.get_major_ticks())
        if right_labels and not ax.spines['right'].get_visible():
            problems.append(f"{name}: right-hand axis without an axis line: {ax.yaxis.label.get_text()[:44]!r}")

    # 8. a legend that reaches outside its axes;
    #    legends placed outside on purpose (bbox_to_anchor outside 0-1) are skipped
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible():
            continue
        anchor = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        if anchor.x0 < -0.01 or anchor.x1 > 1.01 or anchor.y0 < -0.01 or anchor.y1 > 1.01:
            continue
        lb = leg.get_window_extent(renderer=renderer); ab = ax.get_window_extent(renderer=renderer)
        over = max(ab.x0 - lb.x0, lb.x1 - ab.x1, ab.y0 - lb.y0, lb.y1 - ab.y1)
        if over > 1:
            problems.append(f"{name}: legend runs {over:.0f} px outside its axes: {[t.get_text() for t in leg.get_texts()][:2]}")

    # 5. text printed over artwork -- inset labels, annotations and frameless
    #    legends sitting on lines, fills, colorbars or a neighbouring panel.
    problems += _text_over_artwork(fig, name, items)

    if problems and verbose:
        print(f"\n[EPSL audit] {name}: {len(problems)} issue(s)", file=sys.stderr)
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)
    elif verbose:
        print(f"[EPSL audit] {name}: clean", file=sys.stderr)
    return problems


# Capture the ORIGINAL savefig once. Without this guard a re-import or
# importlib.reload wraps the already-wrapped function and savefig recurses
# until the stack blows.
_ORIG_SAVEFIG = getattr(Figure.savefig, "_epsl_original", Figure.savefig)


def _savefig_audited(self, fname, *args, **kwargs):
    mode = os.environ.get("EPSL_AUDIT", "1").lower()
    if mode not in ("0", "off", "false"):
        try:
            label = os.path.basename(str(fname))
        except Exception:
            label = "figure"
        # Audit at the dpi this file will be written at. Measuring on the
        # figure's default canvas (100 dpi) while saving at 260 let a clipped
        # title through: glyph advances do not scale exactly with dpi.
        save_dpi = kwargs.get("dpi", None)
        if save_dpi in (None, "figure"):
            save_dpi = self.dpi
        original_dpi = self.dpi
        try:
            try:
                if float(save_dpi) != float(original_dpi):
                    self.set_dpi(float(save_dpi))
            except (TypeError, ValueError):
                pass
            tight = str(kwargs.get("bbox_inches", "")).lower() == "tight"
            problems = audit(self, label, tight=tight)
            # A tight bbox re-widens the figure past the printed width, so the
            # typesetter scales it down again and the lettering shrinks.
            if str(kwargs.get("bbox_inches", "")).lower() == "tight":
                w = self.get_size_inches()[0]
                problems.append(
                    f"{label}: saved with bbox_inches='tight', which widens the "
                    f"{w:.2f} in figure to fit its text; the printed width is then "
                    f"scaled back down and lettering shrinks. Drop it and use "
                    f"tight_layout/constrained_layout instead.")
                if mode != "strict":
                    print(f"  ! {problems[-1]}", file=sys.stderr)
        except Exception as exc:                   # never break a save
            print(f"[EPSL audit] skipped ({exc})", file=sys.stderr)
            problems = []
        finally:
            try:
                if self.dpi != original_dpi:
                    self.set_dpi(original_dpi)
            except Exception:
                pass
        if problems and mode == "strict":
            raise RuntimeError(f"EPSL audit failed for {label}: "
                               f"{len(problems)} issue(s); first: {problems[0]}")
    return _ORIG_SAVEFIG(self, fname, *args, **kwargs)


_savefig_audited._epsl_original = _ORIG_SAVEFIG
Figure.savefig = _savefig_audited
