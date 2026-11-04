#!/usr/bin/env python3
"""Supplementary Videos 1-4: the four fields of Fig. 2 at every age from 540 Ma to the present, main model
(rock-chemistry-adjusted surface, calibration of the main run, cellfields.compute).

  Video 1  elevation of the rock-chemistry-adjusted surface (m)
  Video 2  erosion rate (mm/yr); land with net deposition in grey
  Video 3  net inorganic CO2 drawdown (silicate sink minus sulfide source, t C km-2 yr-1)
  Video 4  erosion rate relative to the local drawdown optimum, E/E_opt; land with net deposition in grey

Projection, land outline and colour normalisations are those of Fig. 2, so a frame at 450, 300, 90 or 0 Ma can be
set beside the corresponding column of the figure; the E/E_opt video uses the PuOr_r colour map, the grey of
depositing land is lighter, and the 500 mm/yr runoff contour of Fig. 2c is not drawn. Video 3 is the inorganic
budget as long as the environment variable NET_BUDGET is unset or 'inorg' (the default).

Each frame is 1,920 x 1,080 pixels, rendered in memory and piped to ffmpeg (H.264, yuv420p, 6 frames per second by
default): each video is ONE file and no frame files are written; a video is written under a temporary name and
renamed only when complete, and a finished video of the same name is moved to a _to_delete folder beside it rather
than overwritten. The first frame of each video passes the text-collision audit of _epsl_style before anything is
encoded.

Inputs:  the land-only input cache (nature_recast_cache.zip, build_cache.py) and the calibration amplitudes of the
         main run (data/nature_recast/kin_bridge_s2_amplitudes.txt); needs imageio-ffmpeg
Outputs: figures/nature/videos/Video_S{1,2,3,4}_*.mp4

    python3 make_videos.py --ages 540 300 0 --out-dir /tmp/video_test     # pilot, three frames
    python3 make_videos.py                                              # 109 ages -> figures/nature/videos/
Options: --fps frames per second (default 6); --step plot every n-th 0.1° node (default 2, i.e. 0.2°).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import _nature_style as ns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize, LinearSegmentedColormap, ListedColormap
import imageio_ffmpeg
from _wintri_map import projected_mesh, decorate_global_map, draw_mask_contour
import cellfields as cf
from _load import FIG

DEPO = ListedColormap(['#C4C4C4'])     # land with net deposition: light grey in the erosion-rate and E/E_opt videos
ELEV = LinearSegmentedColormap.from_list('elev', ['#2E6B3E', '#8DB36B', '#E8DDA0', '#C49A5A', '#8C5A3C', '#F2EEEA'])
# per video: file name, field key, colour-bar label, colour map and normalisation, and custom colour-bar ticks (positions,
# labels) or None; 'lx' is log10(E/E_opt), so -2..2 spans 0.01-100 with the optimum at the centre
VIDEOS = [('Video_S1_elevation_adjusted_540-0Ma', 'z', 'Elevation (m)', dict(cmap=ELEV, norm=Normalize(0, 4000)), None),
          ('Video_S2_erosion_rate_540-0Ma', 'E', 'Erosion rate (mm yr$^{-1}$)', dict(cmap='YlOrBr', norm=LogNorm(1e-3, 1.0)), None),
          ('Video_S3_net_CO2_drawdown_540-0Ma', 'netC', 'Net inorganic CO$_2$ drawdown (t C km$^{-2}$ yr$^{-1}$)',
           dict(cmap='RdBu', norm=TwoSlopeNorm(0, -1.5, 3.0)), None),
          ('Video_S4_erosion_relative_to_optimum_540-0Ma', 'lx', 'Erosion rate relative to the local drawdown optimum, $E/E_{opt}$',
           dict(cmap='PuOr_r', norm=TwoSlopeNorm(0, -2, 2)), ([-2, -1, 0, 1, 2], ['0.01', '0.1', '1', '10', '100']))]
W_IN, H_IN, DPI = 12.8, 7.2, 150          # 1920 x 1080 pixels
LABEL, TICK, AGE = 15, 13, 24              # point sizes for screen viewing


def fields(age):
    """Per-cell fields of one age on the 0.1° grid (dict of 2-D arrays), with the derived fields the videos draw:
    netC (t C km-2 yr-1), lx = log10(E/E_opt), depo (land with no net erosion) and E set to NaN where not eroding."""
    g, _ = cf.compute(age)
    g['netC'] = g['net'] * 12.011              # mol C m-2 yr-1 -> t C km-2 yr-1, as in Figs 1, 2 and 4
    with np.errstate(divide='ignore', invalid='ignore'):
        g['lx'] = np.log10(g['x'])
    g['depo'] = (np.nan_to_num(g['mask']) > 0) & (np.nan_to_num(g['E'], nan=1.0) <= 0)
    g['E'] = np.where(g['E'] > 0, g['E'], np.nan)
    return g


def frame(fig, g, age, key, label, kw, ticks, step):
    """Redraw fig as one video frame: the map of field key for this age, an age label and a horizontal colour bar.

    kw holds the colour map and normalisation, ticks optional colour-bar (positions, labels), step the node stride."""
    fig.clf()
    ax = fig.add_axes([0.02, 0.20, 0.96, 0.78])
    sl = (slice(None, None, step), slice(None, None, step))
    X, Y = projected_mesh(g['lon'][::step], g['lat'][::step])
    if key in ('E', 'lx'):
        ax.pcolormesh(X, Y, np.ma.masked_where(~g['depo'][sl], np.ones(g['depo'][sl].shape)), cmap=DEPO, shading='auto', rasterized=True)
    pm = ax.pcolormesh(X, Y, np.ma.masked_invalid(g[key][sl]), shading='auto', rasterized=True, **kw)
    draw_mask_contour(ax, X, Y, np.nan_to_num(g['mask'][sl]), lw=0.6)
    decorate_global_map(ax)
    ax.set_aspect('equal', adjustable='box')                # true Winkel-Tripel proportions
    ax.text(0.015, 0.985, f'{age} Ma' if age else 'Present', transform=ax.transAxes, ha='left', va='top', fontsize=AGE,
            fontweight='bold', color='#30363A', bbox=dict(facecolor='white', edgecolor='#888888', linewidth=0.5, boxstyle='round,pad=0.2'))
    cax = fig.add_axes([0.25, 0.145, 0.50, 0.025])
    cb = fig.colorbar(pm, cax=cax, orientation='horizontal', extend='both')
    cb.ax.tick_params(labelsize=TICK, length=3)
    if ticks:
        cb.set_ticks(ticks[0]); cb.set_ticklabels(ticks[1])
    fig.text(0.5, 0.03, label, ha='center', va='bottom', fontsize=LABEL)


def rgb(fig):
    """Render fig and return its pixels as a contiguous (height, width, 3) uint8 array."""
    fig.canvas.draw()
    return np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])


def main():
    """Parse the options, audit the first frame of every video, then render all ages and encode the four videos."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--ages', nargs='*', type=int)
    ap.add_argument('--out-dir', default=str(FIG / 'videos'))
    ap.add_argument('--fps', type=int, default=6)
    ap.add_argument('--step', type=int, default=2, help='plot every n-th 0.1-degree node (2 = 0.2 degree)')
    a = ap.parse_args()
    ages = a.ages or list(range(540, -1, -5))             # default: the 109 ages, 540 Ma to the present at 5-Myr steps
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    figs = {v[0]: plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor='white') for v in VIDEOS}
    writers, tmp = {}, {}
    g = fields(ages[0])                                    # audit the first frame of every video before encoding anything
    failed = []
    for name, key, label, kw, ticks in VIDEOS:
        frame(figs[name], g, ages[0], key, label, kw, ticks, a.step)
        if ns._e.audit(figs[name], f'{name} (first frame)'):
            failed.append(name)
    if failed:
        raise SystemExit(f'text-collision audit failed for {failed}; nothing encoded')
    for i, age in enumerate(ages):
        g = g if i == 0 else fields(age)                   # the first age is already loaded for the audit
        for name, key, label, kw, ticks in VIDEOS:
            fig = figs[name]
            frame(fig, g, age, key, label, kw, ticks, a.step)
            img = rgb(fig)
            if i == 0:                                     # open one ffmpeg writer per video at the first frame
                tmp[name] = out / f'.{name}.partial.mp4'
                writers[name] = imageio_ffmpeg.write_frames(
                    str(tmp[name]), (img.shape[1], img.shape[0]), fps=a.fps, codec='libx264', pix_fmt_in='rgb24',
                    pix_fmt_out='yuv420p', output_params=['-crf', '20', '-movflags', '+faststart'], macro_block_size=8)
                writers[name].send(None)                   # prime the generator before sending frames
            writers[name].send(img.tobytes())
        print(f'{age} Ma', flush=True)
    for name, w in writers.items():
        w.close()
        final = out / f'{name}.mp4'
        if final.exists():                                  # never overwrite a finished video: move it aside
            aside = out / '_to_delete'; aside.mkdir(exist_ok=True)
            final.rename(aside / f'{name}.{time.strftime("%Y%m%d_%H%M%S")}.mp4')
        tmp[name].rename(final)
        print('wrote', final)


if __name__ == '__main__':
    main()
