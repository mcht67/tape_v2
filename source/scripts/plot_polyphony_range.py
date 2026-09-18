#!/usr/bin/env python3
"""
Polyphony-range figure: for a PolyBirdMix soundscape segment with no exact
ground-truth polyphony degree, shows how a plausible [min, max] range is
constructed from call-level annotations -- (a) the spectrogram with each
detected call's time/frequency box, (b) the calls as a Gantt-style timeline,
(c) the resulting simultaneous-call count over time, and (d) how the two
candidate lower bounds (max simultaneous overlap, number of unique species)
and the upper bound (total call detections) combine into the range.

Renders two contrasting example clips from mcht67/PolyBirdMix's
POW_soundscape_test config (test_5s split) -- one where the lower bound is
set by unique-species count, one by true temporal overlap -- fetched once
via source/scripts/fetch_polyphony_range_example.py and stored under
plots/data/polyphony_range/. Follows the same conventions as
plot_masking_procedure.py / plot_source_separation.py: plain argparse CLI,
shared plot_style helpers, output under plots/figures/.

Usage:
    complete-venv/bin/python source/scripts/plot_polyphony_range.py \\
        [--data-dir plots/data/polyphony_range] [--out-dir plots/figures/polyphony_range]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import librosa
import librosa.display
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_style import save_fig, ONE_COL_WIDTH_IN, TWO_COL_WIDTH_IN, FIGURE_FONT_PT, FIGURE_SERIF_FONTS

# Sized for a single column of the target document (see plot_style.py) so
# LaTeX places it at \includegraphics[width=\columnwidth] with no additional
# scaling -- that's what makes FONT_SIZE come out physically correct on the
# page. FIG_HEIGHT_IN also matters for that, not just FIG_WIDTH_IN:
# \includegraphics[width=...] scales height to preserve the saved image's own
# pixel aspect ratio, so an author-time aspect ratio far from what a printed
# column can hold (e.g. the ~1:3.5 this used to be) reproduces at an absurd
# printed height even though the width comes out exactly right -- panel (d)
# in particular has to stay compact (a few text lines, not a tall multi-box
# flow diagram) to keep the overall aspect ratio sane. Panel (c)'s
# reference-line labels sit inside the axes (a legend) rather than hanging in
# the margin, since anything outside the axes would make bbox_inches="tight"
# silently grow the saved figure past the intended column width.
FIG_WIDTH_IN = ONE_COL_WIDTH_IN
FIG_HEIGHT_IN = 6.2
FONT_SIZE = FIGURE_FONT_PT
N_FFT = 1024
DYNAMIC_RANGE_DB = 80.0
SHADING = "gouraud"
FREQ_YLIM = (200, 9000)

SPECIES_PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
COMMON_NAMES = {"eastow": "Eastern Towhee", "woothr": "Wood Thrush", "tuftit": "Tufted Titmouse"}
OVERLAP_COLOR = "#c44e52"
SPECIES_COUNT_COLOR = "#8172b3"
RANGE_COLOR = "#55a868"


def _set_rcparams(font_size=FONT_SIZE):
    plt.rcParams.update({
        "font.size": font_size,
        "font.family": "serif",
        "font.serif": FIGURE_SERIF_FONTS,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
    })


def simultaneous_call_counts(start_times, end_times, t_end):
    """Piecewise-constant simultaneous-call count over [0, t_end], evaluated
    between consecutive event boundaries. Returns (boundaries, counts,
    max_overlap) with `counts` one shorter than `boundaries`, suitable for
    ax.step(..., where='post') after appending the final boundary's value."""
    boundaries = sorted({0.0, t_end, *start_times, *end_times})
    counts = []
    for t0, t1 in zip(boundaries[:-1], boundaries[1:]):
        mid = (t0 + t1) / 2
        counts.append(sum(1 for s, e in zip(start_times, end_times) if s <= mid < e))
    counts.append(counts[-1])
    return boundaries, counts, max(counts)


# Panel-drawing helpers shared between build_figure (single narrow column)
# and build_figure_grid (2x2, spanning both columns) -- the grid's per-panel
# cells end up close enough in width to the single-column layout's panels
# that the same drawing code (legend placement, label wrapping, etc.) works
# for both; only the outer axes layout differs between the two figures.

def _draw_panel_a(ax_a, cax_a, y, sr, starts, ends, lows, highs, codes, species_color, font_size, show_xlabel):
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=N_FFT // 4)), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=N_FFT // 4, x_axis="time", y_axis="log",
                                    cmap="magma", vmin=-DYNAMIC_RANGE_DB, vmax=0.0, ax=ax_a,
                                    rasterized=True, shading=SHADING)
    ax_a.set_ylim(*FREQ_YLIM)
    for s, e, lo, hi, code in zip(starts, ends, lows, highs, codes):
        ax_a.add_patch(patches.Rectangle((s, lo), e - s, hi - lo, fill=False,
                                          edgecolor=species_color[code], linewidth=1.6))
    ax_a.set_title("(a) Spectrogram", loc="left")
    if show_xlabel:
        ax_a.set_xlabel("Time [s]")
    else:
        ax_a.set_xlabel("")
        ax_a.tick_params(labelbottom=False)
    ax_a.set_ylabel("Frequency [Hz]")
    fig = ax_a.figure
    fig.colorbar(img, cax=cax_a, format="%+2.0f dB")
    unique_species = sorted(species_color.keys())
    legend_handles = [patches.Patch(facecolor="none", edgecolor=species_color[c], linewidth=1.6,
                                     label=COMMON_NAMES.get(c, c)) for c in unique_species]
    # Lower right, not the more common upper right: call boxes tend to sit in
    # the upper/mid frequency range, so the legend is less likely to sit on
    # top of one there.
    ax_a.legend(handles=legend_handles, loc="lower right", framealpha=0.9, handlelength=1.2,
                fontsize=font_size - 1, borderpad=0.4, labelspacing=0.3)


def _draw_panel_b(ax_b, starts, ends, codes, species_color, n_events, duration, font_size, show_xlabel):
    for i, (s, e, code) in enumerate(zip(starts, ends, codes)):
        ax_b.barh(i, e - s, left=s, height=0.5, color=species_color[code], edgecolor="black", linewidth=0.6)
    ax_b.set_yticks(range(n_events))
    ax_b.set_yticklabels([f"call {i + 1}" for i in range(n_events)])
    ax_b.invert_yaxis()
    ax_b.set_ylim(n_events - 0.3, -0.7)
    # Explicit, not inherited via sharex: build_figure_grid's (b) isn't
    # stacked under (a), so without this its x-axis auto-scales to the bars'
    # own span instead of the full 0-duration clip.
    ax_b.set_xlim(0, duration)
    ax_b.set_title("(b) Call events", loc="left")
    if show_xlabel:
        ax_b.set_xlabel("Time [s]")
    else:
        ax_b.tick_params(labelbottom=False)


def _draw_panel_c(ax_c, boundaries, counts, max_overlap, n_unique_species, duration, font_size):
    ax_c.step(boundaries, counts, where="post", color="black", linewidth=1.3)
    ax_c.fill_between(boundaries, counts, step="post", color="black", alpha=0.08)
    # Labels for the reference lines sit outside the axes (in the margin
    # already reserved for (a)'s colorbar column, which is otherwise blank at
    # this row) rather than in an in-axes legend box -- the lines span the
    # panel's full width, so no in-axes corner avoids sitting on top of them.
    side_label_transform = mtransforms.blended_transform_factory(ax_c.transAxes, ax_c.transData)
    if max_overlap == n_unique_species:
        ax_c.axhline(max_overlap, color=RANGE_COLOR, linestyle="--", linewidth=1.1)
        ax_c.text(1.03, max_overlap, f"overlap =\nspecies = {max_overlap}", transform=side_label_transform,
                  color=RANGE_COLOR, fontsize=font_size - 1, va="center", ha="left")
    else:
        ax_c.axhline(max_overlap, color=OVERLAP_COLOR, linestyle="--", linewidth=1.1)
        ax_c.axhline(n_unique_species, color=SPECIES_COUNT_COLOR, linestyle="--", linewidth=1.1)
        ax_c.text(1.03, max_overlap, f"overlap = {max_overlap}", transform=side_label_transform,
                  color=OVERLAP_COLOR, fontsize=font_size - 1, va="center", ha="left")
        ax_c.text(1.03, n_unique_species, f"species = {n_unique_species}", transform=side_label_transform,
                  color=SPECIES_COUNT_COLOR, fontsize=font_size - 1, va="center", ha="left")
    ax_c.set_ylim(-0.3, max(max_overlap, n_unique_species) + 1.4)
    ax_c.set_yticks(range(0, max(max_overlap, n_unique_species) + 2))
    ax_c.set_xlim(0, duration)
    ax_c.set_xlabel("Time [s]")
    ax_c.set_ylabel("Concurrent calls")
    ax_c.set_title("(c) Call count", loc="left")


def _draw_panel_d(ax_d, max_overlap, n_unique_species, min_polyphony, max_polyphony, font_size):
    ax_d.set_title("(d) Range construction", loc="left")
    ax_d.set_xlim(0, 4)
    ax_d.set_ylim(0, 4)
    ax_d.axis("off")

    ax_d.text(1.0, 3.75, f"max overlap = {max_overlap}", ha="center", va="center",
              fontsize=font_size - 1, color=OVERLAP_COLOR)
    ax_d.text(3.0, 3.75, f"unique species = {n_unique_species}", ha="center", va="center",
              fontsize=font_size - 1, color=SPECIES_COUNT_COLOR)
    # Small converging arrows make the max(...) relation visible at a glance,
    # rather than requiring the reader to parse it out of the text alone.
    arrow_kwargs = dict(arrowstyle="-|>", linewidth=1.1, shrinkA=4, shrinkB=4)
    ax_d.annotate("", xy=(1.8, 3.15), xytext=(1.25, 3.55),
                  arrowprops=dict(color=OVERLAP_COLOR, **arrow_kwargs))
    ax_d.annotate("", xy=(2.2, 3.15), xytext=(2.75, 3.55),
                  arrowprops=dict(color=SPECIES_COUNT_COLOR, **arrow_kwargs))
    ax_d.text(2.0, 2.85, f"min = max(overlap, species) = {min_polyphony}", ha="center", va="center",
              fontsize=font_size - 1, color=RANGE_COLOR)
    ax_d.text(2.0, 2.05, f"max = total call detections = {max_polyphony}", ha="center", va="center",
              fontsize=font_size - 1, color="0.25")

    # Number line summarizing the resulting [min, max] range.
    line_x0, line_x1, line_y = 0.4, 3.6, 1.0
    n_ticks = max_polyphony + 2
    tick_x = lambda v: line_x0 + (line_x1 - line_x0) * v / (n_ticks - 1)
    ax_d.axvspan(tick_x(min_polyphony), tick_x(max_polyphony), ymin=(line_y - 0.35) / 4, ymax=(line_y + 0.35) / 4,
                 color=RANGE_COLOR, alpha=0.25, zorder=1)
    ax_d.hlines(line_y, line_x0, line_x1, color="0.3", linewidth=1.3, zorder=2)
    for v in range(n_ticks):
        ax_d.vlines(tick_x(v), line_y - 0.12, line_y + 0.12, color="0.3", linewidth=1.1, zorder=2)
        ax_d.text(tick_x(v), line_y - 0.4, str(v), ha="center", va="center", fontsize=font_size - 1)
    ax_d.scatter([tick_x(min_polyphony), tick_x(max_polyphony)], [line_y, line_y], marker="v", s=40,
                 color=RANGE_COLOR, zorder=3)
    ax_d.text(2.0, 0.25, f"plausible range: [{min_polyphony}, {max_polyphony}]",
              ha="center", va="center", fontsize=font_size, color=RANGE_COLOR)


def _example_fields(ann):
    starts = ann["start_time"]
    ends = ann["end_time"]
    lows = ann["low_freq"]
    highs = ann["high_freq"]
    codes = ann["ebird_code"]
    duration = ann["segment_end"] - ann["segment_start"]
    n_events = len(codes)
    unique_species = sorted(set(codes))
    species_color = {c: SPECIES_PALETTE[i % len(SPECIES_PALETTE)] for i, c in enumerate(unique_species)}
    boundaries, counts, max_overlap = simultaneous_call_counts(starts, ends, duration)
    min_polyphony = ann["min_polyphony"]
    max_polyphony = ann["max_polyphony"]
    assert min_polyphony == max(max_overlap, len(unique_species))
    assert max_polyphony == n_events
    return dict(starts=starts, ends=ends, lows=lows, highs=highs, codes=codes, duration=duration,
                n_events=n_events, species_color=species_color, boundaries=boundaries, counts=counts,
                max_overlap=max_overlap, n_unique_species=len(unique_species),
                min_polyphony=min_polyphony, max_polyphony=max_polyphony)


def build_figure(y, sr, ann, font_size=FONT_SIZE):
    """No in-figure heading -- the caption already describes the figure, so a
    heading would just duplicate it. That also frees up the top margin,
    which goes to panel (b): with only 4 "call N" y-tick labels it looks like
    the panel needing the least room, but at 8pt each label still needs
    >=~0.17in of vertical pitch to avoid touching its neighbors, so its
    height allocation isn't actually negotiable below that -- verified with
    the get_window_extent()-based overlap check below, not just by eye."""
    _set_rcparams(font_size)
    f = _example_fields(ann)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    # (a)/(b)/(c) share left/right margins wide enough for the rotated y-axis
    # label + tick digits (left) and the colorbar (right); anything left
    # overflowing the canvas gets picked up by bbox_inches="tight" in
    # save_fig, silently growing the saved figure past the intended column
    # width -- panel titles must also stay short (loc="left" titles start at
    # the axes edge and aren't wrapped). (d) doesn't need the colorbar's
    # margin at all, so it gets its own, much wider axes below the gridspec
    # rather than sharing (a)/(b)/(c)'s narrow one -- reusing that margin for
    # (d) leaves barely half the figure's actual width for its text.
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.75, 1.0],
                           width_ratios=[1, 0.06], hspace=0.30, wspace=0.08,
                           left=0.20, right=0.74, top=0.968, bottom=0.347)
    ax_a = fig.add_subplot(gs[0, 0])
    cax_a = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)
    ax_d = fig.add_axes([0.04, 0.024, 0.92, 0.234])

    _draw_panel_a(ax_a, cax_a, y, sr, f["starts"], f["ends"], f["lows"], f["highs"], f["codes"],
                  f["species_color"], font_size, show_xlabel=False)
    _draw_panel_b(ax_b, f["starts"], f["ends"], f["codes"], f["species_color"], f["n_events"], f["duration"],
                  font_size, show_xlabel=False)
    _draw_panel_c(ax_c, f["boundaries"], f["counts"], f["max_overlap"], f["n_unique_species"],
                  f["duration"], font_size)
    _draw_panel_d(ax_d, f["max_overlap"], f["n_unique_species"], f["min_polyphony"], f["max_polyphony"],
                  font_size)

    return fig


def build_figure_grid(y, sr, ann, font_size=FONT_SIZE):
    """2x2 layout spanning both columns of the target document
    (TWO_COL_WIDTH_IN): (a) top-left, (b) top-right, (c) bottom-left,
    (d) bottom-right. (a) and (c) share a column (and the time axis, via
    sharex) the same way they're stacked in build_figure; (b) is alone in
    its column so it keeps its own x-axis labels instead of relying on a
    time-series panel below it. Each panel's cell ends up close in width to
    build_figure's single column, so the panel-drawing code -- legend
    placement, label wrapping -- is identical between the two figures."""
    _set_rcparams(font_size)
    f = _example_fields(ann)

    fig = plt.figure(figsize=(TWO_COL_WIDTH_IN, 4.5))
    outer_gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1], height_ratios=[1.05, 1],
                                 wspace=0.32, hspace=0.42, left=0.078, right=0.98, top=0.95, bottom=0.11)
    left_gs = outer_gs[:, 0].subgridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1, 0.05],
                                          hspace=0.30, wspace=0.06)
    ax_a = fig.add_subplot(left_gs[0, 0])
    cax_a = fig.add_subplot(left_gs[0, 1])
    ax_c = fig.add_subplot(left_gs[1, 0], sharex=ax_a)
    ax_b = fig.add_subplot(outer_gs[0, 1])
    ax_d = fig.add_subplot(outer_gs[1, 1])

    _draw_panel_a(ax_a, cax_a, y, sr, f["starts"], f["ends"], f["lows"], f["highs"], f["codes"],
                  f["species_color"], font_size, show_xlabel=False)
    _draw_panel_b(ax_b, f["starts"], f["ends"], f["codes"], f["species_color"], f["n_events"], f["duration"],
                  font_size, show_xlabel=True)
    _draw_panel_c(ax_c, f["boundaries"], f["counts"], f["max_overlap"], f["n_unique_species"],
                  f["duration"], font_size)
    _draw_panel_d(ax_d, f["max_overlap"], f["n_unique_species"], f["min_polyphony"], f["max_polyphony"],
                  font_size)

    return fig


# (audio file, annotations file, output stem) -- two contrasting PolyBirdMix
# POW_soundscape_test rows: one where the min bound is set by unique species
# count (overlap < species), one where it's set by true temporal overlap
# (overlap > species), exercising both branches of min = max(overlap, species).
EXAMPLES = [
    ("example.wav", "annotations.json", "polyphony_range"),
    ("example_overlap.wav", "annotations_overlap.json", "polyphony_range_overlap"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("plots/data/polyphony_range"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/polyphony_range"))
    args = parser.parse_args()

    for audio_file, annotations_file, out_stem in EXAMPLES:
        y, sr = librosa.load(args.data_dir / audio_file, sr=None, mono=True)
        ann = json.loads((args.data_dir / annotations_file).read_text())

        fig = build_figure(y, sr, ann)
        save_fig(fig, args.out_dir / out_stem)
        plt.close(fig)

        fig_grid = build_figure_grid(y, sr, ann)
        save_fig(fig_grid, args.out_dir / f"{out_stem}_grid")
        plt.close(fig_grid)

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
