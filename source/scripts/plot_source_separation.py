#!/usr/bin/env python3
"""
Source-separation example figure: input mixture mel-spectrogram alongside
its separated sources, sharing one dB color scale.

Cleaned-up, script-form version of plots/data/source_separation/
plot_source_separation.ipynb's most-recent (mel-spectrogram, "2x2 sources")
layout, using the already-cropped example clips shipped in that same data
folder -- reused as-is rather than re-deriving the crop. Follows the same
conventions as the RQ1-RQ5 scripts: plain argparse CLI, shared plot_style
helpers, output under plots/figures/.

Usage:
    complete-venv/bin/python source/scripts/plot_source_separation.py \\
        [--data-dir plots/data/source_separation] [--out-dir plots/figures/source_separation]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import librosa
import librosa.display
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_style import save_fig, FIGURE_FONT_PT, FIGURE_SERIF_FONTS, TWO_COL_WIDTH_IN


def _whole_second_ticks(ax):
    """Restrict the time axis to whole-second ticks -- specshow's default
    locator often lands on 0.5 s steps, which looks cluttered at this panel
    width."""
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

# All three layouts below are sized to TWO_COL_WIDTH_IN (see plot_style.py):
# each shows 5 full spectrograms (input + 4 sources), too many to stay
# legible in a single narrow column regardless of arrangement.
FIG_WIDTH_IN = TWO_COL_WIDTH_IN
FIG_HEIGHT_IN = 2.0
FONT_SIZE = FIGURE_FONT_PT
N_FFT = 2048
N_MELS = 256
FMAX = 8000
SHADING = "gouraud"
PANEL_ASPECT = 0.85
INPUT_WIDTH_RATIO = 1.15


def build_figure(input_path: Path, source_paths: list[Path], source_labels: list[str], input_label: str):
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": FIGURE_SERIF_FONTS,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
    })

    y_input, sr = librosa.load(input_path, sr=None, mono=True)
    ys_sources = []
    for p in source_paths:
        y_s, sr_s = librosa.load(p, sr=sr, mono=True)
        if sr_s != sr:
            raise ValueError(f"Sample rate mismatch: {input_path} is {sr} Hz, {p} is {sr_s} Hz.")
        ys_sources.append(y_s)

    hop_length = N_FFT // 8
    all_waveforms = [y_input] + ys_sources
    all_mel_specs = [
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=hop_length, n_mels=N_MELS, fmax=FMAX)
        for y in all_waveforms
    ]
    ref = max(S.max() for S in all_mel_specs)
    all_specs_db = [librosa.power_to_db(S, ref=ref) for S in all_mel_specs]
    S_input_db, S_sources_db = all_specs_db[0], all_specs_db[1:]
    vmin, vmax = -80.0, 0.0

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[INPUT_WIDTH_RATIO, 2], wspace=0.28,
                                 left=0.08, right=0.90, bottom=0.13)
    ax_input = fig.add_subplot(outer_gs[0, 0])
    inner_gs = outer_gs[0, 1].subgridspec(2, 2, wspace=0.08, hspace=0.35)
    axes_sources = [fig.add_subplot(inner_gs[0, 0]), fig.add_subplot(inner_gs[0, 1]),
                     fig.add_subplot(inner_gs[1, 0]), fig.add_subplot(inner_gs[1, 1])]

    librosa.display.specshow(S_input_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel",
                              fmax=FMAX, vmin=vmin, vmax=vmax, ax=ax_input, rasterized=True, shading=SHADING)
    _whole_second_ticks(ax_input)
    ax_input.set_title(input_label, loc="left", fontsize=FONT_SIZE)
    ax_input.set_xlabel("Time [s]")
    ax_input.set_ylabel("Mel frequency [Hz]")
    ax_input.set_box_aspect(PANEL_ASPECT)

    img = None
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for ax, S_db, label, (row, col) in zip(axes_sources, S_sources_db, source_labels, positions):
        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time",
                                        y_axis="mel" if col == 0 else None, fmax=FMAX,
                                        vmin=vmin, vmax=vmax, ax=ax, rasterized=True, shading=SHADING)
        _whole_second_ticks(ax)
        ax.set_title(label, loc="left", fontsize=FONT_SIZE - 0.5)
        if row == 1:
            ax.set_xlabel("Time [s]")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel("Mel frequency [Hz]")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        ax.set_box_aspect(PANEL_ASPECT)

    fig.colorbar(img, ax=[ax_input] + axes_sources, format="%+2.0f dB", label="dB", fraction=0.02, pad=0.02)
    return fig


def build_figure_row(input_path: Path, source_paths: list[Path], source_labels: list[str], input_label: str):
    """Alternative layout: all panels side by side in a single row (input +
    4 sources = 5 spectrogram columns, plus a 6th dedicated colorbar column),
    instead of the input | 2x2-sources split."""
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": FIGURE_SERIF_FONTS,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
    })

    y_input, sr = librosa.load(input_path, sr=None, mono=True)
    ys_sources = []
    for p in source_paths:
        y_s, sr_s = librosa.load(p, sr=sr, mono=True)
        if sr_s != sr:
            raise ValueError(f"Sample rate mismatch: {input_path} is {sr} Hz, {p} is {sr_s} Hz.")
        ys_sources.append(y_s)

    hop_length = N_FFT // 8
    all_waveforms = [y_input] + ys_sources
    all_mel_specs = [
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=hop_length, n_mels=N_MELS, fmax=FMAX)
        for y in all_waveforms
    ]
    ref = max(S.max() for S in all_mel_specs)
    all_specs_db = [librosa.power_to_db(S, ref=ref) for S in all_mel_specs]
    vmin, vmax = -80.0, 0.0
    all_labels = [input_label] + list(source_labels)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 0.6))
    # 5 spectrogram columns + 1 narrow dedicated colorbar column.
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.06], wspace=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    cax = fig.add_subplot(gs[0, 5])

    img = None
    for i, (ax, S_db, label) in enumerate(zip(axes, all_specs_db, all_labels)):
        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time",
                                        y_axis="mel" if i == 0 else None, fmax=FMAX,
                                        vmin=vmin, vmax=vmax, ax=ax, rasterized=True, shading=SHADING)
        _whole_second_ticks(ax)
        ax.set_title(label, loc="left", fontsize=FONT_SIZE - 0.5)
        ax.set_xlabel("Time [s]")
        if i == 0:
            ax.set_ylabel("Mel frequency [Hz]")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        ax.set_box_aspect(PANEL_ASPECT)

    fig.colorbar(img, cax=cax, format="%+2.0f dB", label="dB")
    return fig


def build_figure_stacked(input_path: Path, source_paths: list[Path], source_labels: list[str], input_label: str):
    """Alternative layout: input sits on its own row on top, centered and
    the same size as the source panels, the 4 sources sit in a single row
    below it, one shared colorbar spanning both rows on the right."""
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": FIGURE_SERIF_FONTS,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
    })

    y_input, sr = librosa.load(input_path, sr=None, mono=True)
    ys_sources = []
    for p in source_paths:
        y_s, sr_s = librosa.load(p, sr=sr, mono=True)
        if sr_s != sr:
            raise ValueError(f"Sample rate mismatch: {input_path} is {sr} Hz, {p} is {sr_s} Hz.")
        ys_sources.append(y_s)

    hop_length = N_FFT // 8
    all_waveforms = [y_input] + ys_sources
    all_mel_specs = [
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=hop_length, n_mels=N_MELS, fmax=FMAX)
        for y in all_waveforms
    ]
    ref = max(S.max() for S in all_mel_specs)
    all_specs_db = [librosa.power_to_db(S, ref=ref) for S in all_mel_specs]
    S_input_db, S_sources_db = all_specs_db[0], all_specs_db[1:]
    vmin, vmax = -80.0, 0.0

    # Manual axes placement (figure-fraction rectangles), rather than a
    # gridspec + constrained_layout: constrained_layout negotiates each
    # subfigure's margins independently based on its own tick/label sizes,
    # which -- even with identical column ratios -- ended up giving the
    # (single-axes) input row noticeably wider columns than the (four-axes)
    # source row. Explicit rectangles with an identical (panel_w, panel_h)
    # guarantee Input comes out pixel-identical in size to each Source panel.
    # Scaled to make fig_w come out to TWO_COL_WIDTH_IN: these were tuned at
    # a 9in design width, and shrinking a fixed inch-width figure needs every
    # dimension in it scaled together, not just the outer figsize.
    _scale = TWO_COL_WIDTH_IN / 9.26
    panel_w, panel_h = 1.5 * _scale, 1.5 * PANEL_ASPECT * _scale
    gap_x, gap_y = 0.28 * _scale, 1.05 * _scale
    left_margin, bottom_margin, top_margin = 0.95 * _scale, 0.55 * _scale, 0.3 * _scale
    cbar_gap, cbar_w, right_margin = 0.3 * _scale, 0.22 * _scale, 0.95 * _scale

    sources_w = 4 * panel_w + 3 * gap_x
    fig_w = left_margin + sources_w + cbar_gap + cbar_w + right_margin
    fig_h = bottom_margin + panel_h + gap_y + panel_h + top_margin

    fig = plt.figure(figsize=(fig_w, fig_h))

    input_x0 = left_margin + (sources_w - panel_w) / 2
    input_y0 = bottom_margin + panel_h + gap_y
    ax_input = fig.add_axes([input_x0 / fig_w, input_y0 / fig_h, panel_w / fig_w, panel_h / fig_h])

    axes_sources = []
    for i in range(4):
        x0 = left_margin + i * (panel_w + gap_x)
        axes_sources.append(fig.add_axes([x0 / fig_w, bottom_margin / fig_h, panel_w / fig_w, panel_h / fig_h]))

    cax_x0 = left_margin + sources_w + cbar_gap
    cax = fig.add_axes([cax_x0 / fig_w, bottom_margin / fig_h, cbar_w / fig_w, panel_h / fig_h])

    librosa.display.specshow(S_input_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel",
                              fmax=FMAX, vmin=vmin, vmax=vmax, ax=ax_input, rasterized=True, shading=SHADING)
    _whole_second_ticks(ax_input)
    ax_input.set_title(input_label, loc="left", fontsize=FONT_SIZE)
    ax_input.set_xlabel("Time [s]")
    ax_input.set_ylabel("Mel frequency [Hz]")
    ax_input.set_box_aspect(PANEL_ASPECT)

    img = None
    for i, (ax, S_db, label) in enumerate(zip(axes_sources, S_sources_db, source_labels)):
        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time",
                                        y_axis="mel" if i == 0 else None, fmax=FMAX,
                                        vmin=vmin, vmax=vmax, ax=ax, rasterized=True, shading=SHADING)
        _whole_second_ticks(ax)
        ax.set_title(label, loc="left", fontsize=FONT_SIZE - 0.5)
        ax.set_xlabel("Time [s]")
        if i == 0:
            ax.set_ylabel("Mel frequency [Hz]")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        ax.set_box_aspect(PANEL_ASPECT)

    fig.colorbar(img, cax=cax, format="%+2.0f dB", label="dB")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("plots/data/source_separation"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/source_separation"))
    args = parser.parse_args()

    input_path = args.data_dir / "input_cropped.wav"
    source_paths = [args.data_dir / f"source_{i}_cropped.wav" for i in range(4)]
    source_labels = [f"Source {i + 1}" for i in range(4)]

    fig = build_figure(input_path, source_paths, source_labels, input_label="Input")
    save_fig(fig, args.out_dir / "source_separation")
    plt.close(fig)

    fig_row = build_figure_row(input_path, source_paths, source_labels, input_label="Input")
    save_fig(fig_row, args.out_dir / "source_separation_row")
    plt.close(fig_row)

    fig_stacked = build_figure_stacked(input_path, source_paths, source_labels, input_label="Input")
    save_fig(fig_stacked, args.out_dir / "source_separation_stacked")
    plt.close(fig_stacked)

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
