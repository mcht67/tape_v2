#!/usr/bin/env python3
"""
Time-frequency masking pipeline figure: input spectrogram with detected
event regions and frequency bounds, the RMS energy envelope used for event
detection, the resulting soft time-frequency mask, and the masked output
spectrogram.

Cleaned-up, script-form version of plots/data/time_freq_masking/
plot_masking_procedure.ipynb's 2x2 layout, using an example clip shipped in
that same data folder. Follows the same conventions as
plot_source_separation.py: plain argparse CLI, shared plot_style helpers,
output under plots/figures/.

Usage:
    complete-venv/bin/python source/scripts/plot_masking_procedure.py \\
        [--data-dir plots/data/time_freq_masking] [--out-dir plots/figures/time_freq_masking]
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import librosa
import librosa.display
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> source/

from plot_style import save_fig, FIGURE_FONT_PT, FIGURE_SERIF_FONTS, TWO_COL_WIDTH_IN
from utils.dsp import duration_s_to_num_samples, num_samples_to_duration_s

# Both layouts below are sized to TWO_COL_WIDTH_IN (see plot_style.py): each
# panel is a full spectrogram/mask needing real width to stay legible, so
# neither the 2x2 grid nor the single-row layout fits a single narrow column.
FIG_WIDTH_IN = TWO_COL_WIDTH_IN
FIG_HEIGHT_IN = 5.1
FONT_SIZE = FIGURE_FONT_PT
N_FFT = 1024
FREQ_LOG_SCALE = True
DYNAMIC_RANGE_DB = 80.0
SHADING = "gouraud"
# Tuned in build_figure_row() so the legend's bottom edge lands exactly on
# the colorbars' bottom edge, with no leftover blank margin above the
# legend -- see that function's own geometry comment for the derivation.
FIG_HEIGHT_ROW = 1.89

REGION_COLOR = "#2ca02c"
REGION_COLOR_RAW = "#a8a8a8"
THRESHOLD_COLOR = "#d62728"


@dataclass
class EventDetectionResult:
    """Diagnostics from detect_event_bounds, kept around for plotting."""
    events: List[Tuple[float, float]]
    raw_regions: List[Tuple[float, float]]
    energy_times: np.ndarray
    energy_envelope: np.ndarray
    energy_threshold: float


@dataclass
class MaskingResult:
    """Diagnostics from stft_mask_bandpass, kept around for plotting."""
    y_out: np.ndarray
    bounds_list: List[Tuple[float, float, float, float]]
    mask: np.ndarray
    n_fft: int
    hop_length: int


def detect_event_bounds(y, sr, smooth_ms=25, threshold_ratio=0.1, min_gap_ms=20,
                         min_call_ms=10, rms_frame_length_ms=20, rms_hop_length_ms=5):
    """Detect bird-call regions (onset, offset) via RMS thresholding, merge
    bursts separated by short gaps, and discard regions shorter than
    min_call_ms."""
    hop_length = int(rms_hop_length_ms * 0.001 * sr)
    frame_length = int(rms_frame_length_ms * 0.001 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    win = int(smooth_ms / rms_hop_length_ms) or 1
    smooth_rms = np.convolve(rms, np.ones(win) / win, mode="same")

    thresh = threshold_ratio * np.max(smooth_rms)
    active_mask = smooth_rms > thresh

    diff = np.diff(active_mask.astype(int))
    onsets = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1
    if active_mask[0]:
        onsets = np.r_[0, onsets]
    if active_mask[-1]:
        offsets = np.r_[offsets, len(active_mask)]
    onsets = np.clip(onsets, 0, len(times) - 1)
    offsets = np.clip(offsets, 0, len(times) - 1)

    raw_regions = [(float(times[o]), float(times[f])) for o, f in zip(onsets, offsets)]

    if len(onsets) == 0:
        events = []
    else:
        min_gap_s = min_gap_ms * 0.001
        min_call_s = min_call_ms * 0.001
        merged_onsets = [onsets[0]]
        merged_offsets = []
        for i in range(1, len(onsets)):
            if times[onsets[i]] - times[offsets[i - 1]] <= min_gap_s:
                continue
            merged_offsets.append(offsets[i - 1])
            merged_onsets.append(onsets[i])
        merged_offsets.append(offsets[-1])
        events = [(float(times[o]), float(times[f]))
                  for o, f in zip(merged_onsets, merged_offsets)
                  if (times[f] - times[o]) >= min_call_s]

    return EventDetectionResult(
        events=events,
        raw_regions=raw_regions,
        energy_times=times,
        energy_envelope=smooth_rms,
        energy_threshold=float(thresh),
    )


def stft_mask_bandpass(y, sr, n_fft=N_FFT, hop_length=None, collapse="max", smooth_bins=5,
                        low_pct=2, high_pct=98, edge_bins=5, edge_ms=30,
                        events: Optional[List[Tuple[float, float]]] = None):
    """Remove energy outside percentile-based frequency bands (per event
    region), with soft ramps at both the frequency edges and the region's
    time edges."""
    if hop_length is None:
        hop_length = n_fft // 4
    edge_frames = int(round((edge_ms / 1000.0) * sr / hop_length))

    stft_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    magnitudes = np.abs(stft_complex)
    phases = np.angle(stft_complex)
    freq_bin_centers_hz = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_freq_bins, n_time_frames = magnitudes.shape

    if events is None:
        segments_samples = [(0, len(y))]
    else:
        segments_samples = [(duration_s_to_num_samples(onset, sr),
                              duration_s_to_num_samples(offset, sr)) for (onset, offset) in events]

    def ramp_profile(profile_length, plateau_start_idx, plateau_end_idx, edge_width):
        """1D profile: flat 1.0 on [plateau_start_idx, plateau_end_idx], linear
        ramp extending edge_width steps outward on each side, 0 elsewhere."""
        profile = np.zeros(profile_length)
        clipped_start_idx = max(0, plateau_start_idx)
        clipped_end_idx = min(profile_length - 1, plateau_end_idx)
        if clipped_start_idx <= clipped_end_idx:
            profile[clipped_start_idx:clipped_end_idx + 1] = 1.0
        for step in range(1, edge_width + 1):
            ramp_weight = (edge_width - (step - 1)) / float(edge_width + 1)
            left_idx = plateau_start_idx - step
            right_idx = plateau_end_idx + step
            if left_idx >= 0:
                profile[left_idx] = max(profile[left_idx], ramp_weight)
            if right_idx < profile_length:
                profile[right_idx] = max(profile[right_idx], ramp_weight)
        return profile

    combined_mask = np.zeros_like(magnitudes)
    bounds_list = []
    for (segment_start_sample, segment_end_sample) in segments_samples:
        frame_start_idx = max(0, int(np.floor(segment_start_sample / float(hop_length))))
        frame_end_idx = min(n_time_frames, int(np.ceil(segment_end_sample / float(hop_length))))
        if frame_start_idx >= frame_end_idx:
            continue

        segment_block = magnitudes[:, frame_start_idx:frame_end_idx]
        if collapse == "max":
            collapsed_spectrum = np.max(segment_block, axis=1)
        elif collapse == "median":
            collapsed_spectrum = np.median(segment_block, axis=1)
        else:
            collapsed_spectrum = np.mean(segment_block, axis=1)
        if smooth_bins and smooth_bins > 1:
            collapsed_spectrum = uniform_filter1d(collapsed_spectrum, size=smooth_bins)
        collapsed_spectrum = np.maximum(collapsed_spectrum, 0.0)

        spectrum_total = collapsed_spectrum.sum()
        if spectrum_total <= 0:
            continue
        cumulative_energy = np.cumsum(collapsed_spectrum / spectrum_total)
        freq_low_hz = np.interp(low_pct / 100.0, cumulative_energy, freq_bin_centers_hz)
        freq_high_hz = np.interp(high_pct / 100.0, cumulative_energy, freq_bin_centers_hz)
        bounds_list.append((num_samples_to_duration_s(segment_start_sample, sr),
                             num_samples_to_duration_s(segment_end_sample, sr),
                             float(freq_low_hz), float(freq_high_hz)))

        freq_low_bin_idx = int(np.searchsorted(freq_bin_centers_hz, freq_low_hz))
        freq_high_bin_idx = int(np.searchsorted(freq_bin_centers_hz, freq_high_hz))
        freq_profile = ramp_profile(n_freq_bins, freq_low_bin_idx, freq_high_bin_idx, edge_bins)
        full_time_profile = ramp_profile(n_time_frames, frame_start_idx, frame_end_idx - 1, edge_frames)

        window_start_idx = max(0, frame_start_idx - edge_frames)
        window_end_idx = min(n_time_frames, frame_end_idx + edge_frames)
        if window_start_idx >= window_end_idx:
            continue
        segment_mask = np.outer(freq_profile, full_time_profile[window_start_idx:window_end_idx])
        combined_mask[:, window_start_idx:window_end_idx] = np.maximum(
            combined_mask[:, window_start_idx:window_end_idx], segment_mask
        )

    stft_masked = combined_mask * magnitudes * np.exp(1j * phases)
    y_out = librosa.istft(stft_masked, hop_length=hop_length, win_length=n_fft, length=len(y))

    return MaskingResult(y_out=y_out, bounds_list=bounds_list, mask=combined_mask,
                          n_fft=n_fft, hop_length=hop_length)


def _set_rcparams():
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": FIGURE_SERIF_FONTS,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
    })


def _spectrogram_panel(ax, S_db, sr, hop_length, vmin, vmax, label, show_ylabel=True):
    y_axis_type = "log" if FREQ_LOG_SCALE else "hz"
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time",
                                    y_axis=y_axis_type, cmap="magma", vmin=vmin, vmax=vmax,
                                    ax=ax, rasterized=True, shading=SHADING)
    ax.set_title(label, loc="left")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]" if show_ylabel else "")
    return img


def build_figure(y, sr, y_out, event_result: EventDetectionResult, mask_result: MaskingResult,
                  labels=("(a)", "(b)", "(c)", "(d)")):
    """2x2 grid layout: (a) input spectrogram + frequency bounds, (b) energy
    envelope + threshold + region cleanup, (c) the mask, (d) output
    spectrogram -- with dedicated colorbar columns for the dB and mask-value
    scales."""
    _set_rcparams()
    n_fft, hop_length = mask_result.n_fft, mask_result.hop_length

    S_in = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_out = np.abs(librosa.stft(y_out, n_fft=n_fft, hop_length=hop_length))
    ref = S_in.max()
    S_in_db = librosa.amplitude_to_db(S_in, ref=ref)
    S_out_db = librosa.amplitude_to_db(S_out, ref=ref)
    vmin, vmax = -DYNAMIC_RANGE_DB, 0.0

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.4, hspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    cax_db = fig.add_subplot(gs[0, 2])
    cax_mask = fig.add_subplot(gs[1, 2])

    # (a) input spectrogram + frequency bounds
    _spectrogram_panel(ax_a, S_in_db, sr, hop_length, vmin, vmax, labels[0])
    for (start_s, end_s, f_low, f_high) in mask_result.bounds_list:
        ax_a.add_patch(patches.Rectangle((start_s, f_low), end_s - start_s, f_high - f_low,
                                          fill=False, edgecolor=REGION_COLOR, linestyle="--",
                                          linewidth=1.2))

    # (b) energy envelope + threshold + region cleanup
    for i, (s, e) in enumerate(event_result.raw_regions):
        ax_b.axvspan(s, e, color=REGION_COLOR_RAW, alpha=0.35,
                     label="candidate region" if i == 0 else None)
    for i, (s, e) in enumerate(event_result.events):
        ax_b.axvspan(s, e, color=REGION_COLOR, alpha=0.35,
                     label="active region" if i == 0 else None)
    ax_b.plot(event_result.energy_times, event_result.energy_envelope, color="black",
              linewidth=1, label="RMS envelope")
    ax_b.axhline(event_result.energy_threshold, color=THRESHOLD_COLOR, linestyle="--",
                 linewidth=1.1, label="energy threshold")
    ax_b.set_title(labels[1], loc="left")
    ax_b.set_xlabel("Time [s]")
    ax_b.set_ylabel("RMS energy")
    ax_b.set_xlim(event_result.energy_times[0], event_result.energy_times[-1])
    ax_b.legend(loc="upper right", framealpha=0.9, handlelength=1.5)

    # (c) the mask
    freq_bin_centers_hz = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask_times = librosa.frames_to_time(np.arange(mask_result.mask.shape[1]), sr=sr,
                                         hop_length=hop_length)
    mesh = ax_c.pcolormesh(mask_times, freq_bin_centers_hz, mask_result.mask, cmap="viridis",
                            vmin=0.0, vmax=1.0, shading=SHADING, rasterized=True)
    if FREQ_LOG_SCALE:
        ax_c.set_yscale("log")
        ax_c.set_ylim(max(freq_bin_centers_hz[1], 20), freq_bin_centers_hz[-1])
    ax_c.set_title(labels[2], loc="left")
    ax_c.set_xlabel("Time [s]")
    ax_c.set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh, cax=cax_mask, label="mask value")

    # (d) output spectrogram
    img_d = _spectrogram_panel(ax_d, S_out_db, sr, hop_length, vmin, vmax, labels[3])
    fig.colorbar(img_d, cax=cax_db, format="%+2.0f dB", label="dB")

    return fig


def build_figure_row(y, sr, y_out, event_result: EventDetectionResult, mask_result: MaskingResult,
                      labels=("(a)", "(b)", "(c)", "(d)")):
    """Single-row layout, left to right: (a) energy envelope + threshold +
    region cleanup, (b) input spectrogram + bounds, (c) the mask, (d) output
    spectrogram -- (a) and (b) swapped from their build_figure() order so
    the three frequency-axis panels (b, c, d) sit contiguously: only (b),
    the first of the three, draws the "Frequency [Hz]" label + y-ticks, and
    (c)/(d) hide theirs, which is only possible because they're now adjacent
    to the one panel that does show it. That frees the width (b)-(c) and
    (c)-(d) would otherwise need for their own axis labels, so those two
    gaps are set much tighter than the (a)-(b) gap (which still needs full
    room -- (a) has its own distinct "RMS energy" label, and (b) still needs
    space for its now-first-shown "Frequency [Hz]" label). That asymmetry
    isn't expressible with a GridSpec's single uniform wspace, so (a) and
    (b)/(c)/(d) are laid out as two side-by-side GridSpecs instead.

    Colorbars sit above the panels rather than below (a horizontal colorbar
    below the panels sits right next to each panel's own "Time [s]" xlabel
    and reads as if it, too, were labeling the time axis) -- the dB one
    aligned exactly to (b)'s own left/right edges, the mask-value one to
    (c)'s, each read off the axes' actual rendered position rather than a
    hand-picked split point. No "dB"/"mask value" heading on either (the
    tick format/values already carry the unit), which also lets them run
    thinner than a labeled colorbar would need. The (a, b, c)-legend sits
    above (a), left-aligned to its own left edge and spanning 2 rows now
    that it doesn't have to share a row with the colorbars."""
    _set_rcparams()
    n_fft, hop_length = mask_result.n_fft, mask_result.hop_length

    S_in = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_out = np.abs(librosa.stft(y_out, n_fft=n_fft, hop_length=hop_length))
    ref = S_in.max()
    S_in_db = librosa.amplitude_to_db(S_in, ref=ref)
    S_out_db = librosa.amplitude_to_db(S_out, ref=ref)
    vmin, vmax = -DYNAMIC_RANGE_DB, 0.0

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_ROW))
    # panel_top/bottom, cbar_gap/cbar_height, and legend_anchor were all
    # derived in absolute inches, then converted to FIG_HEIGHT_ROW fractions
    # (which is why FIG_HEIGHT_ROW isn't a round number) -- panel_top/bottom
    # sets the panel row's own height (too short and the log-frequency
    # y-ticks on (b) start colliding); cbar_gap is the minimum clearance the
    # (b)/(c) titles need below the colorbars; legend_anchor is set so the
    # legend's own bottom edge (its box bottom, not the anchor point --
    # matplotlib offsets the two by ~0.056in of internal padding) lands
    # exactly on the colorbars' bottom edge (panel_top + cbar_gap), with
    # FIG_HEIGHT_ROW trimmed down to the legend's top edge plus a small
    # 0.03in margin -- i.e. no blank canvas above the legend. This is the
    # tightest combination found with 0 real overlaps (a sub-0.02in
    # canvas-bbox rounding overflow remains, invisible in the saved PNG).
    panel_top, bottom = 0.6561, 0.1905
    cbar_gap, cbar_height = 0.1164, 0.0635
    legend_anchor = 0.9841
    # a-b gap widened (was left=0.275/0.355, a 0.08 gap) so the "Frequency
    # [Hz]" ylabel -- now the leftmost text in the b/c/d group -- sits a
    # little further from panel (a) instead of crowding its right edge.
    gs_a = fig.add_gridspec(1, 1, left=0.085, right=0.265, top=panel_top, bottom=bottom)
    gs_bcd = fig.add_gridspec(1, 3, left=0.385, right=0.995, top=panel_top, bottom=bottom, wspace=0.14)
    ax_a = fig.add_subplot(gs_a[0, 0])
    ax_b = fig.add_subplot(gs_bcd[0, 0])
    ax_c = fig.add_subplot(gs_bcd[0, 1])
    ax_d = fig.add_subplot(gs_bcd[0, 2])

    # Legend: 2 rows (was 1, spanning nearly the full figure width) now that
    # it only has to sit above (a), left-aligned to (a)'s own left edge
    # (x=0.085, matching gs_a's left) instead of the figure's. handlelength/
    # columnspacing trimmed slightly so its 2nd row doesn't run wide enough
    # to reach into where the dB colorbar starts (see cax_db below).
    legend_handles = [
        mlines.Line2D([], [], color="black", linewidth=0.9, label="RMS envelope"),
        mlines.Line2D([], [], color=THRESHOLD_COLOR, linestyle="--", linewidth=1.0, label="threshold"),
        patches.Patch(color=REGION_COLOR, alpha=0.35, label="active region"),
    ]
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.085, legend_anchor),
               ncol=2, frameon=False, handlelength=1.2, columnspacing=0.7)

    # Colorbars aligned exactly to (b)'s and (c)'s own left/right edges --
    # read from the axes' actual rendered position (get_position()) rather
    # than recomputing the GridSpec's column-width formula by hand.
    fig.canvas.draw()
    pos_b, pos_c = ax_b.get_position(), ax_c.get_position()
    cbar_bottom = panel_top + cbar_gap
    cax_db = fig.add_axes([pos_b.x0, cbar_bottom, pos_b.width, cbar_height])
    cax_mask = fig.add_axes([pos_c.x0, cbar_bottom, pos_c.width, cbar_height])

    # (a) energy envelope + threshold + region cleanup
    for (s, e) in event_result.events:
        ax_a.axvspan(s, e, color=REGION_COLOR, alpha=0.35)
    ax_a.plot(event_result.energy_times, event_result.energy_envelope, color="black", linewidth=0.9)
    ax_a.axhline(event_result.energy_threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.0)
    ax_a.set_title(labels[0], loc="left")
    ax_a.set_xlabel("Time [s]")
    ax_a.set_ylabel("RMS energy")
    ax_a.set_xlim(event_result.energy_times[0], event_result.energy_times[-1])

    # (b) input spectrogram + frequency bounds -- first of the 3 contiguous
    # frequency-axis panels, so the only one that shows the axis label/ticks.
    _spectrogram_panel(ax_b, S_in_db, sr, hop_length, vmin, vmax, labels[1])
    for (start_s, end_s, f_low, f_high) in mask_result.bounds_list:
        ax_b.add_patch(patches.Rectangle((start_s, f_low), end_s - start_s, f_high - f_low,
                                          fill=False, edgecolor=REGION_COLOR, linestyle="--",
                                          linewidth=1.0))

    # (c) the mask -- specshow instead of a hand-rolled log scale, so its
    # frequency axis is pixel-identical to (b)/(d)'s; label/ticks hidden
    # since (b) already shows them and (c) sits right next to it.
    img_c = librosa.display.specshow(mask_result.mask, sr=sr, hop_length=hop_length, x_axis="time",
                                      y_axis="log", cmap="viridis", vmin=0.0, vmax=1.0,
                                      ax=ax_c, rasterized=True, shading=SHADING)
    ax_c.set_title(labels[2], loc="left")
    ax_c.set_xlabel("Time [s]")
    ax_c.set_ylabel("")
    ax_c.tick_params(labelleft=False)

    # (d) output spectrogram
    img_d = _spectrogram_panel(ax_d, S_out_db, sr, hop_length, vmin, vmax, labels[3], show_ylabel=False)
    ax_d.tick_params(labelleft=False)

    # Each panel is only ~1.1in wide here, too narrow for the default time
    # ticker's 0.5s spacing (11 labels) without them colliding -- set after
    # _spectrogram_panel()/specshow(), which install their own time-axis
    # locator and would otherwise override this.
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

    # No "dB"/"mask value" heading -- the tick format/values already carry
    # the unit (format="%+2.0f dB"), so a separate label would be redundant.
    # Ticks on top (toward the figure edge, away from the panels) since that
    # heading no longer needs the space above them.
    fig.colorbar(img_d, cax=cax_db, orientation="horizontal", format="%+2.0f dB")
    cax_db.xaxis.set_ticks_position("top")
    # cax_db/cax_mask sit right next to each other (aligned to (b)/(c), which
    # are themselves close together) -- center-aligned edge tick labels would
    # overlap across that narrow gap, so the two touching ticks are pulled
    # back into their own colorbar instead.
    cax_db.get_xticklabels()[-1].set_ha("right")
    fig.colorbar(img_c, cax=cax_mask, orientation="horizontal")
    cax_mask.xaxis.set_ticks_position("top")
    cax_mask.get_xticklabels()[0].set_ha("left")

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("plots/data/time_freq_masking"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/time_freq_masking"))
    parser.add_argument("--audio", type=str, default="example.wav")
    args = parser.parse_args()

    y, sr = librosa.load(args.data_dir / args.audio, sr=None, mono=True)

    event_result = detect_event_bounds(y, sr)
    mask_result = stft_mask_bandpass(y, sr, events=event_result.events)

    fig = build_figure(y, sr, mask_result.y_out, event_result, mask_result)
    save_fig(fig, args.out_dir / "masking_procedure")
    plt.close(fig)

    fig_row = build_figure_row(y, sr, mask_result.y_out, event_result, mask_result)
    save_fig(fig_row, args.out_dir / "masking_procedure_row")
    plt.close(fig_row)

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
