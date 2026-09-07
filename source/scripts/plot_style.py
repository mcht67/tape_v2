"""Shared matplotlib styling/legend/save primitives for the RQ1-RQ5 plotting
scripts. Kept deliberately small -- each RQ script composes these rather than
calling raw matplotlib for the repeated bits (dot+whisker points, proxy-
artist legends, dashed zero-reference lines, PNG+PDF+tex saving)."""

from pathlib import Path

import matplotlib.lines as mlines
import numpy as np


def save_fig(fig, out_stem: Path, tex: str | None = None) -> None:
    """Write out_stem.png (dpi=200) + out_stem.pdf, and out_stem.tex if a
    LaTeX table string is given alongside the figure."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    if tex is not None:
        out_stem.with_suffix(".tex").write_text(tex)


def draw_dot_whisker(ax, x, mean, sd, *, facecolor, edgecolor, marker="o",
                      ecolor=None, capsize=3, s=80, filled=True, zorder=3,
                      orientation="vertical"):
    """Draw one dot+whisker point. Whisker and marker are drawn with
    separate calls (errorbar then scatter) so a hollow marker (paradigm =
    self-supervised) can be rendered independently of the whisker color --
    a single errorbar(marker=...) call can't give a clean hollow/filled
    marker independent of its own line color.

    `x` is always the categorical position and `mean` the value axis;
    orientation="horizontal" swaps them onto the plot's x/y axes (for
    horizontal dumbbell/dot plots where the value axis runs left-right)."""
    has_sd = sd is not None and not (isinstance(sd, float) and np.isnan(sd))
    if orientation == "vertical":
        if has_sd:
            ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=ecolor or edgecolor,
                        capsize=capsize, zorder=zorder - 1)
        ax.scatter([x], [mean], marker=marker,
                   facecolor=facecolor if filled else "none",
                   edgecolor=edgecolor, linewidth=1.5, s=s, zorder=zorder)
    else:
        if has_sd:
            ax.errorbar(mean, x, xerr=sd, fmt="none", ecolor=ecolor or edgecolor,
                        capsize=capsize, zorder=zorder - 1)
        ax.scatter([mean], [x], marker=marker,
                   facecolor=facecolor if filled else "none",
                   edgecolor=edgecolor, linewidth=1.5, s=s, zorder=zorder)


def build_encoding_legend(fig, legend_groups, loc="lower center", ncol=None,
                           bbox_to_anchor=(0.5, -0.05), entries_ncol=1):
    """Render one or more stacked mini-legends from proxy artists.

    legend_groups: list of (title, entries) where entries is a list of
    (label, marker_kwargs) -- marker_kwargs passed straight to
    matplotlib.lines.Line2D (e.g. dict(marker="o", color="k",
    markerfacecolor="k") for a filled circle, or markerfacecolor="none" for
    hollow). One ax.legend() can't cleanly decompose 3 independent visual
    encodings (color/shape/fill) into one legend, so each group gets its own
    titled legend, stacked left-to-right along the bottom of the figure.

    entries_ncol: number of columns to lay a given group's own entries out
    in (left-to-right, wrapping into that many rows) instead of a single
    stacked column -- either one int applied to every group, or a list with
    one value per group.
    """
    n = len(legend_groups)
    if ncol is None:
        ncol = n
    if not isinstance(entries_ncol, (list, tuple)):
        entries_ncol = [entries_ncol] * n
    legends = []
    for i, (title, entries) in enumerate(legend_groups):
        handles = [
            mlines.Line2D([], [], linestyle="none", markersize=9, **kwargs)
            for _, kwargs in entries
        ]
        labels = [label for label, _ in entries]
        x = (i + 0.5) / n
        leg = fig.legend(handles, labels, title=title, loc=loc,
                          bbox_to_anchor=(x, bbox_to_anchor[1]), ncol=entries_ncol[i], frameon=False)
        fig.add_artist(leg)
        legends.append(leg)
    return legends


def zero_ref_line(ax, axis="y", value=0.0):
    """Dashed neutral-gray reference line for a 'no change' baseline."""
    if axis == "y":
        ax.axhline(value, color="0.5", linestyle="--", linewidth=1, zorder=0)
    else:
        ax.axvline(value, color="0.5", linestyle="--", linewidth=1, zorder=0)


def mask_missing(matrix):
    """Return a masked array (NaN cells masked) for heatmaps with gaps, so
    they render as a distinct 'no data' color rather than silently as 0."""
    return np.ma.masked_invalid(matrix.to_numpy() if hasattr(matrix, "to_numpy") else matrix)
