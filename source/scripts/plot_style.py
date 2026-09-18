"""Shared matplotlib styling/legend/save primitives for the RQ1-RQ5 plotting
scripts. Kept deliberately small -- each RQ script composes these rather than
calling raw matplotlib for the repeated bits (dot+whisker points, proxy-
artist legends, dashed zero-reference lines, PNG+PDF+tex saving)."""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import numpy as np

# Page geometry of the target LaTeX document (two-column, \textwidth =
# 469.75502pt, \columnwidth = 226.34169pt, 10pt base font), measured via
# \the\textwidth / \the\columnwidth / \f@size. A matplotlib font size in
# points is only physically correct on the page if the figure is placed at
# its native size (no \includegraphics scale factor) -- so a figure meant
# to sit in one column must be authored at ONE_COL_WIDTH_IN, and one meant
# to span both columns at TWO_COL_WIDTH_IN, both using FIGURE_FONT_PT.
# Authoring every figure at one shared inch-width regardless of placement
# (the previous convention) makes \includegraphics shrink single- and
# double-column figures by different factors, so the "same" font size in
# points ends up rendering at different physical sizes across the document.
_PT_PER_IN = 72.27
ONE_COL_WIDTH_IN = 226.34169 / _PT_PER_IN
TWO_COL_WIDTH_IN = 469.75502 / _PT_PER_IN
BASE_FONT_PT = 10  # document body font size
FIGURE_FONT_PT = 9  # matches \small in the 10pt document, for figure/axis text

# The document's preamble sets \renewcommand{\rmdefault}{ptm} -- "ptm" is the
# PSNFSS code for (Adobe/URW) Times, so the body/caption font is Times, not
# LaTeX's default Computer Modern. matplotlib's own "serif" family falls back
# to its bundled DejaVu Serif, which has a noticeably larger x-height and
# heavier strokes than Times at the same nominal point size -- so even with
# FIGURE_FONT_PT exactly matching the document (verified: a figure saved at
# its target column width and placed via \includegraphics[width=...] comes
# out within ~0.1% of its authored point size), DejaVu Serif text reads as
# visibly larger than same-sized Times text. Nimbus Roman is Times-metric-
# compatible (the standard substitute on Linux/TeXLive) for machines that
# don't have the real Times New Roman available.
FIGURE_SERIF_FONTS = ["Times New Roman", "Times", "Nimbus Roman No9 L", "Nimbus Roman", "DejaVu Serif"]


def set_rcparams(font_size=FIGURE_FONT_PT):
    """Apply the document-matched figure font (size + Times-family serif) via
    matplotlib rcParams. Call once before building any figures in a script --
    rcParams are global and persist across all subsequent plt.subplots()/
    plt.figure() calls, so this doesn't need to be repeated per figure."""
    import matplotlib.pyplot as plt
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


def check_figure_layout(fig, label="", verbose=True):
    """Verify a figure's actual rendered geometry rather than trusting a
    preview render at whatever size a viewer happens to display it -- a
    downscaled/enlarged preview can make genuinely overlapping text look fine
    (or cramped-but-legal spacing look broken). Checks, using each artist's
    real get_window_extent(): (1) any two text elements (titles -- including
    loc="left"/"right" titles, tick labels, axis labels, legend boxes,
    fig.texts -- overlapping each other, (2) any text overlapping a *different*
    panel's plot area (its own panel is exempted), (3) the figure's
    tight-bbox exceeding its nominal canvas (which would make save_fig's
    bbox_inches="tight" silently grow the saved image past its authored,
    target-column-width size). Returns the number of issues found (0 = clean);
    also prints each one plus the resulting height-at-authored-width when
    verbose, since that's what actually reproduces on the page via
    \\includegraphics[width=...] (which scales height to match the saved
    image's pixel aspect ratio -- an author-time aspect ratio that doesn't
    match a real page reproduces at an unusable printed size even when the
    width comes out exactly right)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    texts = []
    for ax in fig.axes:
        lt = getattr(ax, "_left_title", None)
        if lt is not None and lt.get_text():
            texts.append((f"{id(ax)}:left-title:{lt.get_text()!r}", lt, ax))
        rt = getattr(ax, "_right_title", None)
        if rt is not None and rt.get_text():
            texts.append((f"{id(ax)}:right-title:{rt.get_text()!r}", rt, ax))
        if ax.get_title():
            texts.append((f"{id(ax)}:title:{ax.get_title()!r}", ax.title, ax))
        # ax.get_xticks()/get_yticks() (and the label objects paired with
        # them) are the locator's full candidate set, not what's actually
        # painted -- e.g. a log-scale LogLocator commonly returns whole
        # decades reaching well past the axes' real view limits (a panel
        # capped at ~11kHz can still list a "10^6 Hz" candidate). matplotlib
        # silently skips drawing those at render time, but leaves their Text
        # objects (and stale/meaningless window extents) sitting around, so
        # they must be filtered to the current view before checking overlap
        # -- otherwise a tick matplotlib never actually draws can "overlap"
        # something purely because its leftover position happens to coincide
        # with unrelated content.
        xlo, xhi = sorted(ax.get_xlim())
        for loc, t in zip(ax.get_xticks(), ax.get_xticklabels()):
            if t.get_text() and xlo <= loc <= xhi:
                texts.append((f"{id(ax)}:tick:{t.get_text()!r}", t, ax))
        ylo, yhi = sorted(ax.get_ylim())
        for loc, t in zip(ax.get_yticks(), ax.get_yticklabels()):
            if t.get_text() and ylo <= loc <= yhi:
                texts.append((f"{id(ax)}:tick:{t.get_text()!r}", t, ax))
        for t in ax.texts:
            if t.get_text():
                texts.append((f"{id(ax)}:text:{t.get_text()[:25]!r}", t, ax))
        for lab in (ax.xaxis.get_label(), ax.yaxis.get_label()):
            if lab.get_text():
                texts.append((f"{id(ax)}:axlabel:{lab.get_text()!r}", lab, ax))
        leg = ax.get_legend()
        if leg is not None:
            texts.append((f"{id(ax)}:legendbox", leg, ax))
    for t in fig.texts:
        if t.get_text():
            texts.append(("fig:text", t, None))
    # build_encoding_legend() (used throughout RQ1-RQ5) adds fig.legend(...)
    # + fig.add_artist(leg) legends -- these live in fig.legends, not any
    # ax.get_legend(), so without this they were never checked here at all
    # (an RQ2 legend overlapping its own x-tick labels rendered with 0
    # issues reported before this was added).
    for i, leg in enumerate(fig.legends):
        texts.append((f"fig:legend[{i}]", leg, None))

    boxes = [(n, t.get_window_extent(renderer=renderer), owner) for n, t, owner in texts]
    plot_areas = [(ax.get_title() or id(ax), ax.get_window_extent(renderer=renderer), ax) for ax in fig.axes]

    n_issues = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            n1, b1, o1 = boxes[i]
            n2, b2, o2 = boxes[j]
            if b1.overlaps(b2):
                if verbose:
                    print(f"  [{label}] TEXT-TEXT OVERLAP:", n1, "<->", n2)
                n_issues += 1
    for n, b, owner in boxes:
        for pname, pbb, pax in plot_areas:
            if pax is owner:
                continue
            if b.overlaps(pbb):
                if verbose:
                    print(f"  [{label}] TEXT-INTO-PLOTAREA:", n, "-> into panel", repr(pname))
                n_issues += 1

    w, h = fig.get_size_inches()
    tb = fig.get_tightbbox(renderer)
    ovf = (round(-tb.x0, 3) if tb.x0 < 0 else 0, round(tb.x1 - w, 3) if tb.x1 > w else 0,
           round(tb.y1 - h, 3) if tb.y1 > h else 0, round(-tb.y0, 3) if tb.y0 < 0 else 0)
    if any(ovf):
        n_issues += 1
    if verbose:
        print(f"[{label}] {n_issues} issues, overflow L/R/T/B={ovf}, "
              f"authored size={w:.3f}x{h:.3f}in (aspect h/w={h / w:.3f})")
    return n_issues


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
                           bbox_to_anchor=(0.5, -0.05), entries_ncol=1, labelspacing=None):
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

    labelspacing: vertical spacing between a legend's own entry rows, in
    font-size units -- forwarded straight to matplotlib's Legend (default:
    None, i.e. matplotlib's own rcParam default, currently unset here).
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
                          bbox_to_anchor=(x, bbox_to_anchor[1]), ncol=entries_ncol[i], frameon=False,
                          labelspacing=labelspacing)
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


def lighten_color(color, amount=0.5):
    """Blend `color` toward white by `amount` (0 = unchanged, 1 = white) --
    used to distinguish two series sharing the same identity/domain color
    (e.g. synthetic vs. soundscape) by shade instead of a 2nd hue."""
    r, g, b = mcolors.to_rgb(color)
    return (1 - amount) * r + amount, (1 - amount) * g + amount, (1 - amount) * b + amount
