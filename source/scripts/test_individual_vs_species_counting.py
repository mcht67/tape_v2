#!/usr/bin/env python3
"""Does the BirdMAE polyphony model count individuals, or only species?

The polyphony_reg head is trained to count calling *individuals*. Whenever a
clip contains several individuals of the same species, the individual count
exceeds the number of distinct species. A model that had (only) learned
"how many species are calling" would track the species count on exactly
those clips. This script tests which of the two the model follows.

Data
----
- Fine-Tuning (synthetic XCM test set): archive/Fine-Tuning/BirdSetBirdMAE/
  eval_results/XCM_reg_test_results.pkl. Ground truth is the exact individual
  count (y_true.polyphony). The pkl carries no species info and no ids, so
  the species count is recovered from data/XCM/XCM_polyphonic['test']
  (species_polyphony = per-species individual counts -> n_sp = #non-zero),
  after re-applying the same max_polyphony / min_snr filter as
  evaluate_on_test_split.py. Rows are then aligned by position; alignment is
  verified on polyphony and snr_dB and the script aborts on mismatch.
- XCM-Generalization-fine-tune (real soundscapes, 7 regions): scape_eval_
  results/{S}_reg_scape_test_results.pkl. Ground truth is only a range
  [min_polyphony, max_polyphony]. Species count per clip comes from
  data/{S}/{S}_soundscape_test['test_5s'] (n_sp = #distinct
  ebird_code_multilabel). No filter is applied in evaluate_on_soundscape_
  data.py, so rows align by position; verified on min/max_polyphony.

Test
----
Both data types are fitted with the same model:

    pred ~ a + beta_new * n_sp + beta_dup * extra

where extra is the number of individuals beyond the species count
(synthetic: n_ind - n_sp; soundscape: min_polyphony - n_sp, i.e. the
individuals that are *certain* beyond one per species, because
add_min_max_polyphony sets min_polyphony = max(max_overlap, n_sp)).
beta_new is the response to an extra individual of a *new* species,
beta_dup to an extra individual of an *already present* species. The index

    rho = beta_dup / beta_new

is ~1 for an individual counter and ~0 for a species counter. 95% CIs by
row bootstrap.

On the "deciding" clips (synthetic: n_sp < n_ind; soundscape: min_polyphony
> n_sp, where a species counter would predict below the valid range) we
additionally compare the distance of the prediction to the species count
with its distance to the individual count/range (share closer to species
count, Wilcoxon signed-rank test), and break the rounded predictions down
into: at/below species count, between, within individual range, above.

Caveat: same-species individuals overlap more acoustically than different
species, so rho < 1 can partly reflect genuinely harder separation rather
than "species counting". rho ~ 0 would be strong evidence for the latter.

Outputs (--out-dir, default plots/figures/individual_vs_species):
    individual_vs_species_synthetic.csv
    individual_vs_species_soundscape.csv
    individual_vs_species.(png|pdf)

Usage (needs the local datasets, i.e. run on the cluster):
    complete-venv/bin/python source/scripts/test_individual_vs_species_counting.py
    ... --self-test   # replace predictions by simulated individual/species
                      # counters and check that rho comes out ~1 / ~0
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_style import TWO_COL_WIDTH_IN, check_figure_layout, save_fig, set_rcparams
from utils.dataset import get_local_data_dir

BACKBONE = "BirdSetBirdMAE"
FINETUNE_STUDY = "Fine-Tuning"
GENERALIZATION_STUDY = "XCM-Generalization-fine-tune"
SYNTHETIC_SUBSET = "XCM"
SOUNDSCAPE_SUBSETS = ["UHH", "HSN", "POW", "NES", "PER", "SSW", "SNE"]
DEFAULT_MAX_POLYPHONY = 8
DEFAULT_MIN_SNR = -10
SEED = 42

# Reference categorical palette (dataviz skill, light mode), fixed order.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Outcome categories for rounded predictions on deciding clips.
OUTCOMES = ["at/below species count", "between", "within individual range", "above range"]
OUTCOME_COLORS = [C_ORANGE, C_YELLOW, C_BLUE, "#9e9d97"]


# --------------------------------------------------------------------------
# Loading + alignment
# --------------------------------------------------------------------------

def _filter_params(study_dir: Path, subset: str):
    """max_polyphony / min_snr used by the eval run of `subset` in study_dir."""
    for params_path in sorted(study_dir.glob("*/logs/params.yaml")):
        cfg = yaml.safe_load(params_path.read_text())
        if cfg.get("dataset", {}).get("subset") == subset:
            ds = cfg["dataset"]
            return ds.get("max_polyphony", DEFAULT_MAX_POLYPHONY), ds.get("min_snr", DEFAULT_MIN_SNR)
    print(f"[WARN] no params.yaml for subset {subset} in {study_dir}; using defaults.")
    return DEFAULT_MAX_POLYPHONY, DEFAULT_MIN_SNR


def load_synthetic(archive_root: Path, data_root: Path, subset: str, pkl_path: Path | None = None):
    """Per-sample frame with n_ind, n_sp, pred for the synthetic test split."""
    from datasets import load_from_disk

    study_dir = archive_root / FINETUNE_STUDY / BACKBONE
    pkl_path = pkl_path or study_dir / "eval_results" / f"{subset}_reg_test_results.pkl"
    res = pd.read_pickle(pkl_path)

    max_poly, min_snr = _filter_params(study_dir, subset)
    ds_dir = data_root / get_local_data_dir(dataset_config=f"{subset}_polyphonic", subset=subset)
    meta = (load_from_disk(str(ds_dir))["test"]
            .select_columns(["polyphony", "species_polyphony", "snr_dB"]).to_pandas())
    if max_poly is not None:
        meta = meta[meta.polyphony <= max_poly]
    if min_snr is not None:
        meta = meta[meta.snr_dB >= min_snr]
    meta = meta.reset_index(drop=True)

    if len(meta) != len(res):
        raise RuntimeError(f"{subset}: {len(meta)} filtered dataset rows vs {len(res)} pkl rows ({pkl_path}).")
    if not np.array_equal(meta.polyphony.to_numpy(), res["y_true.polyphony"].to_numpy().astype(int)):
        raise RuntimeError(f"{subset}: polyphony column of dataset and pkl not aligned.")
    if "variable_values.snr_dB" in res and not np.allclose(meta.snr_dB, res["variable_values.snr_dB"]):
        raise RuntimeError(f"{subset}: snr_dB column of dataset and pkl not aligned.")

    n_ind = meta.polyphony.to_numpy()
    n_sp = meta.species_polyphony.apply(lambda v: int(np.count_nonzero(v))).to_numpy()
    if np.any(n_sp > n_ind):
        raise RuntimeError(f"{subset}: species count exceeds individual count for some rows.")
    return pd.DataFrame({
        "subset": subset, "n_ind": n_ind, "n_sp": n_sp,
        "extra": n_ind - n_sp, "pred": res["predictions.polyphony_reg"].to_numpy(),
    })


def load_soundscape(archive_root: Path, data_root: Path, subset: str):
    """Per-clip frame with n_sp, min/max individual range, pred."""
    from datasets import load_from_disk

    pkl_path = (archive_root / GENERALIZATION_STUDY / BACKBONE / "scape_eval_results"
                / f"{subset}_reg_scape_test_results.pkl")
    res = pd.read_pickle(pkl_path)

    ds_dir = data_root / get_local_data_dir(dataset_config=f"{subset}_soundscape_test", subset=subset)
    meta = (load_from_disk(str(ds_dir))["test_5s"]
            .select_columns(["min_polyphony", "max_polyphony", "ebird_code_multilabel"]).to_pandas())

    if len(meta) != len(res):
        raise RuntimeError(f"{subset}: {len(meta)} soundscape rows vs {len(res)} pkl rows ({pkl_path}).")
    for col in ("min_polyphony", "max_polyphony"):
        if not np.array_equal(meta[col].to_numpy(), res[f"y_true.{col}"].to_numpy().astype(int)):
            raise RuntimeError(f"{subset}: {col} of dataset and pkl not aligned.")

    n_sp = meta.ebird_code_multilabel.apply(lambda v: len(set(v)) if v is not None else 0).to_numpy()
    lo, hi = meta.min_polyphony.to_numpy(), meta.max_polyphony.to_numpy()
    if np.any(lo < n_sp):
        raise RuntimeError(f"{subset}: min_polyphony < species count, contradicts add_min_max_polyphony.")
    return pd.DataFrame({
        "subset": subset, "n_sp": n_sp, "min_ind": lo, "max_ind": hi,
        "extra": lo - n_sp, "pred": res["predictions.polyphony_reg"].to_numpy(),
    })


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _ols(n_sp, extra, pred):
    X = np.column_stack([np.ones_like(n_sp, dtype=float), n_sp, extra])
    coef, *_ = np.linalg.lstsq(X, pred, rcond=None)
    return coef  # a, beta_new, beta_dup


def fit_rho(df: pd.DataFrame, n_boot: int):
    """beta_new, beta_dup and rho = beta_dup / beta_new with bootstrap 95% CIs."""
    n_sp, extra, pred = (df[c].to_numpy(dtype=float) for c in ("n_sp", "extra", "pred"))
    _, b_new, b_dup = _ols(n_sp, extra, pred)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        boots.append(_ols(n_sp[idx], extra[idx], pred[idx])[1:])
    boots = np.array(boots)
    rho_boot = boots[:, 1] / boots[:, 0]
    q = lambda a: np.nanpercentile(a, [2.5, 97.5])
    return {
        "beta_new": b_new, "beta_new_lo": q(boots[:, 0])[0], "beta_new_hi": q(boots[:, 0])[1],
        "beta_dup": b_dup, "beta_dup_lo": q(boots[:, 1])[0], "beta_dup_hi": q(boots[:, 1])[1],
        "rho": b_dup / b_new, "rho_lo": q(rho_boot)[0], "rho_hi": q(rho_boot)[1],
    }


def _wilcoxon_p(d_sp, d_ind):
    diff = d_sp - d_ind
    if len(diff) < 2 or np.all(diff == 0):
        return np.nan
    return wilcoxon(d_sp, d_ind).pvalue


def _outcome_shares(pred_round, n_sp, lo, hi):
    """Share of deciding clips per outcome category (see OUTCOMES)."""
    cats = np.select(
        [pred_round <= n_sp, pred_round < lo, pred_round <= hi],
        [0, 1, 2], default=3)
    return {name: float(np.mean(cats == i)) if len(cats) else np.nan for i, name in enumerate(OUTCOMES)}


def synthetic_stats(df: pd.DataFrame, n_boot: int):
    dec = df[df.n_sp < df.n_ind]
    d_ind, d_sp = (dec.pred - dec.n_ind).abs(), (dec.pred - dec.n_sp).abs()
    row = {"subset": df.subset.iat[0], "n": len(df), "n_deciding": len(dec),
           **fit_rho(df, n_boot),
           "mae_vs_individuals_deciding": d_ind.mean(), "mae_vs_species_deciding": d_sp.mean(),
           "frac_closer_to_species": float(np.mean(d_sp < d_ind)) if len(dec) else np.nan,
           "wilcoxon_p": _wilcoxon_p(d_sp.to_numpy(), d_ind.to_numpy())}
    row.update(_outcome_shares(np.rint(dec.pred).to_numpy(), dec.n_sp.to_numpy(),
                               dec.n_ind.to_numpy(), dec.n_ind.to_numpy()))
    return row


def soundscape_stats(df: pd.DataFrame, n_boot: int, name: str):
    dec = df[df.min_ind > df.n_sp]
    d_sp = (dec.pred - dec.n_sp).abs()
    d_int = np.maximum(0, np.maximum(dec.min_ind - dec.pred, dec.pred - dec.max_ind))
    amb = df[(df.min_ind == df.n_sp) & (df.max_ind > df.n_sp)]
    row = {"subset": name, "n": len(df), "n_deciding": len(dec), "n_ambiguous": len(amb),
           **(fit_rho(df, n_boot) if len(dec) >= 3 else {}),
           "mean_dist_to_species_deciding": d_sp.mean(), "mean_dist_to_individual_range_deciding": d_int.mean(),
           "frac_closer_to_species": float(np.mean(d_sp < d_int)) if len(dec) else np.nan,
           "wilcoxon_p": _wilcoxon_p(d_sp.to_numpy(), np.asarray(d_int))}
    row.update(_outcome_shares(np.rint(dec.pred).to_numpy(), dec.n_sp.to_numpy(),
                               dec.min_ind.to_numpy(), dec.max_ind.to_numpy()))
    return row


def verdict(rho, lo, hi):
    if np.isnan(rho):
        return "too few deciding clips"
    if lo > 0.5:
        return "follows individuals" if lo > 0.8 or hi >= 1 else "leans to individuals"
    if hi < 0.5:
        return "follows species" if hi < 0.2 else "leans to species"
    return "inconclusive"


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_MUTED, labelcolor=C_TEXT)
    ax.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def plot_synthetic_panel(ax, syn: pd.DataFrame):
    groups = [("0 (all different species)", syn.extra == 0, C_BLUE),
              ("1 same-species extra", syn.extra == 1, C_ORANGE),
              ("≥2 same-species extra", syn.extra >= 2, C_AQUA)]
    levels = np.arange(syn.n_ind.min(), syn.n_ind.max() + 1)
    ax.plot(levels, levels, ls="--", color=C_MUTED, lw=1, zorder=1, label="individual count (target)")
    for label, mask, color in groups:
        g = syn[mask].groupby("n_ind").pred.agg(["mean", "sem", "count"])
        g = g[g["count"] >= 5]
        if g.empty:
            continue
        ax.errorbar(g.index, g["mean"], yerr=g["sem"].fillna(0), color=color, lw=2, marker="o",
                    ms=4, capsize=0, label=label, zorder=3)
        if mask is groups[0][1]:
            continue  # species count == individual count, already the dashed line
        # Where a pure species counter would land for this group.
        sp_mean = syn[mask].groupby("n_ind").n_sp.mean().loc[g.index]
        ax.scatter(g.index, sp_mean, facecolors="none", edgecolors=color, s=22, lw=1, zorder=2)
    ax.scatter([], [], facecolors="none", edgecolors=C_MUTED, s=22, lw=1, label="species count (hollow)")
    ax.set_xlabel("true number of individuals")
    ax.set_ylabel("mean predicted count")
    ax.set_title(f"A  Synthetic {syn.subset.iat[0]} test set", loc="left", color=C_TEXT)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5)
    _style_axis(ax)


def _separate_synthetic_row(ax, rows, y):
    """Thin rule between the synthetic row (no n_ambiguous) and soundscape rows."""
    if len(rows) > 1 and "n_ambiguous" not in rows[0]:
        ax.axhline((y[0] + y[1]) / 2, color=C_MUTED, lw=0.6)


def plot_outcome_panel(ax, rows: list[dict]):
    labels = [f"{r['subset']} (n={r['n_deciding']})" for r in rows]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for name, color in zip(OUTCOMES, OUTCOME_COLORS):
        vals = np.array([r.get(name, np.nan) for r in rows], dtype=float)
        vals = np.nan_to_num(vals)
        ax.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=1, height=0.7, label=name)
        left += vals
    ax.set_yticks(y, labels)
    _separate_synthetic_row(ax, rows, y)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of deciding clips (rounded prediction)")
    ax.set_title("B  Where predictions land when individuals > species", loc="left", color=C_TEXT)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    _style_axis(ax)
    ax.grid(False, axis="y")


def plot_rho_panel(ax, rows: list[dict]):
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-10, 0, color="#fbe3d9", zorder=0)
    ax.axvline(0, color=C_ORANGE, lw=1, ls="--")
    ax.axvline(1, color=C_BLUE, lw=1, ls="--")
    for yi, r in zip(y, rows):
        if "rho" not in r or np.isnan(r["rho"]):
            continue
        color = C_TEXT if r["subset"] in (SYNTHETIC_SUBSET, "pooled") else C_MUTED
        ax.plot([r["rho_lo"], r["rho_hi"]], [yi, yi], color=color, lw=2, solid_capstyle="round")
        ax.scatter([r["rho"]], [yi], color=color, s=28, zorder=3)
    ax.set_yticks(y, [r["subset"] for r in rows])
    _separate_synthetic_row(ax, rows, y)
    lo = min([0] + [r["rho_lo"] for r in rows if "rho_lo" in r and np.isfinite(r["rho_lo"])])
    hi = max([1] + [r["rho_hi"] for r in rows if "rho_hi" in r and np.isfinite(r["rho_hi"])])
    ax.set_xlim(max(lo, -1) - 0.1, min(hi, 2.5) + 0.1)
    ax.set_xlabel(r"$\rho = \beta_{dup} / \beta_{new}$  (95% CI)")
    ax.set_title(r"C  Weight of a same-species extra individual", loc="left", color=C_TEXT)
    ax.text(0, 0.02, " counts\n species", transform=ax.get_xaxis_transform(), ha="left", va="bottom",
            color=C_MUTED, fontsize=7)
    ax.text(1, 0.02, "counts \nindividuals ", transform=ax.get_xaxis_transform(), ha="right", va="bottom",
            color=C_MUTED, fontsize=7)
    _style_axis(ax)
    ax.grid(False, axis="y")


def make_figure(syn: pd.DataFrame | None, syn_row: dict | None, scape_rows: list[dict], out_stem: Path):
    set_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH_IN * 1.6, 3.4),
                             gridspec_kw={"width_ratios": [1.2, 1.1, 0.9]})
    if syn is not None:
        plot_synthetic_panel(axes[0], syn)
    else:
        axes[0].text(0.5, 0.5, "synthetic data unavailable", ha="center", transform=axes[0].transAxes)
    outcome_rows = ([syn_row] if syn_row else []) + scape_rows
    if outcome_rows:
        plot_outcome_panel(axes[1], outcome_rows)
        plot_rho_panel(axes[2], outcome_rows)
    fig.tight_layout()
    check_figure_layout(fig, label=out_stem.name, verbose=True)
    save_fig(fig, out_stem)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _simulate(df: pd.DataFrame, kind: str, rng):
    """Replace predictions by an idealised individual or species counter."""
    df = df.copy()
    ind = df["n_ind"] if "n_ind" in df else df["min_ind"]
    target = ind if kind == "individuals" else df["n_sp"]
    df["pred"] = target + rng.normal(0, 0.3, len(df))
    return df


def run_self_test(syn: pd.DataFrame | None, scapes: list[pd.DataFrame], n_boot: int):
    rng = np.random.default_rng(SEED)
    frames = ([syn] if syn is not None else []) + scapes
    for df in frames:
        for kind, expected in (("individuals", 1.0), ("species", 0.0)):
            res = fit_rho(_simulate(df, kind, rng), n_boot=200)
            ok = abs(res["rho"] - expected) < 0.15
            print(f"[self-test] {df.subset.iat[0]:>4} simulated {kind:<11} rho={res['rho']:.3f} "
                  f"(expected {expected}) -> {'OK' if ok else 'FAIL'}")
            if not ok:
                raise AssertionError("self-test failed")


def main():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive-root", type=Path, default=repo / "archive")
    p.add_argument("--data-root", type=Path, default=repo)
    p.add_argument("--out-dir", type=Path, default=repo / "plots/figures/individual_vs_species")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--subsets", nargs="+", default=SOUNDSCAPE_SUBSETS, help="Soundscape regions to include.")
    p.add_argument("--synthetic-subset", default=SYNTHETIC_SUBSET,
                   help="Debug: synthetic dataset to use instead of XCM (pair with --synthetic-pkl).")
    p.add_argument("--synthetic-pkl", type=Path, default=None, help="Debug: override synthetic results pkl.")
    p.add_argument("--self-test", action="store_true",
                   help="Check the test on simulated individual/species counters, then exit.")
    args = p.parse_args()
    os.chdir(args.data_root)  # get_local_data_dir returns repo-relative paths

    syn = None
    try:
        syn = load_synthetic(args.archive_root, Path("."), args.synthetic_subset, args.synthetic_pkl)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[WARN] skipping synthetic {args.synthetic_subset}: {e}")

    scapes = []
    for s in args.subsets:
        try:
            scapes.append(load_soundscape(args.archive_root, Path("."), s))
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            print(f"[WARN] skipping soundscape {s}: {type(e).__name__}: {e}")

    if args.self_test:
        run_self_test(syn, scapes, args.n_boot)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    syn_row = None
    if syn is not None:
        syn_row = synthetic_stats(syn, args.n_boot)
        pd.DataFrame([syn_row]).to_csv(args.out_dir / "individual_vs_species_synthetic.csv", index=False)

    scape_rows = [soundscape_stats(df, args.n_boot, df.subset.iat[0]) for df in scapes]
    if len(scapes) > 1:
        scape_rows.append(soundscape_stats(pd.concat(scapes, ignore_index=True), args.n_boot, "pooled"))
    if scape_rows:
        pd.DataFrame(scape_rows).to_csv(args.out_dir / "individual_vs_species_soundscape.csv", index=False)

    print("\n=== Individual vs species counting (rho = 1: individuals, 0: species) ===")
    for r in ([syn_row] if syn_row else []) + scape_rows:
        rho = r.get("rho", np.nan)
        print(f"{r['subset']:>7}  n={r['n']:>5}  deciding={r['n_deciding']:>5}  "
              f"rho={rho:6.3f} [{r.get('rho_lo', np.nan):6.3f}, {r.get('rho_hi', np.nan):6.3f}]  "
              f"closer-to-species={r['frac_closer_to_species']:.1%}  p={r['wilcoxon_p']:.2g}  "
              f"-> {verdict(rho, r.get('rho_lo', np.nan), r.get('rho_hi', np.nan))}")
    print("Caveat: same-species individuals overlap acoustically more than different species, so rho < 1 "
          "can partly reflect harder separation; rho ~ 0 would indicate species counting.")

    make_figure(syn, syn_row, scape_rows, args.out_dir / "individual_vs_species")
    print(f"\nWrote CSVs and figure to {args.out_dir}")


if __name__ == "__main__":
    main()
