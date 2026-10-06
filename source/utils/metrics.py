import os

from sklearn.metrics import (
    f1_score, cohen_kappa_score, mean_absolute_error,
    root_mean_squared_error, accuracy_score, precision_recall_fscore_support,
    confusion_matrix,
)
from scipy.stats import pearsonr
import numpy as np
import pandas as pd


def macro_mae_by_level(levels, errors, level_range):
    """Unweighted mean of per-level mean error ("macro-MAE"), over whichever
    levels in `level_range` have at least one example -- same "average the
    per-level MAE, not the per-example MAE" convention used throughout
    source/scripts/plot_rq1.py and compute_rq2_per_level_accuracy.py
    (macro-MAE(1-6) etc.), just computed inline here at evaluation time
    instead of in a later analysis script.

    levels: per-example bin id (e.g. true polyphony level, or min_polyphony
        for range ground truth), any real-valued array that will be rounded
        to the nearest int bin.
    errors: per-example error (e.g. |true - pred| or the clipped
        distance-to-range error), same length as `levels`.
    level_range: iterable of bin ids to average over (e.g. range(0, 7) for
        "all bins", range(1, 7) for "from level 1 upwards").

    Returns a dict with the macro-averaged MAE, how many of level_range's
    bins actually had support, the worst (highest-MAE) bin among those with
    support and its MAE, and the full per-bin MAE map (for printing/
    reporting; not meant to be flattened into a metrics-table row).
    """
    levels = np.round(np.asarray(levels)).astype(int)
    errors = np.asarray(errors, dtype=float)
    per_level = {}
    for level in level_range:
        mask = levels == level
        if mask.any():
            per_level[int(level)] = float(errors[mask].mean())
    if not per_level:
        return {"macro_mae": float("nan"), "n_levels_used": 0, "worst_bin": None,
                "worst_bin_mae": float("nan"), "per_level_mae": {}}
    worst_bin = max(per_level, key=per_level.get)
    return {
        "macro_mae": float(np.mean(list(per_level.values()))),
        "n_levels_used": len(per_level),
        "worst_bin": worst_bin,
        "worst_bin_mae": per_level[worst_bin],
        "per_level_mae": per_level,
    }


def _macro_mae_flat_keys(prefix: str, stats: dict) -> dict:
    """Flatten one macro_mae_by_level() result into the scalar
    `{prefix}_*` keys entity_metrics()/range entity_metrics() add to their
    (flattenable) 'overall'/per-species dict -- the per_level_mae map itself
    is deliberately left out here (kept as a sibling, non-flattened key by
    the caller) since utils.evaluation.flatten_metrics() writes every
    'overall' key as its own metrics-table row, and a dict value there would
    break that one-row-per-scalar-metric convention."""
    return {
        f"{prefix}_macro_mae": stats["macro_mae"],
        f"{prefix}_n_levels_used": stats["n_levels_used"],
        f"{prefix}_worst_bin": stats["worst_bin"],
        f"{prefix}_worst_bin_mae": stats["worst_bin_mae"],
    }


def print_confusion_matrix(y_true, y_pred, labels=None, title="", do_print=True, csv_path=None):
    """Build a confusion matrix (rows = true bin, cols = predicted bin),
    optionally print it as plain text to stdout, and optionally save it as a
    small CSV (integer counts, labels x labels).

    Saving is deliberately cheap and dependency-free (no plotting): the
    intent is that the evaluation scripts (evaluate_on_test_split.py,
    evaluate_on_val_split.py, evaluate_on_soundscape_data.py), which may run
    on a remote/cluster machine, only ever write this lightweight CSV next
    to their test_metrics.json/val_metrics.json -- rendering the actual
    heatmap PNG is left to source/scripts/render_confusion_matrices.py,
    meant to be run locally against a synced archive/ checkout, so a run
    never needs to produce (and you never need to download) a PNG per
    confusion matrix.

    do_print: print the text-formatted matrix (as before).
    csv_path: if given, write the matrix there as CSV (labels as both the
        index and header row); parent directories are created if needed.
        Independent of do_print, so a caller can save without printing.

    Returns (cm, labels) for callers that also want the raw array.
    """
    y_true = np.round(np.asarray(y_true)).astype(int)
    y_pred = np.round(np.asarray(y_pred)).astype(int)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not labels:
        if do_print:
            print(f"Confusion matrix ({title}): no data, skipping.")
        return None, labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if do_print:
        col_width = max(5, len(str(max(labels))) + 2)
        header = " " * (col_width + 1) + "".join(f"{l:>{col_width}}" for l in labels)
        lines = [f"\nConfusion matrix{f' ({title})' if title else ''} (rows=true, cols=pred, n={cm.sum()}):", header]
        for lbl, row in zip(labels, cm):
            lines.append(f"{lbl:>{col_width}} " + "".join(f"{v:>{col_width}}" for v in row))
        print("\n".join(lines))

    if csv_path:
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        pd.DataFrame(cm, index=pd.Index(labels, name="true"), columns=pd.Index(labels, name="pred")).to_csv(csv_path)
        if do_print:
            print(f"Saved confusion matrix CSV to {csv_path}")

    return cm, labels


def _range_macro_mae_stats(yt_min_, yt_max_, yp_, err) -> dict:
    """Both macro-MAE flavors compute_polyphony_range_metrics() reports (see
    its print_cm docstring), for one (yt_min, yt_max, yp) triple: bins run
    0..max(yt_max_) ("all bins") and 1..max(yt_max_) ("from level 1"), same
    bin-count convention as compute_polyphony_metrics()'s macro_mae_by_level
    calls."""
    max_level = int(yt_max_.max()) if len(yt_max_) else 0
    levels_all, levels_from1 = range(0, max_level + 1), range(1, max_level + 1)

    range_all = macro_mae_by_level(yt_min_, err, levels_all)
    range_from1 = macro_mae_by_level(yt_min_, err, levels_from1)

    unamb_mask = (yt_min_ == yt_max_)
    n_unambiguous = int(unamb_mask.sum())
    if n_unambiguous:
        true_level_unamb = yt_min_[unamb_mask]
        abs_err_unamb = np.abs(true_level_unamb - yp_[unamb_mask])
    else:
        true_level_unamb = np.array([], dtype=int)
        abs_err_unamb = np.array([], dtype=float)
    unamb_all = macro_mae_by_level(true_level_unamb, abs_err_unamb, levels_all)
    unamb_from1 = macro_mae_by_level(true_level_unamb, abs_err_unamb, levels_from1)

    return {"n_unambiguous": n_unambiguous, "range_all_bins": range_all, "range_from_1": range_from1,
            "unambiguous_all_bins": unamb_all, "unambiguous_from_1": unamb_from1}


def compute_polyphony_metrics(y_true, y_pred, cm_type, species_mapping=None,
                                num_classes=None, per_species=False, print_cm=True,
                                cm_csv_path=None):
    """
    Unified metrics for polyphony objectives, regardless of reg/class framing.
    y_true, y_pred: shape (N, num_species) for species-level objectives,
                    or (N, 1) for the combined 'total' objectives.
    cm_type: 'species_regression_round' | 'species_classification'
             (reuse same cm_type convention for the 'total_*' objectives,
             just with num_species == 1)
    print_cm: print the aggregate ('overall', not per-species) confusion
             matrix and macro-MAE breakdown to stdout. Ground truth here is
             exact (test/val split), so there is no unambiguous-vs-range
             distinction -- see compute_polyphony_range_metrics() for the
             soundscape (range ground truth) counterpart, which reports
             both.
    cm_csv_path: if given, also save the aggregate confusion matrix as a CSV
             there (see print_confusion_matrix()) -- independent of
             print_cm, so it can be saved without also being printed.
    """
    num_entities = y_true.shape[1]
    per_entity_yt, per_entity_yp = [], []

    if cm_type == "species_regression_round" or cm_type == "regression_round":
        for s in range(num_entities):
            yt_s, yp_s = prepare_polyphony_for_cm(y_true[:, s], y_pred[:, s])
            per_entity_yt.append(yt_s); per_entity_yp.append(yp_s)
        all_labels = sorted(np.unique(np.concatenate(per_entity_yt + per_entity_yp)).tolist())
    elif cm_type == "species_classification" or cm_type == "classification":  # species_classification
        for s in range(num_entities):
            yt_s, yp_s = prepare_classification_for_cm(y_true[:, s], y_pred[:, s, :])
            per_entity_yt.append(yt_s); per_entity_yp.append(yp_s)
        all_labels = list(range(num_classes)) if num_classes else sorted(
            np.unique(np.concatenate(per_entity_yt)).tolist())

    def entity_metrics(yt, yp):
        labels_present = sorted(set(yt.tolist()) | set(yp.tolist()))
        abs_err = np.abs(yt - yp)
        max_level = int(yt.max()) if len(yt) else 0
        macro_all = macro_mae_by_level(yt, abs_err, range(0, max_level + 1))
        macro_from1 = macro_mae_by_level(yt, abs_err, range(1, max_level + 1))
        return {
            "mae": mean_absolute_error(yt, yp),
            "rmse": root_mean_squared_error(yt, yp),
            "accuracy": accuracy_score(yt, yp),
            "off_by_one_accuracy": float(np.mean(np.abs(yt - yp) <= 1)),
            "macro_f1": f1_score(yt, yp, labels=labels_present, average='macro', zero_division=0),
            "weighted_f1": f1_score(yt, yp, labels=labels_present, average='weighted', zero_division=0),
            "qwk": cohen_kappa_score(yt, yp, labels=all_labels, weights='quadratic'),
            "pearson_r": pearsonr(yt, yp)[0] if len(yt) > 1 and np.std(yt) > 0 and np.std(yp) > 0 else float('nan'),
            "support": len(yt),
            **_macro_mae_flat_keys("macro_mae_all_bins", macro_all),
            **_macro_mae_flat_keys("macro_mae_from_1", macro_from1),
        }

    yt_all = np.concatenate(per_entity_yt)
    yp_all = np.concatenate(per_entity_yp)
    results = {"overall": entity_metrics(yt_all, yp_all)}

    # Sibling (non-flattened) per-bin breakdown, for the aggregate case only
    # -- same "all bins" / "from level 1" pair as the flat scalar keys above,
    # kept as a full per-level map here for inspection/plotting rather than
    # as metrics-table rows (see _macro_mae_flat_keys()).
    max_level = int(yt_all.max()) if len(yt_all) else 0
    macro_all = macro_mae_by_level(yt_all, np.abs(yt_all - yp_all), range(0, max_level + 1))
    macro_from1 = macro_mae_by_level(yt_all, np.abs(yt_all - yp_all), range(1, max_level + 1))
    results["macro_mae_breakdown"] = {"all_bins": macro_all["per_level_mae"], "from_level_1": macro_from1["per_level_mae"]}

    if print_cm:
        print(f"\n[{cm_type}] Macro-MAE (all bins): {macro_all['macro_mae']:.4f} "
              f"({macro_all['n_levels_used']} levels used; worst bin={macro_all['worst_bin']}, "
              f"MAE={macro_all['worst_bin_mae']:.4f})")
        print(f"[{cm_type}] Macro-MAE (from level 1): {macro_from1['macro_mae']:.4f} "
              f"({macro_from1['n_levels_used']} levels used; worst bin={macro_from1['worst_bin']}, "
              f"MAE={macro_from1['worst_bin_mae']:.4f})")
    if print_cm or cm_csv_path:
        print_confusion_matrix(yt_all, yp_all, labels=all_labels, title=cm_type,
                                do_print=print_cm, csv_path=cm_csv_path)

    if per_species and num_entities > 1:
        results["per_species"] = {}
        for s in range(num_entities):
            if species_mapping and str(s) in species_mapping:
                _, label_str = species_mapping[str(s)]
            elif species_mapping and s in species_mapping:
                _, label_str = species_mapping[s]
            else:
                label_str = str(s)
            results["per_species"][label_str] = entity_metrics(per_entity_yt[s], per_entity_yp[s])

    return results

def compute_polyphony_range_metrics(y_true, y_pred, cm_type, min_key="min_polyphony",
                                     max_key="max_polyphony", species_mapping=None,
                                     num_classes=None, per_species=False,
                                     overlap_threshold=1, print_cm=True, cm_csv_path=None):
    """
    Metrics for polyphony objectives evaluated against interval (min/max)
    ground truth, as used for soundscape test data.

    y_true: dict containing at least min_key and max_key. For the total
            objective these are shape (N,); for species-level objectives
            (min_key="min_species_polyphony", max_key="max_species_polyphony")
            these are shape (N, num_species).
    y_pred: shape (N,) or (N, 1) for total objectives; shape (N, num_species)
            for species regression; shape (N, num_species, num_classes) for
            species classification. Raw (non-discretized) model output —
            this function performs its own rounding/argmax based on cm_type,
            matching compute_polyphony_metrics.
    cm_type: 'species_regression_round' | 'species_classification'
    num_classes: only used for species_classification when you want all_labels
            to include classes with zero support in this test set; inferred
            from y_pred.shape[-1] if not given.
    overlap_threshold: polyphony count above which a clip is considered
        'polyphonic' for the auxiliary overlap-detection metric
        (default: max > 1).
    print_cm: print the aggregate ('overall', not per-species) macro-MAE
        breakdown and confusion matrix to stdout. Two macro-MAE flavors are
        reported, each for "all bins" (every polyphony level with support)
        and "from level 1" (excluding level 0, the dominant soundscape
        class -- same distinction plot_rq1.py's macro-MAE(1-6) makes):
          * "macro_range_mae_*": uses every clip, including ambiguous
            (min != max) ones, binned by min_polyphony and scored by the
            same clipped distance-to-[min,max] error as range_mae -- a
            macro-averaged version of the paper's range_mae that does not
            discard ambiguous clips.
          * "macro_mae_unambiguous_*": restricted to unambiguous clips
            (min_polyphony == max_polyphony), scored by plain |true - pred|
            against that single true level -- the same restricted
            macro-MAE(1-6) convention already used throughout
            source/scripts/plot_rq1.py and compute_rq2_per_level_accuracy.py
            (there, applied post-hoc; here, computed at evaluation time).
        The confusion matrix printed is also restricted to unambiguous
        clips, since ambiguous ground truth has no single true label to
        put on a confusion-matrix axis.
    cm_csv_path: if given, also save that (unambiguous-only) confusion
        matrix as a CSV there (see print_confusion_matrix()) -- independent
        of print_cm, so it can be saved without also being printed.
    """
    yt_min = np.asarray(y_true[min_key])
    yt_max = np.asarray(y_true[max_key])
    yp = np.asarray(y_pred)

    # normalize total-objective (N,) inputs to (N, 1)
    if yt_min.ndim == 1:
        yt_min = yt_min[:, None]
    if yt_max.ndim == 1:
        yt_max = yt_max[:, None]
    if cm_type == "species_regression_round" and yp.ndim == 1:
        yp = yp[:, None]

    num_entities = yt_min.shape[1]
    per_entity_min, per_entity_max, per_entity_pred = [], [], []

    if cm_type == "species_regression_round" or cm_type == "regression_round":  # species_regression_round
        for s in range(num_entities):
            per_entity_min.append(yt_min[:, s].astype(int))
            per_entity_max.append(yt_max[:, s].astype(int))
            per_entity_pred.append(np.round(yp[:, s]).astype(int))
    elif cm_type == "species_classification" or cm_type == "classification":  # species_classification
        inferred_num_classes = num_classes or yp.shape[-1]
        for s in range(num_entities):
            per_entity_min.append(yt_min[:, s].astype(int))
            per_entity_max.append(yt_max[:, s].astype(int))
            per_entity_pred.append(np.argmax(yp[:, s, :], axis=-1))
    else:
        raise ValueError(f"Unknown cm_type: {cm_type}")

    def effective_error(yt_min_, yt_max_, yp_):
        """Clipped distance from prediction to the [min, max] interval."""
        below = np.maximum(yt_min_ - yp_, 0)
        above = np.maximum(yp_ - yt_max_, 0)
        return np.maximum(below, above)

    def width_stratified(yt_min_, yt_max_, yp_, bins=(0, 0, 1, 2, np.inf)):
        """Range accuracy / MSE stratified by ground-truth interval width."""
        width = yt_max_ - yt_min_
        err = effective_error(yt_min_, yt_max_, yp_)
        strat = {}
        edges = sorted(set(bins))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (width > lo) & (width <= hi) if i > 0 else (width == lo)
            if mask.sum() == 0:
                continue
            key = f"width_{lo}-{hi}" if i > 0 else f"width_{lo}"
            strat[key] = {
                "range_accuracy": float(np.mean(err[mask] == 0)),
                "range_mse": float(np.mean(err[mask] ** 2)),
                "support": int(mask.sum()),
            }
        return strat

    def entity_metrics(yt_min_, yt_max_, yp_):
        err = effective_error(yt_min_, yt_max_, yp_)
        mid = (yt_min_ + yt_max_) / 2.0
        mid_round = np.round(mid).astype(int)

        all_labels = sorted(np.unique(np.concatenate(
            [yt_min_, yt_max_, mid_round, yp_]
        )).tolist())

        qwk_mid = cohen_kappa_score(mid_round, yp_, labels=all_labels, weights='quadratic')
        qwk_min = cohen_kappa_score(yt_min_, yp_, labels=all_labels, weights='quadratic')
        qwk_max = cohen_kappa_score(yt_max_, yp_, labels=all_labels, weights='quadratic')

        yt_overlap = (yt_max_ > overlap_threshold).astype(int)
        yp_overlap = (yp_ > overlap_threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt_overlap, yp_overlap, average='binary', zero_division=0
        )

        pearson_mid = (
            pearsonr(mid, yp_)[0]
            if len(yp_) > 1 and np.std(mid) > 0 and np.std(yp_) > 0
            else float('nan')
        )

        range_macro = _range_macro_mae_stats(yt_min_, yt_max_, yp_, err)

        return {
            "range_mae": float(np.mean(err)),
            "range_mse": float(np.mean(err ** 2)),
            "range_accuracy": float(np.mean(err == 0)),
            "off_by_one_range_accuracy": float(np.mean(err <= 1)),
            "qwk_vs_midpoint": qwk_mid,
            "qwk_vs_min": qwk_min,
            "qwk_vs_max": qwk_max,
            "pearson_r_vs_midpoint": pearson_mid,
            "mean_interval_width": float(np.mean(yt_max_ - yt_min_)),
            "overlap_precision": prec,
            "overlap_recall": rec,
            "overlap_f1": f1,
            "support": len(yp_),
            "n_unambiguous": range_macro["n_unambiguous"],
            **_macro_mae_flat_keys("macro_range_mae_all_bins", range_macro["range_all_bins"]),
            **_macro_mae_flat_keys("macro_range_mae_from_1", range_macro["range_from_1"]),
            **_macro_mae_flat_keys("macro_mae_unambiguous_all_bins", range_macro["unambiguous_all_bins"]),
            **_macro_mae_flat_keys("macro_mae_unambiguous_from_1", range_macro["unambiguous_from_1"]),
        }

    yt_min_all = np.concatenate(per_entity_min)
    yt_max_all = np.concatenate(per_entity_max)
    yp_all = np.concatenate(per_entity_pred)
    err_all = effective_error(yt_min_all, yt_max_all, yp_all)

    results = {
        "overall": entity_metrics(yt_min_all, yt_max_all, yp_all),
        "overall_by_interval_width": width_stratified(yt_min_all, yt_max_all, yp_all),
    }

    # Sibling (non-flattened) per-bin breakdowns for the aggregate case, plus
    # the printed summary -- see print_cm docstring above for what each of
    # the two macro-MAE flavors means.
    agg_macro = _range_macro_mae_stats(yt_min_all, yt_max_all, yp_all, err_all)
    results["macro_mae_breakdown"] = {
        "range_all_bins": agg_macro["range_all_bins"]["per_level_mae"],
        "range_from_level_1": agg_macro["range_from_1"]["per_level_mae"],
        "unambiguous_all_bins": agg_macro["unambiguous_all_bins"]["per_level_mae"],
        "unambiguous_from_level_1": agg_macro["unambiguous_from_1"]["per_level_mae"],
    }

    if print_cm:
        def _log(name, stats):
            print(f"[{cm_type}] {name}: {stats['macro_mae']:.4f} ({stats['n_levels_used']} levels used; "
                  f"worst bin={stats['worst_bin']}, MAE={stats['worst_bin_mae']:.4f})")
        print(f"\n[{cm_type}] n_unambiguous={agg_macro['n_unambiguous']} of {len(yp_all)} clips")
        _log("Macro range-MAE (all bins)", agg_macro["range_all_bins"])
        _log("Macro range-MAE (from level 1)", agg_macro["range_from_1"])
        _log("Macro MAE, unambiguous only (all bins)", agg_macro["unambiguous_all_bins"])
        _log("Macro MAE, unambiguous only (from level 1)", agg_macro["unambiguous_from_1"])

    if print_cm or cm_csv_path:
        if agg_macro["n_unambiguous"] > 0:
            unamb_mask = (yt_min_all == yt_max_all)
            print_confusion_matrix(yt_min_all[unamb_mask], yp_all[unamb_mask],
                                    title=f"{cm_type}, unambiguous ground truth only",
                                    do_print=print_cm, csv_path=cm_csv_path)
        elif print_cm:
            print(f"[{cm_type}] No unambiguous-ground-truth clips available; skipping confusion matrix.")

    if per_species and num_entities > 1:
        results["per_species"] = {}
        for s in range(num_entities):
            if species_mapping and str(s) in species_mapping:
                _, label_str = species_mapping[str(s)]
            elif species_mapping and s in species_mapping:
                _, label_str = species_mapping[s]
            else:
                label_str = str(s)
            results["per_species"][label_str] = entity_metrics(
                per_entity_min[s], per_entity_max[s], per_entity_pred[s]
            )

    return results


def prepare_classification_for_cm(y_true, y_pred):
    """
    Prepare multi-class classification logits for confusion matrix.
    
    Args:
        y_true: integer class labels, shape (N,)
        y_pred: raw logits, shape (N, num_classes)
    
    Returns:
        yt: integer labels as numpy array
        yp: predicted class indices as numpy array
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Convert logits to predicted class index
    yp = np.argmax(y_pred, axis=-1)
    yt = y_true.astype(int)
    
    return yt, yp

def prepare_polyphony_for_cm(y_true, y_pred):
    """
    y_true: (N,) or (N, 1)
    y_pred: (N,) or (N, 1)
    """
    y_true = np.atleast_1d(np.asarray(y_true).squeeze()).astype(int)
    y_pred = np.atleast_1d(np.round(np.asarray(y_pred).squeeze())).astype(int)
    return y_true, y_pred

def prepare_event_logits_for_cm(y_true, y_pred_logits, threshold=0.5):
    """
    y_true: (N, T)
    y_pred_logits: (N, T)
    """
    y_true = np.asarray(y_true).astype(int)

    # sigmoid
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))
    y_pred = (y_pred_probs >= threshold).astype(int)

    # flatten time
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    return y_true_flat, y_pred_flat