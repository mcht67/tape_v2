from sklearn.metrics import (
    f1_score, cohen_kappa_score, mean_absolute_error,
    root_mean_squared_error, accuracy_score, precision_recall_fscore_support
)
from scipy.stats import pearsonr
import numpy as np

def compute_polyphony_metrics(y_true, y_pred, cm_type, species_mapping=None,
                                num_classes=None, per_species=False):
    """
    Unified metrics for polyphony objectives, regardless of reg/class framing.
    y_true, y_pred: shape (N, num_species) for species-level objectives,
                    or (N, 1) for the combined 'total' objectives.
    cm_type: 'species_regression_round' | 'species_classification'
             (reuse same cm_type convention for the 'total_*' objectives,
             just with num_species == 1)
    """
    num_entities = y_true.shape[1]
    per_entity_yt, per_entity_yp = [], []

    if cm_type == "species_regression_round" or "regression_round":
        for s in range(num_entities):
            yt_s, yp_s = prepare_polyphony_for_cm(y_true[:, s], y_pred[:, s])
            per_entity_yt.append(yt_s); per_entity_yp.append(yp_s)
        all_labels = sorted(np.unique(np.concatenate(per_entity_yt + per_entity_yp)).tolist())
    elif cm_type == "species_classification" or "classification":  # species_classification
        for s in range(num_entities):
            yt_s, yp_s = prepare_classification_for_cm(y_true[:, s], y_pred[:, s, :])
            per_entity_yt.append(yt_s); per_entity_yp.append(yp_s)
        all_labels = list(range(num_classes)) if num_classes else sorted(
            np.unique(np.concatenate(per_entity_yt)).tolist())

    def entity_metrics(yt, yp):
        labels_present = sorted(set(yt.tolist()) | set(yp.tolist()))
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
        }

    results = {"overall": entity_metrics(np.concatenate(per_entity_yt), np.concatenate(per_entity_yp))}

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
                                     overlap_threshold=1):
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
        }

    yt_min_all = np.concatenate(per_entity_min)
    yt_max_all = np.concatenate(per_entity_max)
    yp_all = np.concatenate(per_entity_pred)

    results = {
        "overall": entity_metrics(yt_min_all, yt_max_all, yp_all),
        "overall_by_interval_width": width_stratified(yt_min_all, yt_max_all, yp_all),
    }

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