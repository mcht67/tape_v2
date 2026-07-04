from sklearn.metrics import (
    f1_score, cohen_kappa_score, mean_absolute_error,
    root_mean_squared_error, accuracy_score
)
from scipy.stats import pearsonr
import numpy as np

def compute_polyphony_metrics(y_true, y_pred, cm_type, species_mapping=None,
                                num_classes=None, per_species=True):
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

    if cm_type == "species_regression_round":
        for s in range(num_entities):
            yt_s, yp_s = prepare_polyphony_for_cm(y_true[:, s], y_pred[:, s])
            per_entity_yt.append(yt_s); per_entity_yp.append(yp_s)
        all_labels = sorted(np.unique(np.concatenate(per_entity_yt + per_entity_yp)).tolist())
    else:  # species_classification
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