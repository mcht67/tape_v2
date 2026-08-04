import os
import numpy as np
import pandas as pd
import tensorflow as tf 

from collections import Counter

import numpy as np
import tensorflow as tf
from collections import Counter


def collect_predictions(model, dataset, input_feature_name, variables=None,
                         birdset_id2label=None, batch_size=64):
    """
    Run inference over `dataset` and return arrays ready for metric computation.

    Returns:
        y_true: {"polyphony": (N,) or None, "species_polyphony": (N, num_species) or None}
        predictions: {objective_name: array}, matching model's named outputs
        variable_values: dict of {var_name: array of shape (N,)}
    """
    variables = variables or []
    n = len(dataset)

    gt_total, gt_min, gt_max = [None] * n, [None] * n, [None] * n
    gt_species, gt_min_species, gt_max_species = [None] * n, [None] * n, [None] * n
    variable_rows = [None] * n

    # Accumulate predictions per output head across batches instead of
    # keeping every embedding + every activation in memory at once.
    pred_chunks = {}

    # Only fetch the columns we actually need — avoids materializing
    # unrelated columns (e.g. raw audio) that HF may otherwise decode.
    needed_cols = {input_feature_name, "polyphony_degree", "polyphony",
                   "min_polyphony", "max_polyphony", "species_polyphony",
                   "min_species_polyphony", "max_species_polyphony",
                   "birdset_code_multilabel", "birdset_id_multilabel",
                   "ebird_code_multilabel"} | set(variables)
    needed_cols &= set(dataset.column_names)

    idx = 0
    # Dataset.iter() streams fixed-size batches as dicts of column -> list,
    # so only one batch's worth of decoded features is ever in memory,
    # and np.stack only ever stacks `batch_size` items at a time.
    for batch in dataset.select_columns(list(needed_cols)).iter(batch_size=batch_size):
        bsz = len(batch[input_feature_name])

        X = tf.constant(np.stack(batch[input_feature_name]), dtype=tf.float32)
        raw_predictions = model(X, training=False)
        for k, v in raw_predictions.items():
            arr = v.numpy()
            if k == "polyphony_reg" and arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            pred_chunks.setdefault(k, []).append(arr)

        for j in range(bsz):
            i = idx + j
            variable_rows[i] = {var: batch[var][j] for var in variables}

            # Get species agnostic polyphony values (total, min, max) and
            # species-specific polyphony counts
            if "polyphony_degree" in batch:
                gt_total[i] = batch["polyphony_degree"][j]
            if "polyphony" in batch:
                gt_total[i] = batch["polyphony"][j]
            if "min_polyphony" in batch:
                gt_min[i] = batch["min_polyphony"][j]
            if "max_polyphony" in batch:
                gt_max[i] = batch["max_polyphony"][j]

            # Get per-species polyphony counts if available
            if "species_polyphony" in batch:
                gt_species[i] = batch["species_polyphony"][j]
            if "min_species_polyphony" in batch:
                gt_min_species[i] = batch["min_species_polyphony"][j]
            if "max_species_polyphony" in batch:
                gt_max_species[i] = batch["max_species_polyphony"][j]

            if "species_polyphony" not in batch and birdset_id2label is not None:
                counts = None
                if batch.get('birdset_code_multilabel') is not None and batch['birdset_code_multilabel'][j] is not None:
                    counts = Counter(batch['birdset_code_multilabel'][j])
                elif batch.get('birdset_id_multilabel') is not None and batch['birdset_id_multilabel'][j] is not None:
                    counts = Counter(batch['birdset_id_multilabel'][j])
                elif batch.get('ebird_code_multilabel') is not None and batch['ebird_code_multilabel'][j] is not None:
                    counts = Counter(batch['ebird_code_multilabel'][j])

                labels = [counts.get(int(birdset_id), 0) for birdset_id in birdset_id2label.keys()] \
                    if counts is not None else [0] * len(birdset_id2label)
                gt_species[i] = labels

        idx += bsz

    predictions = {k: np.concatenate(v, axis=0) for k, v in pred_chunks.items()}

    y_true = {
        "polyphony": np.array(gt_total),  # shape (N,) — matches predictions["polyphony_reg"]
        "min_polyphony": np.array(gt_min),
        "max_polyphony": np.array(gt_max),
        "species_polyphony": np.array(gt_species),
        "min_species_polyphony": np.array(gt_min_species),
        "max_species_polyphony": np.array(gt_max_species),
    }

    variable_values = {
        var: np.array([row[var] for row in variable_rows])
        for var in variables
    }

    return y_true, predictions, variable_values

# def collect_predictions(model, dataset, input_feature_name, variables=None, birdset_id2label=None, batch_size=64):
#     """
#     Run inference over `dataset` and return arrays ready for metric computation.

#     Returns:
#         y_true: {"polyphony": (N,) or None, "species_polyphony": (N, num_species) or None}
#         predictions: {objective_name: array}, matching model's named outputs
#         variable_values: dict of {var_name: array of shape (N,)}
#     """
#     variables = variables or []
#     embeddings, variable_rows = [], []
#     gt_total, gt_min, gt_max = [None] * len(dataset), [None] * len(dataset), [None] * len(dataset)
#     gt_species, gt_min_species, gt_max_species = [None] * len(dataset), [None] * len(dataset), [None] * len(dataset)

#     for i, example in enumerate(dataset):
#         embeddings.append(example[input_feature_name])
#         variable_rows.append({var: example[var] for var in variables})

#         # Get species agnostic polyphony values (total, min, max) and species-specific polyphony counts
#         if "polyphony_degree" in example:
#             gt_total[i] = example["polyphony_degree"]
#         if "polyphony" in example:
#             gt_total[i] = example["polyphony"]
#         if "min_polyphony" in example:
#             gt_min[i] = example["min_polyphony"]
#         if "max_polyphony" in example:
#             gt_max[i] = example["max_polyphony"]

#         # Get per-species polyphony counts if available
#         if "species_polyphony" in example:
#             gt_species[i] = example["species_polyphony"]
#         if "min_species_polyphony" in example:
#             gt_min_species[i] = example["min_species_polyphony"]
#         if "max_species_polyphony" in example:
#             gt_max_species[i] = example["max_species_polyphony"]

#         # if species_names:
#         if not "species_polyphony" in example and birdset_id2label is not None:
#             counts = None
#             if 'birdset_code_multilabel' in example and example['birdset_code_multilabel'] is not None:
#                 counts = Counter(example['birdset_code_multilabel'])
#             elif 'birdset_id_multilabel' in example and example['birdset_id_multilabel'] is not None:
#                 counts = Counter(example['birdset_id_multilabel'])
#             elif 'ebird_code_multilabel' in example and example['ebird_code_multilabel'] is not None:
#                 counts = Counter(example['ebird_code_multilabel'])

#             labels = [counts.get(int(birdset_id), 0) for birdset_id in birdset_id2label.keys()]
            
#             gt_species[i] = labels
            
#     X = tf.constant(np.stack(embeddings), dtype=tf.float32)
#     raw_predictions = model(X, training=False)

#     # raw_predictions is a dict of {output_name: tensor} since the model has
#     # multiple named heads. Convert to numpy and squeeze trailing singleton
#     # dims only where main() expects 1D (the total-polyphony regression head).
#     predictions = {}
#     for k, v in raw_predictions.items():
#         arr = v.numpy()
#         if k == "polyphony_reg" and arr.ndim == 2 and arr.shape[1] == 1:
#             arr = arr[:, 0]
#         predictions[k] = arr

#     y_true = {
#         "polyphony": np.array(gt_total),  # shape (N,) — matches predictions["polyphony_reg"]
#         "min_polyphony": np.array(gt_min),
#         "max_polyphony": np.array(gt_max),
#         "species_polyphony": np.array(gt_species), #if species_names else None,
#         "min_species_polyphony": np.array(gt_min_species),
#         "max_species_polyphony": np.array(gt_max_species),
#     }

#     variable_values = {
#         var: np.array([row[var] for row in variable_rows])
#         for var in variables
#     }

#     return y_true, predictions, variable_values


def arrays_to_records(y_true, predictions, variable_values=None, species_names=None):
    """Convert arrays back into self-contained per-example records for storage."""
    n = next(iter(predictions.values())).shape[0]

    if variable_values is None:
        variable_values = []

    variable_rows = [
        {var: variable_values[var][i] for var in variable_values}
        for i in range(n)
    ]

    records = []
    for i in range(n):
        rec_y_true = {}
        if y_true.get("polyphony") is not None:
            rec_y_true["polyphony"] = None if y_true["polyphony"][i] is None else float(y_true["polyphony"][i])
        if y_true.get("min_polyphony") is not None:
            rec_y_true["min_polyphony"] = None if y_true["min_polyphony"][i] is None else float(y_true["min_polyphony"][i])
        if y_true.get("max_polyphony") is not None:
            rec_y_true["max_polyphony"] = None if y_true["max_polyphony"][i] is None else float(y_true["max_polyphony"][i])
        if y_true.get("species_polyphony") is not None and species_names:
            rec_y_true["species_polyphony"] = dict(zip(species_names, y_true["species_polyphony"][i].tolist()))
        if y_true.get("min_species_polyphony") is not None and species_names:
            rec_y_true["min_species_polyphony"] = dict(zip(species_names, y_true["min_species_polyphony"][i].tolist()))
        if y_true.get("max_species_polyphony") is not None and species_names:
            rec_y_true["max_species_polyphony"] = dict(zip(species_names, y_true["max_species_polyphony"][i].tolist()))

        rec_predictions = {}
        for obj, arr in predictions.items():
            val = arr[i]
            rec_predictions[obj] = val.tolist() if hasattr(val, "tolist") else val

        records.append({
            "y_true": rec_y_true,
            "predictions": rec_predictions,
            "variable_values": variable_rows[i],
        })
    return records


def records_to_arrays(records, variables=None, species_names=None):
    """Convert list of per-example records back into arrays for metric computation."""
    variables = variables or []
    n = len(records)
    if n == 0:
        return {"polyphony": None, "species_polyphony": None}, {}, {}

    predictions = {
        obj: np.stack([np.array(r["predictions"][obj]) for r in records])
        for obj in records[0]["predictions"]
    }

    y_true = {"polyphony": None, "species_polyphony": None}
    if "polyphony" in records[0]["y_true"]:
        y_true["polyphony"] = np.array([r["y_true"]["polyphony"] for r in records])
    if "min_polyphony" in records[0]["y_true"]:
        y_true["min_polyphony"] = np.array([r["y_true"]["min_polyphony"] for r in records])
    if "max_polyphony" in records[0]["y_true"]:
        y_true["max_polyphony"] = np.array([r["y_true"]["max_polyphony"] for r in records])
        
    if "min_species_polyphony" in records[0]["y_true"]:
        y_true["min_species_polyphony"] = np.stack([
            [r["y_true"]["min_species_polyphony"][sp] for sp in species_names] for r in records
        ])
    if "max_species_polyphony" in records[0]["y_true"]:
        y_true["max_species_polyphony"] = np.stack([
            [r["y_true"]["max_species_polyphony"][sp] for sp in species_names] for r in records
        ])
    if species_names and "species_polyphony" in records[0]["y_true"]:
        y_true["species_polyphony"] = np.stack([
            [r["y_true"]["species_polyphony"][sp] for sp in species_names] for r in records
        ])

    variable_values = {
        var: np.array([r["variable_values"][var] for r in records])
        for var in variables
    }

    return y_true, predictions, variable_values

def flatten_metrics(metrics_dict):
        """
        Flatten compute_polyphony_metrics() output into a flat {row_name: value} dict.
        """
        flat = dict(metrics_dict["overall"])

        if "per_species" in metrics_dict:
            for species, species_metrics in metrics_dict["per_species"].items():
                for metric_name, value in species_metrics.items():
                    flat[f"{species}/{metric_name}"] = value

        return flat


def update_metrics_table(column_name, metrics_dict, csv_path):
    """
    Add or overwrite the column for `subset_name` in the experiment's metrics table.
    Creates the table if it doesn't exist yet.
    """

    flat_metrics = flatten_metrics(metrics_dict)
    new_col = pd.Series(flat_metrics, name=column_name)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        df[column_name] = new_col  # adds new column, or overwrites if it already exists
    else:
        df = new_col.to_frame()

    df.to_csv(csv_path, float_format="%.4f")

    return df, csv_path

def build_update_metrics_table(report, cfg, output_dir, table_name="metrics"):
    study_name = cfg.log.get("study_name", None)
    subset = cfg.dataset.get("subset", "test")

    # Get spatial experiment key for saving metrics to overview table
    exp_key = None
    objectives_set = set(cfg.objectives)
    if cfg.embeddings.dimension_type == "spatial":
        # Check which experiment is being run 
        if objectives_set == {"polyphony_reg"}:
            exp_key = "reg"
        elif objectives_set == {"polyphony_class"}:
            exp_key = "class"
        elif objectives_set == {"polyphony_reg", "polyphony_class"}:
            exp_key = "multi_v0"
        elif objectives_set == {"polyphony_reg", "event_logits"}:
            exp_key = "multi_v1"
        elif objectives_set == {"polyphony_class", "event_logits"}:
            exp_key = "multi_v2"
        elif objectives_set == {"polyphony_reg", "framewise_polyphony_reg"}:
            exp_key = "multi_v3"
        elif objectives_set == {"polyphony_class", "framewise_polyphony_class"}:
            exp_key = "multi_v4"
        elif objectives_set == {"polyphony_reg", "event_logits", "framewise_polyphony_reg"}:
            exp_key = "multi_v5"
        elif objectives_set == {"polyphony_class", "event_logits", "framewise_polyphony_class"}:
            exp_key = "multi_v6"
        else:
            exp_key = "unknown_experiment"

    # Save to metrics overview table
    for objective in report:
        if objective == "polyphony_reg" or objective == "species_polyphony_reg":
            obj_key = "reg"
        elif objective == "polyphony_class" or objective == "species_polyphony_class":
            obj_key = "class"
        elif objective == "event_logits":
            obj_key = "events"
        elif objective == "framewise_polyphony_reg":
            obj_key = "frame_reg"
        elif objective == "framewise_polyphony_class":
            obj_key = "frame_class"
        else:
            obj_key = objective
        
    metrics_dict = report[objective]

    if study_name == "Pooled-Embeddings-SNR-Range-Effects":
        column_name = f"{subset}_{obj_key}_{cfg.dataset.snr_range[0]}-{cfg.dataset.snr_range[1]}"
    elif study_name == "Pooled-Embeddings-Min-SNR-Effects":
        column_name = f"{subset}_{obj_key}_{cfg.dataset.min_snr}"
    elif study_name == "Pooled-Embeddings-Max-Polyphony-Effects":
        column_name = f"{subset}_{obj_key}_{cfg.dataset.max_polyphony}"
    elif exp_key is not None:
        column_name = f"{subset}_{exp_key}_{obj_key}"
    else:
        column_name = f"{subset}_{obj_key}"

    csv_path = os.path.join(output_dir, f"{table_name}.csv")
    _, csv_path = update_metrics_table(column_name, metrics_dict, csv_path)
    print(f"Saved {table_name} to {csv_path}")

