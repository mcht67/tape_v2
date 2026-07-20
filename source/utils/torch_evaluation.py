# torch_evaluation.py
#
# Torch counterpart of utils.evaluation.collect_predictions, used by both
# evaluate_on_test_split.py and evaluate_on_soundscape_data.py. Matches its
# actual output contract and ground-truth resolution logic (checked against
# the real evaluation.py / dataset.py):
#
#   y_true, predictions, variable_values = collect_predictions_torch(...)
#
#   y_true:          {ground_truth_column: np.ndarray}  -- "polyphony",
#                     "min_polyphony", "max_polyphony", "species_polyphony",
#                     "min_species_polyphony", "max_species_polyphony".
#                     Only columns that are actually derivable for the given
#                     dataset are included (checked once per split, not per
#                     example -- HF dataset schemas are uniform).
#   predictions:      {objective_name: np.ndarray}       raw model outputs
#   variable_values:  {variable_name: np.ndarray}        extra columns (e.g. "snr_dB")
#
# Ground-truth resolution mirrors collect_predictions exactly:
#   - "polyphony": prefers the `polyphony` column, falls back to
#     `polyphony_degree` if that's what the dataset has.
#   - "species_polyphony": prefers the `species_polyphony` column; if absent,
#     derives per-species counts from `birdset_code_multilabel` /
#     `birdset_id_multilabel` / `ebird_code_multilabel` via Counter +
#     birdset_id2label, same as dataset.add_species_polyphony.
#   - "min_polyphony" / "max_polyphony" / "min_species_polyphony" /
#     "max_species_polyphony": read directly if present, otherwise omitted.
#
# Unlike collect_predictions (which stacks the *entire* dataset into one
# tensor and does a single forward pass -- its `batch_size` parameter is
# accepted but unused), this batches via DataLoader for memory safety on
# large eval sets. Functionally equivalent, just safer for big datasets.

from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from datasets import Audio
from hydra.utils import instantiate

from utils.torch_models import MultiTaskTemporalCNNHead, MultiTaskSimpleMLPHead


MULTILABEL_COLUMN_CANDIDATES = ("birdset_code_multilabel", "birdset_id_multilabel", "ebird_code_multilabel")


def _resolve_truth_source(dataset, col, birdset_id2label):
    """
    Returns a callable(example) -> value for ground-truth column `col`, or
    None if it can't be derived for this dataset split. Checked once per
    split (dataset.features), not per example.
    """
    features = dataset.features

    if col == "polyphony":
        if "polyphony" in features:
            return lambda ex: ex["polyphony"]
        if "polyphony_degree" in features:
            return lambda ex: ex["polyphony_degree"]
        return None

    if col == "species_polyphony":
        if "species_polyphony" in features:
            return lambda ex: ex["species_polyphony"]
        if birdset_id2label is not None:
            multilabel_col = next((c for c in MULTILABEL_COLUMN_CANDIDATES if c in features), None)
            if multilabel_col is not None:
                ids = list(birdset_id2label.keys())

                def _derive(ex, multilabel_col=multilabel_col, ids=ids):
                    values = ex[multilabel_col]
                    counts = Counter(values) if values is not None else Counter()
                    return [counts.get(int(bid), 0) for bid in ids]

                return _derive
        return None

    # min_polyphony, max_polyphony, min_species_polyphony, max_species_polyphony
    if col in features:
        return lambda ex, col=col: ex[col]
    return None


class _EvalDataset(Dataset):
    def __init__(self, hf_dataset, feature_col, truth_sources, variables):
        self.dataset = hf_dataset
        self.feature_col = feature_col
        self.truth_sources = truth_sources
        self.variables = variables or []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        feat = item[self.feature_col]
        if isinstance(feat, dict) and "array" in feat:
            feat = feat["array"]
        x = torch.tensor(feat, dtype=torch.float32)

        truth = {col: torch.as_tensor(fn(item)) for col, fn in self.truth_sources.items()}
        extra = {v: item[v] for v in self.variables}

        return x, truth, extra


def load_torch_model_for_eval(model_cfg, head_cfg, objectives_cfg, checkpoint_path, device):
    """
    Instantiates the torch backbone from `model_cfg` (same hydra config used
    by torch_train.py), attaches a MultiTaskTemporalCNNHead matching `objectives_cfg`,
    and loads weights from a checkpoint written by ModelAndHistorySaverTorch
    (`{"model_state_dict": ..., ...}`, e.g. checkpoint_dir/best.pt).
    """
    model = instantiate(model_cfg)
    head_input_size = model.get_head_input_size()
    head_cfg.input_size = head_input_size
    head_cfg.objectives_cfg = objectives_cfg
    head = instantiate(head_cfg)
    model.replace_head(head)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

DEFAULT_TRUTH_COLUMNS_BY_OBJECTIVE_PREFIX = {
    "polyphony": "polyphony",
    "species_polyphony": "species_polyphony",
}


def _infer_truth_columns(objectives_list):
    """
    Default ground-truth column per objective, e.g. polyphony_reg/
    polyphony_class -> "polyphony". Pass `truth_columns` explicitly (e.g.
    ["min_polyphony", "max_polyphony", ...] for soundscape range evaluation)
    to override this.
    """
    cols = []
    for obj in objectives_list:
        for prefix, col in DEFAULT_TRUTH_COLUMNS_BY_OBJECTIVE_PREFIX.items():
            if obj.startswith(prefix) and col not in cols:
                cols.append(col)
    return cols


def collect_predictions_torch(
    model,
    dataset,
    input_feature_name,
    objectives_cfg,
    device,
    batch_size=32,
    variables=None,
    truth_columns=None,
    birdset_id2label=None,
    num_workers=2,
    cast_audio_sampling_rate=32000,
):
    """
    Torch counterpart of utils.evaluation.collect_predictions. `dataset` is a
    HuggingFace dataset split with a raw-audio column named
    `input_feature_name` (cast to Audio() here if it isn't already).

    `birdset_id2label` is required to derive `species_polyphony` on the fly
    for datasets that don't have it precomputed (see module docstring) --
    pass the same mapping you already build via get_birdset_id2label().
    """
    objectives_list = list(objectives_cfg.keys())
    truth_columns = truth_columns if truth_columns is not None else _infer_truth_columns(objectives_list)
    variables = variables or []

    if input_feature_name in dataset.features and not isinstance(dataset.features[input_feature_name], Audio):
        dataset = dataset.cast_column(input_feature_name, Audio(sampling_rate=cast_audio_sampling_rate))

    truth_sources = {}
    for col in truth_columns:
        fn = _resolve_truth_source(dataset, col, birdset_id2label)
        if fn is None:
            print(f"[collect_predictions_torch] Could not derive ground-truth column '{col}' for this dataset -- skipping it.")
        else:
            truth_sources[col] = fn

    loader = DataLoader(
        _EvalDataset(dataset, input_feature_name, truth_sources, variables),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=default_collate,
    )

    y_true = {c: [] for c in truth_sources}
    predictions = {obj: [] for obj in objectives_list}
    variable_values = {v: [] for v in variables}

    model.eval()
    with torch.no_grad():
        for x, truth, extra in loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs.logits  # dict {obj_name: tensor}

            for obj in objectives_list:
                predictions[obj].append(preds[obj].cpu().numpy())
            for c in truth_sources:
                y_true[c].append(truth[c].numpy())
            for v in variables:
                variable_values[v].append(np.asarray(extra[v]))

    y_true = {k: np.concatenate(v) for k, v in y_true.items()}
    predictions = {k: np.concatenate(v) for k, v in predictions.items()}
    variable_values = {k: np.concatenate(v) for k, v in variable_values.items()}

    return y_true, predictions, variable_values
