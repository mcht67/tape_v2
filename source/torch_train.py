import os
import json
import time
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from datasets import load_from_disk, Audio
from omegaconf import OmegaConf
from hydra.utils import instantiate
from dotenv import load_dotenv
import shutil

from utils.general import get_num_workers
from utils.config import set_random_seeds, Params
from utils.dataset import add_labels, get_birdset_id2label, get_local_data_dir, load_dataset_with_retry, filter_dataset_by_polyphony_and_snr

from utils.logs import get_log_paths, get_archive_paths, build_confusion_matrix_specs
from utils.metrics import compute_polyphony_metrics, prepare_event_logits_for_cm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.torch_losses import (
    compute_species_count_class_weights,
    create_losses_from_objectives_torch,
    setup_loss_scheduler_torch,
)
from utils.torch_logging import TorchSummaryWriterLogger, ModelAndHistorySaverTorch
from utils.torch_models import MultiTaskTemporalCNNHead, MultiTaskSimpleMLPHead

# Disable caching to avoid huggingface caching issues when running multiple experiments in parallel
# from datasets import disable_caching
# disable_caching()


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, feature_col, objective_names):
        # No with_format("torch") -- datasets' torch formatter has a bug
        # where it unconditionally imports torchvision.io.VideoReader on
        # any tensorize call if torchvision is installed, even for
        # non-video data. Convert to tensors ourselves instead.
        keep_cols = {feature_col, *objective_names}
        drop_cols = [c for c in hf_dataset.column_names if c not in keep_cols]
        self.dataset = hf_dataset.remove_columns(drop_cols) if drop_cols else hf_dataset
        self.feature_col = feature_col
        self.objective_names = objective_names

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, indices):
        batch = self.dataset[indices]  # dict of lists/np arrays, keyed by column

        feature_tensor = torch.as_tensor(np.asarray(batch[self.feature_col]), dtype=torch.float32)
        labels = {
            obj_name: torch.as_tensor(
                np.asarray(batch[obj_name]),
                dtype=torch.long if obj_name.endswith("_class") else torch.float32,
            )
            for obj_name in self.objective_names
        }
        return feature_tensor, labels

    def __getitem__(self, idx):
        item = self.dataset[idx]
        feature_tensor = torch.as_tensor(np.asarray(item[self.feature_col]), dtype=torch.float32)
        labels = {
            obj_name: torch.as_tensor(
                item[obj_name],
                dtype=torch.long if obj_name.endswith("_class") else torch.float32,
            )
            for obj_name in self.objective_names
        }
        return feature_tensor, labels


def get_torch_dataloaders(dataset, feature_col, objective_names, batch_size, num_workers=2):
    loaders = {}
    for split, shuffle in (("train", True), ("validation", False), ("test", False)):
        ds = HFDatasetWrapper(dataset[split], feature_col, objective_names)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
            # No collate_fn: __getitems__ already returns properly batched,
            # stacked tensors, so default_collate has nothing left to do.
        )
    return loaders["train"], loaders["test"], loaders["validation"]

# class HFDatasetWrapper(Dataset):
#     """
#     Wraps a HuggingFace dataset split for multi-objective training. Returns
#     (feature_tensor, {objective_name: label_tensor, ...}) per item, matching
#     what train.py's `to_tf_dataset(columns=..., label_cols=labels)` produced,
#     just torch-side.
#     """

#     def __init__(self, hf_dataset, feature_col, objective_names):
#         self.dataset = hf_dataset
#         self.feature_col = feature_col
#         self.objective_names = objective_names

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]

#         feat = item[self.feature_col]
#         if isinstance(feat, dict) and "array" in feat:
#             feat = feat["array"]
#         feature_tensor = torch.tensor(feat, dtype=torch.float32)

#         labels = {}
#         for obj_name in self.objective_names:
#             value = item[obj_name]
#             dtype = torch.long if obj_name.endswith("_class") else torch.float32
#             labels[obj_name] = torch.tensor(value, dtype=dtype)

#         return feature_tensor, labels


# def collate_fn(batch):
#     """Default collate handles nested dicts fine, this just documents intent."""
#     return default_collate(batch)


# def get_torch_dataloaders(dataset, feature_col, objective_names, batch_size, num_workers=2):
#     loaders = {}
#     for split, shuffle in (("train", True), ("validation", False), ("test", False)):
#         ds = HFDatasetWrapper(dataset[split], feature_col, objective_names)
#         loaders[split] = DataLoader(
#             ds, batch_size=batch_size, shuffle=shuffle,
#             num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
#         )
#     return loaders["train"], loaders["test"], loaders["validation"]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

FRAMEWISE_OBJECTIVES = {"event_logits", "framewise_polyphony_reg", "framewise_polyphony_class"}


def compute_epoch_metrics(obj_name, y_pred, y_true, num_classes=None):
    """
    Runs metrics.compute_polyphony_metrics (or, for event_logits, plain sklearn
    binary metrics) on the full-epoch predictions for a single objective,
    returning a flat {metric_name: value} dict -- written as
    val_{obj_name}_{metric_name}.
    """
    if obj_name == "event_logits":
        # frame-wise binary detection logits, flattened over (batch, time)
        yt, yp = prepare_event_logits_for_cm(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        return {
            "accuracy": accuracy_score(yt, yp),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    is_species = obj_name.startswith("species_")
    is_framewise = obj_name.startswith("framewise_")
    is_class = obj_name.endswith("_class")
    cm_type = "classification" if is_class else "regression_round"

    if is_species:
        yt, yp = y_true, y_pred
    elif is_framewise:
        # (batch, time[, classes]) -> flatten the time axis, then treat each
        # frame like an independent "species" entry of a length-1 batch.
        if is_class:
            yt = y_true.reshape(-1, 1)
            yp = y_pred.reshape(-1, 1, y_pred.shape[-1])
        else:
            yt = y_true.reshape(-1, 1)
            yp = y_pred.reshape(-1, 1)
    else:
        yt = y_true[:, None]
        yp = y_pred[:, None, :] if is_class else y_pred[:, None]

    result = compute_polyphony_metrics(yt, yp, cm_type=cm_type, num_classes=num_classes, per_species=False)
    return result["overall"]


def cm_spec_type_for_objective(obj_name):
    """type string expected by build_confusion_matrix_specs / TorchSummaryWriterLogger."""
    if obj_name == "event_logits":
        return "binary"
    is_species = obj_name.startswith("species_")
    is_class = obj_name.endswith("_class")
    if is_species and is_class:
        return "species_classification"
    if is_species:
        return "species_regression_round"
    if is_class:
        return "classification"
    return "regression_round"


# ----------------------------------------------------------------------------
# Early stopping (manual equivalent of tf.keras.callbacks.EarlyStopping)
# ----------------------------------------------------------------------------

class EarlyStopper:
    def __init__(self, patience=10, start_from_epoch=0):
        self.patience = patience
        self.start_from_epoch = start_from_epoch if start_from_epoch is not None else 0
        self.best = float("inf")
        self.wait = 0
        self.should_stop = False

    def step(self, epoch, val_loss):
        if epoch < self.start_from_epoch or val_loss is None:
            return
        if val_loss < self.best:
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True

def encoder_lr_schedule(epoch, freeze_epochs, target_lr,
                         warmup_epochs=3, decay_epochs=15, min_lr_ratio=0.0):
    """
    Encoder LR as a pure function of the global epoch index. Only meaningful
    once epoch >= freeze_epochs (the encoder is frozen -- and gets no
    gradient -- before that, so the returned value there is unused).

    - First `warmup_epochs` after unfreezing: linear ramp 0 -> target_lr.
    - After that: cosine decay from target_lr down to target_lr * min_lr_ratio
      over `decay_epochs`, then held flat at the floor.
    """
    if epoch < freeze_epochs:
        return target_lr  # unused while frozen (requires_grad=False)

    since_unfreeze = epoch - freeze_epochs
    if since_unfreeze < warmup_epochs:
        return target_lr * (since_unfreeze + 1) / warmup_epochs

    t = min(since_unfreeze - warmup_epochs, decay_epochs)
    cos_factor = 0.5 * (1 + math.cos(math.pi * t / decay_epochs))
    min_lr = target_lr * min_lr_ratio
    return min_lr + (target_lr - min_lr) * cos_factor


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ###################################################
    # Configuration
    ###################################################
    cfg = OmegaConf.load("params.yaml")

    os.environ.setdefault("DEFAULT_DIR", os.getcwd())
    os.environ.setdefault("DVC_EXP_NAME", "test-experiment")

    random_seed = cfg.general.random_seed
    set_random_seeds(random_seed)
    params = Params()

    input_feature = cfg.train.get("input_feature", "audio")
    print(f"input_feature: {input_feature}")
    input_feature_name = cfg.train.get("input_feature_name") 
    input_feature_name = input_feature if not input_feature_name else input_feature_name
    print(f"Using input feature column '{input_feature_name}' (cfg.train.input_feature_name or cfg.train.input_feature)")
    precomputed_embeddings = cfg.train.get("precomputed_embeddings", False)
    total_epochs = cfg.train.epochs
    initial_epoch = cfg.train.get("initial_epoch", 0) or 0
    learning_rate = cfg.train.learning_rate
    batch_size = cfg.train.get("batch_size", 32)
    early_stopping_patience = cfg.train.get("early_stopping_patience", 10)
    early_stopping_delay_epochs = cfg.train.get("early_stopping_delay_epochs", 0)
    num_batches_train = cfg.train.get("num_batches_train", None)
    num_batches_val = cfg.train.get("num_batches_val", None)

    freeze_epochs = cfg.train.get("freeze_epochs", None)
    finetune_learning_rate = cfg.train.get("finetune_learning_rate", None)
    finetune_warmup_epochs = cfg.train.get("finetune_warmup_epochs", 3)
    finetune_decay_epochs = cfg.train.get("finetune_decay_epochs", 20)

    huggingface_path = cfg.dataset.huggingface_path
    load_checkpoint_path = cfg.train.get("load_checkpoint_path", None)

    objectives_cfg = cfg.objectives
    objectives_list = list(objectives_cfg.keys())
    model_cfg = cfg.model
    model_cfg.pop("name", None)
    head_cfg = cfg.head

    log_paths = get_log_paths(cfg)
    log_dir = str(log_paths["train_log_dir"])
    checkpoint_dir = str(log_paths["checkpoint_dir"])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Logging to {log_dir}, checkpoints to {checkpoint_dir}")

    archive_paths = get_archive_paths(cfg)
    archive_log_dir = archive_paths['train_log_dir']
    checkpoint_archive_dir = archive_paths['checkpoint_dir']
    os.makedirs(archive_log_dir, exist_ok=True)
    os.makedirs(checkpoint_archive_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ###################################################
    # Dataset
    ###################################################
    default_dir = os.environ.get("DEFAULT_DIR", "")
    train_config = cfg.dataset.train_config
    subset = cfg.dataset.get("subset", None)
    local_data_dir = get_local_data_dir(dataset_config=train_config, subset=subset)
    dataset_dir = os.path.join(default_dir, local_data_dir)
    print("dataset_dir:", dataset_dir)

    # Load huggingface token from .env file
    load_dotenv('local.env')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

    if not os.path.exists(local_data_dir):
        print(f"Dataset {train_config} not found locally. Downloading from Huggingface...")
        dataset = load_dataset_with_retry(huggingface_path, train_config, token=huggingface_token) #, download_mode='force_redownload')
    else:
        print(f"Dataset {train_config} found locally. Loading from disk: {local_data_dir}...")
        dataset = load_from_disk(local_data_dir)

    # # DEBUG, TODO: remove this after debugging
    # for split in dataset.keys():
    #     dataset[split] = dataset[split].select(range(10))

    num_workers = get_num_workers(gb_per_worker=5, cpu_percentage=0.8)
    dataset = filter_dataset_by_polyphony_and_snr(dataset, cfg, num_workers=num_workers)

    # # Filter dataset by polyphony degree if specified in the config
    # if 'max_polyphony' in cfg.dataset and cfg.dataset.max_polyphony is not None:
    #     max_polyphony = cfg.dataset.max_polyphony
    #     print(f"Filtering dataset to include only examples with polyphony degree <= {max_polyphony}...")
    #     for split in dataset.keys():
    #         if 'polyphony' not in dataset[split].column_names and 'polyphony_degree' in dataset[split].column_names:
    #             dataset[split] = dataset[split].rename_column('polyphony_degree', 'polyphony')
    #         dataset[split] = dataset[split].filter(lambda polyphony: [p <= max_polyphony for p in polyphony], input_columns=['polyphony'], batched=True, num_proc=num_workers, batch_size=100)
    #         print(f"After filtering, {split} split has {len(dataset[split])} examples.")

    # # Filter dataset by SNR if specified in the config
    # if 'snr_range' in cfg.dataset and cfg.dataset.snr_range is not None:
    #     snr_range = cfg.dataset.snr_range
    #     print(f"Filtering dataset to include only examples with SNR in range {snr_range}...")
    #     for split in dataset.keys():
    #         dataset[split] = dataset[split].filter(lambda snr: [snr_range[0] <= s <= snr_range[1] for s in snr], input_columns=['snr_dB'], batched=True, num_proc=num_workers, batch_size=100)
    #         print(f"After filtering, {split} split has {len(dataset[split])} examples.")

    # if 'min_snr' in cfg.dataset and cfg.dataset.min_snr is not None:
    #     min_snr = cfg.dataset.min_snr
    #     print(f"Filtering dataset to include only examples with SNR >= {min_snr}...")
    #     for split in dataset.keys():
    #         dataset[split] = dataset[split].filter(lambda snr: [s >= min_snr for s in snr], input_columns=['snr_dB'], batched=True, num_proc=num_workers, batch_size=100)
    #         print(f"After filtering, {split} split has {len(dataset[split])} examples.")

    # Species-level objectives need num_species / a birdset id<->label mapping,
    # same as train.py.
    birdset_id2label = None
    num_species = None
    mapping = None
    if "species_polyphony_reg" in objectives_cfg or "species_polyphony_class" in objectives_cfg:
        birdset_id2label = get_birdset_id2label(subset, dataset=dataset)
        num_species = len(birdset_id2label)
        mapping = {i: (bird_id, birdset_id2label[bird_id]) for i, bird_id in enumerate(birdset_id2label)}
        with open(os.path.join(log_dir, "species_polyphony_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)

    num_classes = cfg.dataset.max_polyphony + 1 if "max_polyphony" in cfg.dataset else None
    if "polyphony_class" in objectives_cfg:
        objectives_cfg.polyphony_class.num_classes = num_classes
    if "framewise_polyphony_class" in objectives_cfg:
        objectives_cfg.framewise_polyphony_class.num_classes = num_classes
    if "species_polyphony_reg" in objectives_cfg:
        objectives_cfg.species_polyphony_reg.num_species = num_species
    if "species_polyphony_class" in objectives_cfg:
        objectives_cfg.species_polyphony_class.num_classes = num_classes
        objectives_cfg.species_polyphony_class.num_species = num_species

    labels = list(objectives_cfg.keys())

    ###################################################
    # Model (instantiated *before* add_labels: framewise_polyphony_*/event_logits
    # need time_dim/freq_dim, which -- unlike train.py's precomputed-embedding
    # pipeline -- we can only get by actually running a sample through the
    # pretrained encoder, since torch_train.py fine-tunes from raw audio.)
    ###################################################
    print(model_cfg)
    model = instantiate(model_cfg)
    head_input_size = int(model.get_head_input_size())
    head_cfg.input_size = head_input_size
    head_cfg.objectives_cfg = objectives_cfg 
    print("Head config:", head_cfg)
    head = instantiate(head_cfg)
    model.replace_head(head)

    default_sampling_rate = 32000
    if not model.sampling_rate:
        model.set_sampling_rate(default_sampling_rate)

    needs_frame_dims = bool({"framewise_polyphony_reg", "framewise_polyphony_class", "event_logits"} & set(labels))
    time_dim, freq_dim = None, None
    if needs_frame_dims:
        if precomputed_embeddings:
            sample_feat = np.asarray(dataset["train"][0][input_feature_name])
            time_dim = sample_feat.shape[0]
            freq_dim = sample_feat.shape[1] if sample_feat.ndim == 3 else None
        else:
            model.eval()
            with torch.no_grad():
                sample_feat = dataset["train"][0][input_feature_name]
                if isinstance(sample_feat, dict) and "array" in sample_feat:
                    sample_feat = sample_feat["array"]
                sample_audio = torch.tensor(sample_feat, dtype=torch.float32).unsqueeze(0)
                sample_outputs = model(sample_audio)
                spatial_shape = sample_outputs.spatial_embeddings.shape
                time_dim = spatial_shape[1]
                freq_dim = spatial_shape[2] if len(spatial_shape) == 4 else None

    # if labels != ["polyphony_reg"] and labels != ["polyphony_class"]:
    dataset, added_labels = add_labels(
        dataset, labels, birdset_id2label=birdset_id2label,
        time_dim=time_dim, freq_dim=freq_dim,
    )
    print("Added labels:", added_labels)

    sampling_rate = model.get_sampling_rate()
    if not precomputed_embeddings:

        def _extract_waveform(batch, feature_name=input_feature_name):
            batch[feature_name] = [
                np.asarray(item["array"], dtype=np.float32) for item in batch[feature_name]
            ]
            return batch

        for split in dataset:
            if 'sources_audio' in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns(['sources_audio'])
            dataset[split] = dataset[split].cast_column(input_feature_name, Audio(sampling_rate=sampling_rate))
            dataset[split] = dataset[split].map(
                _extract_waveform, batched=True, num_proc=num_workers, batch_size=100
            )

    # Drop any columns that aren't the input feature or one of the objectives
    keep_cols = {input_feature_name, *objectives_list}
    for split in dataset:
        drop_cols = [c for c in dataset[split].column_names if c not in keep_cols]
        if drop_cols:
            dataset[split] = dataset[split].remove_columns(drop_cols)
      

    train_loader, test_loader, val_loader = get_torch_dataloaders(
        dataset=dataset, feature_col=input_feature_name,
        objective_names=objectives_list, batch_size=batch_size, num_workers=num_workers
    ) 

    freeze_encoder = cfg.train.get("freeze_encoder", True)
    if freeze_encoder:
        print("Freezing encoder parameters (only training head).")
        model.freeze_encoder()
    model.to(device)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    ###################################################
    # Losses
    ###################################################
    class_weights_by_objective = {}
    if "species_polyphony_class" in objectives_cfg:
        class_weights_by_objective["species_polyphony_class"] = compute_species_count_class_weights(
            dataset["train"], label_column="species_polyphony_class", num_classes=num_classes,
        )
    if "species_polyphony_reg" in objectives_cfg:
        class_weights_by_objective["species_polyphony_reg"] = compute_species_count_class_weights(
            dataset["train"], label_column="species_polyphony_reg", num_classes=num_classes,
        )

    losses = create_losses_from_objectives_torch(objectives_cfg, class_weights_by_objective)
    for loss_obj in losses.values():
        loss_obj.to(device)

        
    freeze_epochs = cfg.train.get("freeze_epochs", None)              # e.g. 5; None = old behavior (frozen forever)
    finetune_learning_rate = cfg.train.get("finetune_learning_rate", None)  # e.g. 1e-5; None = reuse `learning_rate`

    if freeze_encoder and freeze_epochs is not None:
        head_params = [p for p in model.parameters() if p.requires_grad]
        encoder_params = [p for p in model.parameters() if not p.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": head_params, "lr": learning_rate, "name": "head"},
            {"params": encoder_params, "lr": finetune_learning_rate or learning_rate, "name": "encoder"},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    encoder_param_group = None
    if freeze_encoder and freeze_epochs is not None:
        encoder_param_group = next(g for g in optimizer.param_groups if g.get("name") == "encoder")

    loss_weight_scheduler = setup_loss_scheduler_torch(objectives_cfg, losses)

    ###################################################
    # Resume
    ###################################################
    previous_history = None
    if load_checkpoint_path and os.path.isfile(load_checkpoint_path):
        print(f"Loading checkpoint from {load_checkpoint_path}")
        ckpt = torch.load(load_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for obj_name, w in ckpt.get("loss_weights", {}).items():
            if obj_name in losses:
                losses[obj_name].set_weight(w)
        initial_epoch = ckpt["epoch"] + 1

        history_path = os.path.join(os.path.dirname(load_checkpoint_path), "train_history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                previous_history = json.load(f)
            print(f"Loaded history, resuming from epoch {initial_epoch}")

    ###################################################
    # Logging
    ###################################################
    confusion_matrix_specs = build_confusion_matrix_specs(objectives_cfg)
    for spec in confusion_matrix_specs:
        spec.setdefault("type", cm_spec_type_for_objective(spec["name"]))
        if spec["name"] in ("species_polyphony_reg", "species_polyphony_class") and mapping:
            spec["species_mapping"] = mapping
        if spec["name"] in ("polyphony_class", "species_polyphony_class"):
            spec.setdefault("num_classes", num_classes)

    metric_keys = ["val_loss"] + [f"val_{obj}_loss" for obj in objectives_list] + \
                  [f"val_{obj}_accuracy" for obj in objectives_list]

    logger = TorchSummaryWriterLogger(
        log_dir=log_dir, params=params, metric_keys=metric_keys, loss_objects=losses,
        cfg=cfg, confusion_matrix_specs=confusion_matrix_specs, confusion_matrix_frequency=1,
        log_confusion_matrix=True, previous_history=previous_history,
    )
    checkpoint_saver = ModelAndHistorySaverTorch(
        checkpoint_dir=checkpoint_dir, loss_objects=losses, previous_history=previous_history,
        keep_last_n=cfg.train.get("keep_last_n_checkpoints", 5),
    )
    early_stopper = EarlyStopper(patience=early_stopping_patience, start_from_epoch=early_stopping_delay_epochs)

    ###################################################
    # Train loop
    ###################################################
    encoder_unfrozen = False

    for epoch in range(initial_epoch, total_epochs):
        if freeze_encoder and freeze_epochs is not None and not encoder_unfrozen and epoch >= freeze_epochs:
            print(f"Epoch {epoch + 1}: unfreezing encoder for fine-tuning"
                + (f" (encoder lr={finetune_learning_rate})" if finetune_learning_rate else ""))
            for p in model.parameters():
                p.requires_grad = True
            encoder_unfrozen = True

        if encoder_param_group is not None and epoch >= freeze_epochs:
            target_lr = finetune_learning_rate or learning_rate
            new_lr = encoder_lr_schedule(
                epoch, freeze_epochs, target_lr,
                warmup_epochs=finetune_warmup_epochs,
                decay_epochs=finetune_decay_epochs
            )
            encoder_param_group["lr"] = new_lr
            print(f"Epoch {epoch + 1}: encoder lr = {new_lr:.2e}")

        print(f"Epoch {epoch + 1}\n-------------------------------")
        epoch_start = time.time()

        if loss_weight_scheduler:
            loss_weight_scheduler.on_epoch_begin(epoch)

        # ---- train ----
        model.train()
        train_loss_sums = {obj: 0.0 for obj in objectives_list}
        train_total_loss_sum = 0.0
        n_train_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if num_batches_train and batch_idx >= num_batches_train:
                break

            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}

            outputs = model(x)
            preds = outputs.logits  # dict {obj_name: tensor}

            total_loss = 0.0
            for obj_name in objectives_list:
                obj_loss = losses[obj_name](preds[obj_name], y[obj_name])
                total_loss = total_loss + obj_loss
                train_loss_sums[obj_name] += float(obj_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_total_loss_sum += float(total_loss.item())
            n_train_batches += 1

        n_train_batches = max(n_train_batches, 1)

        # ---- validate ----
        model.eval()
        val_loss_sums = {obj: 0.0 for obj in objectives_list}
        val_total_loss_sum = 0.0
        n_val_batches = 0
        val_preds_all = {obj: [] for obj in objectives_list}
        val_true_all = {obj: [] for obj in objectives_list}

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if num_batches_val and batch_idx >= num_batches_val:
                    break

                x = x.to(device)
                y = {k: v.to(device) for k, v in y.items()}

                outputs = model(x)
                preds = outputs.logits

                total_loss = 0.0
                for obj_name in objectives_list:
                    obj_loss = losses[obj_name](preds[obj_name], y[obj_name])
                    total_loss = total_loss + obj_loss
                    val_loss_sums[obj_name] += float(obj_loss.item())
                    val_preds_all[obj_name].append(preds[obj_name].cpu().numpy())
                    val_true_all[obj_name].append(y[obj_name].cpu().numpy())

                val_total_loss_sum += float(total_loss.item())
                n_val_batches += 1

        n_val_batches = max(n_val_batches, 1)

        val_predictions = {
            obj: (np.concatenate(val_preds_all[obj]), np.concatenate(val_true_all[obj]))
            for obj in objectives_list
        }

        ###################################################
        # Assemble logs dict (mirrors Keras `logs`)
        ###################################################
        logs = {
            "loss": train_total_loss_sum / n_train_batches,
            "val_loss": val_total_loss_sum / n_val_batches,
        }
        for obj_name in objectives_list:
            logs[f"{obj_name}_loss"] = train_loss_sums[obj_name] / n_train_batches
            logs[f"val_{obj_name}_loss"] = val_loss_sums[obj_name] / n_val_batches

            y_pred, y_true = val_predictions[obj_name]
            obj_num_classes = getattr(objectives_cfg[obj_name], "num_classes", None) if obj_name.endswith("_class") else None
            epoch_metrics = compute_epoch_metrics(obj_name, y_pred, y_true, num_classes=obj_num_classes)
            for metric_name, value in epoch_metrics.items():
                if metric_name == "support":
                    continue
                logs[f"val_{obj_name}_{metric_name}"] = float(value) if value == value else 0.0  # nan guard

        print(f"Epoch {epoch + 1}/{total_epochs} done in {time.time() - epoch_start:.1f}s | "
              f"loss={logs['loss']:.4f} val_loss={logs['val_loss']:.4f}")

        ###################################################
        # Logging / checkpointing / early stopping
        ###################################################
        logger.log_epoch(epoch, logs, val_predictions=val_predictions)
        checkpoint_saver.on_epoch_end(epoch, logs, model, optimizer)

        early_stopper.step(epoch, logs.get("val_loss"))
        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1} "
                  f"(no val_loss improvement for {early_stopping_patience} epochs).")
            break

    logger.close()

    # Copy log files and subfolders to archive directory for later analysis
    if os.path.isdir(log_dir):
        print(f"Copying log files and subfolders from {log_dir} to archive directory {archive_log_dir}...")
        shutil.copytree(log_dir, archive_log_dir, dirs_exist_ok=True)
    else:
        print(f"Archive log directory {archive_log_dir} or log directory {log_dir} does not exist. Skipping copy.")

    if os.path.isdir(checkpoint_dir):
        print(f"Copying checkpoint files and subfolders from {checkpoint_dir} to archive directory {checkpoint_archive_dir}...")
        shutil.copytree(checkpoint_dir, checkpoint_archive_dir, dirs_exist_ok=True)
    else:
        print(f"Checkpoint archive directory {checkpoint_archive_dir} or checkpoint directory {checkpoint_dir} does not exist. Skipping copy.")

if __name__ == "__main__":
    main()
