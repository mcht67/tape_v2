# torch_losses.py
#
# PyTorch port of losses.py, keeping the same public shape so torch_train.py
# can mirror train.py almost line-for-line:
#   - DynamicWeightedLoss:            weight-scheduleable loss wrapper
#   - create_losses_from_objectives_torch: builds one DynamicWeightedLoss per objective
#   - LossWeightScheduler / setup_loss_scheduler_torch: piecewise weight schedule,
#     called manually once per epoch (there's no callback lifecycle in a plain loop)
#
# compute_species_count_class_weights is reused unmodified from losses.py --
# it only touches numpy + the HF dataset's .iter(), no TF ops.

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import compute_species_count_class_weights  # noqa: F401  (re-exported)


class DynamicWeightedLoss(nn.Module):
    """Loss with an updateable scalar weight, e.g. for LossWeightScheduler."""

    def __init__(self, base_loss_fn, initial_weight=1.0, name=None):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.name = name
        # buffer (not Parameter) -- follows .to(device) but isn't trained
        self.register_buffer("weight", torch.tensor(float(initial_weight)))

    def forward(self, y_pred, y_true):
        return self.weight * self.base_loss_fn(y_pred, y_true)

    def set_weight(self, value):
        with torch.no_grad():
            self.weight.fill_(float(value))


def make_weighted_sparse_categorical_crossentropy_torch(class_weights):
    """
    Torch equivalent of losses.make_weighted_sparse_categorical_crossentropy.

    y_pred: (..., num_species, num_classes) raw logits
    y_true: (..., num_species) integer class labels
    """
    class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32)

    def loss_fn(y_pred, y_true):
        y_true_long = y_true.long()
        num_classes = y_pred.shape[-1]
        per_element_loss = F.cross_entropy(
            y_pred.reshape(-1, num_classes),
            y_true_long.reshape(-1),
            reduction="none",
        ).reshape(y_true.shape)
        weights = class_weights_tensor.to(y_pred.device)[y_true_long]
        return (per_element_loss * weights).mean()

    return loss_fn


def make_weighted_mse_for_counts_torch(class_weights):
    """
    Torch equivalent of losses.make_weighted_mse_for_counts.

    y_true, y_pred: (..., num_species) regression targets/predictions.
    """
    class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
    num_classes = class_weights_tensor.shape[0]

    def loss_fn(y_pred, y_true):
        y_true = y_true.float()
        squared_error = (y_true - y_pred) ** 2
        y_true_class = torch.clamp(torch.round(y_true).long(), 0, num_classes - 1)
        weights = class_weights_tensor.to(y_pred.device)[y_true_class]
        return (squared_error * weights).mean()

    return loss_fn


def create_losses_from_objectives_torch(objectives, class_weights_by_objective=None):
    """
    Torch equivalent of losses.create_losses_from_objectives.

    Args:
        objectives: cfg.objectives (dict-like), same object used for train.py /
            the Keras model -- objective names must be one of the four
            MultiTaskHead supports: polyphony_reg, polyphony_class,
            species_polyphony_reg, species_polyphony_class.
        class_weights_by_objective: optional {obj_name: np.ndarray} of per-class
            weights (e.g. from compute_species_count_class_weights). If present
            for an objective, overrides the default unweighted loss for it.

    Returns:
        dict[str, DynamicWeightedLoss], keyed by objective name.
    """
    class_weights_by_objective = class_weights_by_objective or {}
    losses = {}

    for obj_name, obj_cfg in objectives.items():
        if obj_name in class_weights_by_objective:
            weights = class_weights_by_objective[obj_name]
            if obj_name.endswith("_class"):
                base_loss = make_weighted_sparse_categorical_crossentropy_torch(weights)
            elif obj_name.endswith("_reg"):
                base_loss = make_weighted_mse_for_counts_torch(weights)
            else:
                raise ValueError(
                    f"Cannot infer weighted loss type for objective '{obj_name}' "
                    "(expected name ending in '_class' or '_reg')"
                )
        elif obj_name in ("polyphony_reg", "species_polyphony_reg", "framewise_polyphony_reg"):
            def base_loss(y_pred, y_true):
                return F.mse_loss(y_pred, y_true.float())
        elif obj_name == "polyphony_class":
            def base_loss(y_pred, y_true):
                return F.cross_entropy(y_pred, y_true.long())
        elif obj_name in ("species_polyphony_class", "framewise_polyphony_class"):
            def base_loss(y_pred, y_true):
                # y_pred: (batch, ..., num_classes), y_true: (batch, ...)
                return F.cross_entropy(
                    y_pred.reshape(-1, y_pred.shape[-1]), y_true.reshape(-1).long()
                )
        elif obj_name == "event_logits":
            def base_loss(y_pred, y_true):
                # y_pred: (batch, time) raw logits, y_true: (batch, time) 0/1
                return F.binary_cross_entropy_with_logits(y_pred, y_true.float())
        else:
            raise ValueError(f"Unknown objective '{obj_name}', no default loss defined")

        weight = obj_cfg.get("weight", 1.0)
        losses[obj_name] = DynamicWeightedLoss(
            base_loss_fn=base_loss,
            initial_weight=weight,
            name=obj_name,
        )

    return losses


class LossWeightScheduler:
    """
    Plain (non-callback) equivalent of losses.LossWeightScheduler. Call
    `on_epoch_begin(epoch)` yourself at the top of each training epoch --
    there's no Keras callback lifecycle to hook into in a manual torch loop.
    """

    def __init__(self, loss_objects, schedule_config):
        self.loss_objects = loss_objects
        self.schedule_config = schedule_config

    def on_epoch_begin(self, epoch):
        updates = {}
        for output_name, schedule in self.schedule_config.items():
            if output_name not in self.loss_objects:
                continue

            switch_epochs = schedule["switch_epochs"]
            weights = schedule["weights"]

            weight = weights[0]
            for i, switch_epoch in enumerate(switch_epochs):
                if epoch >= switch_epoch:
                    weight = weights[i]

            self.loss_objects[output_name].set_weight(weight)
            updates[output_name] = weight

        if updates:
            update_str = ", ".join(f"{k}={v}" for k, v in updates.items())
            print(f"\n[LossWeightScheduler] Epoch {epoch} | {update_str}")


def setup_loss_scheduler_torch(objectives, losses):
    """Torch equivalent of losses.setup_loss_scheduler."""
    schedule_config = {}

    for obj_name, obj_cfg in objectives.items():
        if "scheduler" in obj_cfg:
            schedule_config[obj_name] = obj_cfg["scheduler"]

    if not schedule_config:
        return None

    return LossWeightScheduler(loss_objects=losses, schedule_config=schedule_config)
