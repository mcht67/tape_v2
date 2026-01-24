# losses.py
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable, deserialize_keras_object, serialize_keras_object

# @tf.keras.saving.register_keras_serializable(package="losses")
# class DynamicWeightedLoss(tf.keras.losses.Loss):
#     """Loss with updateable weight for scheduling."""
    
#     def __init__(self, base_loss, initial_weight=1.0, name=None, **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.base_loss = base_loss
#         self.initial_weight = float(initial_weight)
#         self.weight = tf.Variable(
#             self.initial_weight,
#             trainable=False,
#             dtype=tf.float32,
#             name=f"{name}_weight" if name else "loss_weight"
#         )
    
#     def call(self, y_true, y_pred):
#         return self.weight * self.base_loss(y_true, y_pred)
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "base_loss": tf.keras.saving.serialize_keras_object(self.base_loss),
#             "initial_weight": float(self.initial_weight),
#         })
#         return config
    
#     @classmethod
#     def from_config(cls, config):
#         config["base_loss"] = tf.keras.saving.deserialize_keras_object(
#             config["base_loss"]
#         )
#         return cls(**config)

@register_keras_serializable(package="losses")
class DynamicWeightedLoss(tf.keras.losses.Loss):
    """Loss with updateable weight for scheduling."""
    
    def __init__(self, base_loss, initial_weight=1.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_loss = base_loss
        self.initial_weight = float(initial_weight)
        self.weight = tf.Variable(
            self.initial_weight,
            trainable=False,
            dtype=tf.float32,
            name=f"{name}_weight" if name else "loss_weight"
        )
    
    def call(self, y_true, y_pred):
        return self.weight * self.base_loss(y_true, y_pred)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "base_loss": serialize_keras_object(self.base_loss),
            "initial_weight": float(self.initial_weight),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        config["base_loss"] = deserialize_keras_object(config["base_loss"])
        return cls(**config)

# def create_losses_from_model(model, cfg):
#     """Create losses with optional config overrides."""
#     # Get model defaults
#     loss_config = model.get_loss_config()
    
#     losses = {}
    
#     for output_name, loss_specs in loss_config.items():
#         # Start with model defaults
#         merged_spec = loss_specs.copy()
        
#         # Apply overrides from config
#         if hasattr(cfg.loss, 'loss_overrides') and output_name in cfg.loss.loss_overrides:
#             overrides = cfg.loss.loss_overrides[output_name]
#             merged_spec.update(overrides)
        
#         # Extract class and params
#         loss_class = merged_spec["type"]
#         weight = merged_spec["weight"]
#         loss_params = {k: v for k, v in merged_spec.items() 
#                       if k not in ["type", "weight"]}
        
#         # Create loss
#         base_loss = loss_class(**loss_params)
#         losses[output_name] = DynamicWeightedLoss(
#             base_loss=base_loss,
#             initial_weight=weight,
#             name=output_name
#         )
    
#     return losses

# losses.py (cleaner version)
from hydra.utils import instantiate

def create_losses_from_objectives(objectives):
    """
    Create losses directly from objectives config.
    
    Args:
        objectives: Dict from model.objectives containing loss specs
        
    Returns:
        Dict of DynamicWeightedLoss objects
    """
    losses = {}
    
    for obj_name, obj_config in objectives.items():
        # Instantiate base loss via Hydra
        base_loss = instantiate(obj_config["loss"])
        
        # Get weight
        weight = obj_config.get("weight", 1.0)
        
        # Wrap in dynamic weighted loss
        losses[obj_name] = DynamicWeightedLoss(
            base_loss=base_loss,
            initial_weight=weight,
            name=obj_name
        )
    
    return losses

# callbacks.py
import tensorflow as tf

class LossWeightScheduler(tf.keras.callbacks.Callback):
    """Updates loss weights during training based on schedule."""
    
    def __init__(self, loss_objects, schedule_config):
        """
        Args:
            loss_objects: Dict of DynamicWeightedLoss objects
            schedule_config: Dict mapping objective names to schedule dicts
        """
        super().__init__()
        self.loss_objects = loss_objects
        self.schedule_config = schedule_config
    
    def on_epoch_begin(self, epoch, logs=None):
        updates = {}
        
        for output_name, schedule in self.schedule_config.items():
            if output_name not in self.loss_objects:
                continue
            
            switch_epochs = schedule["switch_epochs"]
            weights = schedule["weights"]
            
            # Find weight for this epoch
            weight = weights[0]
            for i, switch_epoch in enumerate(switch_epochs):
                if epoch >= switch_epoch:
                    weight = weights[i]
            
            # Update
            self.loss_objects[output_name].weight.assign(weight)
            updates[output_name] = weight
        
        if updates:
            update_str = ", ".join([f"{k}={v}" for k, v in updates.items()])
            print(f"\n[LossWeightScheduler] Epoch {epoch} | {update_str}")


def setup_loss_scheduler(objectives, losses):
    """
    Create scheduler from objectives config.
    
    Args:
        objectives: Dict from model.objectives
        losses: Dict of DynamicWeightedLoss objects
        
    Returns:
        LossWeightScheduler or None
    """
    schedule_config = {}
    
    # Extract scheduler configs from objectives
    for obj_name, obj_config in objectives.items():
        if "scheduler" in obj_config:
            schedule_config[obj_name] = obj_config["scheduler"]
    
    if not schedule_config:
        return None
    
    return LossWeightScheduler(
        loss_objects=losses,
        schedule_config=schedule_config
    )
# class LossWeightScheduler(tf.keras.callbacks.Callback):
#     """
#     Updates loss weights during training.
    
#     Args:
#         loss_objects: Dict mapping output names to DynamicWeightedLoss objects
#         schedule_config: Dict mapping output names to their schedules
#             Example:
#             {
#                 "perch2_event_logits": {
#                     "switch_epochs": [0, 10, 20, 30, 40],
#                     "weights": [10.0, 5.0, 1.0, 0.5, 0.1]
#                 },
#                 ...
#             }
#     """
#     def __init__(self, loss_objects, schedule_config):
#         super().__init__()
#         self.loss_objects = loss_objects
#         self.schedule_config = schedule_config
    
#     def on_epoch_begin(self, epoch, logs=None):
#         updates = {}
        
#         for output_name, schedule in self.schedule_config.items():
#             if output_name not in self.loss_objects:
#                 continue
            
#             switch_epochs = schedule["switch_epochs"]
#             weights = schedule["weights"]
            
#             # Find the weight for this epoch
#             weight = weights[0]  # default to first weight
#             for i, switch_epoch in enumerate(switch_epochs):
#                 if epoch >= switch_epoch:
#                     weight = weights[i]
            
#             # Update the loss weight
#             self.loss_objects[output_name].weight.assign(weight)
#             updates[output_name] = weight
        
#         if updates:
#             update_str = ", ".join([f"{k}={v}" for k, v in updates.items()])
#             print(f"\n[LossWeightScheduler] Epoch {epoch} | {update_str}")

# def setup_loss_scheduler(losses, cfg):
#     """Create scheduler from config."""

#     if not hasattr(cfg.loss, 'scheduler'):
#         return None
    
#     schedule_config = {}
    
#     # Build schedule config from cfg
#     for output_name in losses.keys():
#         if output_name in cfg.loss.scheduler:
#             schedule_config[output_name] = {
#                 "switch_epochs": cfg.loss.scheduler[output_name].switch_epochs,
#                 "weights": cfg.loss.scheduler[output_name].weights
#             }
    
#     if not schedule_config:
#         return None
    
#     return LossWeightScheduler(
#         loss_objects=losses,
#         schedule_config=schedule_config
#     )