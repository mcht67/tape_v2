from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Union
from transformers import EfficientNetForImageClassification, AutoFeatureExtractor, AutoModel 
from transformers.modeling_outputs import ModelOutput

import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms
import warnings

#from birdset.datamodule.components.augmentations import PowerToDB

##################################
# Model heads
################################## 

class SimpleRegressionHead(torch.nn.Module):
    def __init__(self, input_size: Union[int, torch.Size], dropout: float = 0.2):
        super().__init__()
        if isinstance(input_size, torch.Size):
            if len(input_size) > 2:
                raise ValueError(
                    f"input_size must have at most 2 dimensions, got {len(input_size)} "
                    f"dimensions: {input_size}"
                )
            feature_dim = input_size[-1]
        elif isinstance(input_size, (tuple, list)):
            if len(input_size) > 2:
                raise ValueError(
                    f"input_size must have at most 2 dimensions, got {len(input_size)} "
                    f"dimensions: {input_size}"
                )
            feature_dim = input_size[-1]
        else:
            feature_dim = input_size

        self.regression_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.regression_head(x)

class SimpleMLPHead(nn.Module):
    """
    Literal 1:1 port of model.py's SimpleMLP (event_logits/framewise_*
    removed version) -- always pooled-only, no dual pooled/spatial handling.
    """

    SUPPORTED = {"polyphony_reg", "polyphony_class", "species_polyphony_reg", "species_polyphony_class"}

    # No takes_spatial_embeddings attribute at all -- defaults to False via
    # getattr(...) in the wrapper patches, so this always gets pooled_embeddings,
    # exactly like SimpleRegressionHead.

    def __init__(self, input_size, objectives_cfg, hidden_units=(512, 256), dropout_rate=0.3):
        super().__init__()

        if isinstance(input_size, (torch.Size, tuple, list)):
            feature_dim = input_size[-1]
        else:
            feature_dim = input_size

        self.in_features = feature_dim

        unsupported = set(objectives_cfg.keys()) - self.SUPPORTED
        if unsupported:
            raise ValueError(f"Unknown objective(s) {unsupported}. Supported: {sorted(self.SUPPORTED)}")

        self.objectives_cfg = dict(objectives_cfg)
        self.hidden_units = list(hidden_units)

        # shared trunk -- dropout only after the first hidden layer, same as SimpleMLP
        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        prev_dim = feature_dim
        for i, units in enumerate(self.hidden_units):
            self.hidden_layers.append(nn.Linear(prev_dim, units))
            self.dropout_layers.append(nn.Dropout(dropout_rate) if i == 0 else nn.Identity())
            prev_dim = units
        trunk_dim = prev_dim

        if "polyphony_reg" in self.objectives_cfg:
            self.polyphony_reg_head = nn.Linear(trunk_dim, 1)

        if "polyphony_class" in self.objectives_cfg:
            self.polyphony_num_classes = self.objectives_cfg["polyphony_class"].get("num_classes", 9)
            self.polyphony_class_head = nn.Linear(trunk_dim, self.polyphony_num_classes)

        if "species_polyphony_reg" in self.objectives_cfg:
            self.num_species_reg = self.objectives_cfg["species_polyphony_reg"]["num_species"]
            self.species_polyphony_reg_dense1 = nn.Linear(trunk_dim, 256)
            self.species_polyphony_reg_dropout = nn.Dropout(dropout_rate)
            self.species_polyphony_reg_dense2 = nn.Linear(256, 128)
            self.species_polyphony_reg_head = nn.Linear(128, self.num_species_reg)

        if "species_polyphony_class" in self.objectives_cfg:
            self.num_classes_global = self.objectives_cfg["species_polyphony_class"]["num_classes"]
            self.num_species_global = self.objectives_cfg["species_polyphony_class"]["num_species"]
            self.species_polyphony_class_dense1 = nn.Linear(trunk_dim, 256)
            self.species_polyphony_class_dropout = nn.Dropout(dropout_rate)
            self.species_polyphony_class_dense2 = nn.Linear(256, 128)
            self.species_polyphony_class_head = nn.Linear(
                128, self.num_species_global * self.num_classes_global
            )

    def forward(self, x):
        # mirrors SimpleMLP's self.flatten(inputs) -- no-op on an already
        # pooled (batch, feature) vector
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)

        for linear, dropout in zip(self.hidden_layers, self.dropout_layers):
            x = torch.relu(linear(x))
            x = dropout(x)
        features = x

        outputs = {}

        if "polyphony_reg" in self.objectives_cfg:
            outputs["polyphony_reg"] = self.polyphony_reg_head(features).squeeze(-1)

        if "polyphony_class" in self.objectives_cfg:
            outputs["polyphony_class"] = self.polyphony_class_head(features)

        if "species_polyphony_reg" in self.objectives_cfg:
            h = torch.relu(self.species_polyphony_reg_dense1(features))
            h = self.species_polyphony_reg_dropout(h)
            h = torch.relu(self.species_polyphony_reg_dense2(h))
            outputs["species_polyphony_reg"] = self.species_polyphony_reg_head(h)

        if "species_polyphony_class" in self.objectives_cfg:
            h = torch.relu(self.species_polyphony_class_dense1(features))
            h = self.species_polyphony_class_dropout(h)
            h = torch.relu(self.species_polyphony_class_dense2(h))
            h = self.species_polyphony_class_head(h)
            outputs["species_polyphony_class"] = h.view(-1, self.num_species_global, self.num_classes_global)

        return outputs

class MultiTaskTemporalCNNHead(nn.Module):
    """
    objectives_cfg: same shape as cfg.objectives / model.py's TemporalCNN,
    keyed by one of:
        polyphony_reg, polyphony_class, event_logits,
        framewise_polyphony_reg, framewise_polyphony_class,
        species_polyphony_reg, species_polyphony_class

    forward(spatial_embeddings) -> dict[str, Tensor]
        spatial_embeddings: (batch, time, freq, embedding) or (batch, time, embedding).
            A freq axis (if present) is averaged out first, exactly like
            TemporalCNN's `tf.reduce_mean(inputs, axis=2)`.

    Output shapes (channel-last, matching the TF model):
        polyphony_reg                -> (batch,)
        polyphony_class               -> (batch, num_classes)
        event_logits                  -> (batch, time)
        framewise_polyphony_reg       -> (batch, time)
        framewise_polyphony_class     -> (batch, time, num_classes)
        species_polyphony_reg         -> (batch, num_species)
        species_polyphony_class       -> (batch, num_species, num_classes)
    """

    SUPPORTED = {
        "polyphony_reg",
        "polyphony_class",
        "event_logits",
        "framewise_polyphony_reg",
        "framewise_polyphony_class",
        "species_polyphony_reg",
        "species_polyphony_class",
    }

    # Marks this head as wanting the spatial/frame-wise embedding sequence
    # rather than a single pooled vector -- see torch_models_integration.py.
    takes_spatial_embeddings = True

    def __init__(self, input_size, objectives_cfg, conv_channels=(512, 256), dropout_rate=0.3):
        super().__init__()

        if isinstance(input_size, (torch.Size, tuple, list)):
            feature_dim = input_size[-1]
        else:
            feature_dim = input_size

        self.in_features = feature_dim  # embedding channels of the incoming spatial_embeddings

        unsupported = set(objectives_cfg.keys()) - self.SUPPORTED
        if unsupported:
            raise ValueError(f"Unknown objective(s) {unsupported}. Supported: {sorted(self.SUPPORTED)}")

        self.objectives_cfg = dict(objectives_cfg)
        self.conv_channels = list(conv_channels)
        trunk_dim = self.conv_channels[1]

        # ---- shared encoder, mirrors TemporalCNN.conv1/bn1/dropout1/conv2/bn2 ----
        self.conv1 = nn.Conv1d(feature_dim, self.conv_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.conv_channels[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(self.conv_channels[0], trunk_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(trunk_dim)

        # ---- per-objective heads ----
        if "polyphony_reg" in self.objectives_cfg:
            self.segment_dense1 = nn.Linear(trunk_dim, 128)
            self.segment_dropout = nn.Dropout(dropout_rate)
            self.segment_dense2 = nn.Linear(128, 1)

        if "polyphony_class" in self.objectives_cfg:
            self.polyphony_num_classes = self.objectives_cfg["polyphony_class"].get("num_classes", 9)
            self.segment_dense1_class = nn.Linear(trunk_dim, 128)
            self.segment_dropout_class = nn.Dropout(dropout_rate)
            self.polyphony_class_head = nn.Linear(128, self.polyphony_num_classes)

        if "event_logits" in self.objectives_cfg:
            self.event_head = nn.Conv1d(trunk_dim, 1, kernel_size=1)

        if "framewise_polyphony_reg" in self.objectives_cfg:
            self.frame_polyphony_reg_head = nn.Conv1d(trunk_dim, 1, kernel_size=1)

        if "framewise_polyphony_class" in self.objectives_cfg:
            self.frame_polyphony_num_classes = self.objectives_cfg["framewise_polyphony_class"].get("num_classes", 9)
            self.frame_polyphony_class_head = nn.Conv1d(trunk_dim, self.frame_polyphony_num_classes, kernel_size=1)

        if "species_polyphony_reg" in self.objectives_cfg:
            self.num_species_reg = self.objectives_cfg["species_polyphony_reg"]["num_species"]
            self.species_polyphony_reg_dense1 = nn.Linear(trunk_dim, 256)
            self.species_polyphony_reg_dropout = nn.Dropout(dropout_rate)
            self.species_polyphony_reg_dense2 = nn.Linear(256, 128)
            self.species_polyphony_reg_head = nn.Linear(128, self.num_species_reg)

        if "species_polyphony_class" in self.objectives_cfg:
            self.num_classes_global = self.objectives_cfg["species_polyphony_class"]["num_classes"]
            self.num_species_global = self.objectives_cfg["species_polyphony_class"]["num_species"]
            self.species_polyphony_class_dense1 = nn.Linear(trunk_dim, 256)
            self.species_polyphony_class_dropout = nn.Dropout(dropout_rate)
            self.species_polyphony_class_dense2 = nn.Linear(256, 128)
            self.species_polyphony_class_head = nn.Linear(
                128, self.num_species_global * self.num_classes_global
            )

    def forward(self, spatial_embeddings):
        x = spatial_embeddings
        if x.dim() == 4:
            # (batch, time, freq, embedding) -> average out freq, like
            # TemporalCNN's `tf.reduce_mean(inputs, axis=2)`
            x = x.mean(dim=2)
        elif x.dim() != 3:
            raise ValueError(
                f"Expected spatial_embeddings with 3 or 4 dims (batch[, time, freq], embedding), "
                f"got shape {tuple(x.shape)}"
            )

        # torch Conv1d wants (batch, channels, time); TF Conv1D is channel-last
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        features = self.bn2(x)  # (batch, trunk_dim, time)

        pooled = features.mean(dim=2)  # GlobalAveragePooling1D equivalent -> (batch, trunk_dim)

        outputs = {}

        if "polyphony_reg" in self.objectives_cfg:
            h = torch.relu(self.segment_dense1(pooled))
            h = self.segment_dropout(h)
            outputs["polyphony_reg"] = self.segment_dense2(h).squeeze(-1)

        if "polyphony_class" in self.objectives_cfg:
            h = torch.relu(self.segment_dense1_class(pooled))
            h = self.segment_dropout_class(h)
            outputs["polyphony_class"] = self.polyphony_class_head(h)

        if "event_logits" in self.objectives_cfg:
            outputs["event_logits"] = self.event_head(features).squeeze(1)  # (batch, time)

        if "framewise_polyphony_reg" in self.objectives_cfg:
            outputs["framewise_polyphony_reg"] = self.frame_polyphony_reg_head(features).squeeze(1)  # (batch, time)

        if "framewise_polyphony_class" in self.objectives_cfg:
            out = self.frame_polyphony_class_head(features)  # (batch, num_classes, time)
            outputs["framewise_polyphony_class"] = out.transpose(1, 2)  # -> (batch, time, num_classes)

        if "species_polyphony_reg" in self.objectives_cfg:
            h = torch.relu(self.species_polyphony_reg_dense1(pooled))
            h = self.species_polyphony_reg_dropout(h)
            h = torch.relu(self.species_polyphony_reg_dense2(h))
            outputs["species_polyphony_reg"] = self.species_polyphony_reg_head(h)

        if "species_polyphony_class" in self.objectives_cfg:
            h = torch.relu(self.species_polyphony_class_dense1(pooled))
            h = self.species_polyphony_class_dropout(h)
            h = torch.relu(self.species_polyphony_class_dense2(h))
            h = self.species_polyphony_class_head(h)
            outputs["species_polyphony_class"] = h.view(-1, self.num_species_global, self.num_classes_global)

        return outputs


class MultiTaskSimpleMLPHead(nn.Module):
    """
    Shared MLP trunk (hidden_units, dropout only after the first layer -- same
    as model.py's SimpleMLP), adapted to work on *either* pooled or spatial
    embeddings:

      - Segment-level objectives (polyphony_reg/class, species_polyphony_*)
        always run on the pooled trunk output -- pooled over time first if a
        time axis is present, used as-is otherwise.
      - Frame-wise objectives (event_logits, framewise_polyphony_reg/class)
        run on the *unpooled* trunk output, so they produce genuine per-frame
        predictions -- and are only built/usable if actually configured.

    This is a deliberate departure from a literal SimpleMLP port: SimpleMLP
    flattens its input before any head runs, so it never has a time axis to
    work with, and its event_logits/framewise_* heads silently degrade to
    one scalar per clip (see the note in MultiTaskTemporalCNNHead's module docstring --
    that also doesn't match the frame-wise labels add_labels builds for those
    objectives). This version keeps the time axis alive for the heads that
    actually need it.

    `takes_spatial_embeddings` is set as an *instance* attribute here (not a
    fixed class attribute like MultiTaskTemporalCNNHead's), computed from
    `objectives_cfg` at construction time: True only if a frame-wise
    objective is configured. That's what lets the same head class work with
    either pooled or spatial input depending on what you're training --
    the wrapper patches check this attribute to decide what to pass in,
    and that decision has to be made before any data exists.

    objectives_cfg: same shape as for MultiTaskTemporalCNNHead.
    """

    SUPPORTED = {
        "polyphony_reg",
        "polyphony_class",
        "event_logits",
        "framewise_polyphony_reg",
        "framewise_polyphony_class",
        "species_polyphony_reg",
        "species_polyphony_class",
    }

    FRAMEWISE = {"event_logits", "framewise_polyphony_reg", "framewise_polyphony_class"}

    def __init__(self, input_size, objectives_cfg, hidden_units=(512, 256), dropout_rate=0.3):
        super().__init__()

        if isinstance(input_size, (torch.Size, tuple, list)):
            feature_dim = input_size[-1]
        else:
            feature_dim = input_size

        self.in_features = feature_dim

        unsupported = set(objectives_cfg.keys()) - self.SUPPORTED
        if unsupported:
            raise ValueError(f"Unknown objective(s) {unsupported}. Supported: {sorted(self.SUPPORTED)}")

        self.objectives_cfg = dict(objectives_cfg)
        self.hidden_units = list(hidden_units)

        # Only ask the wrapper for spatial_embeddings if we actually need a
        # time axis (frame-wise objectives); otherwise take the cheaper
        # pooled vector. See class docstring.
        self.takes_spatial_embeddings = bool(self.FRAMEWISE & set(self.objectives_cfg.keys()))

        # ---- shared MLP trunk. nn.Linear only acts on the last dim, so this
        # works unchanged on (batch, feature) or (batch, time, feature) input.
        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        prev_dim = feature_dim
        for i, units in enumerate(self.hidden_units):
            self.hidden_layers.append(nn.Linear(prev_dim, units))
            self.dropout_layers.append(nn.Dropout(dropout_rate) if i == 0 else nn.Identity())
            prev_dim = units
        trunk_dim = prev_dim

        # ---- per-objective heads ----
        if "polyphony_reg" in self.objectives_cfg:
            self.polyphony_reg_head = nn.Linear(trunk_dim, 1)

        if "polyphony_class" in self.objectives_cfg:
            self.polyphony_num_classes = self.objectives_cfg["polyphony_class"].get("num_classes", 9)
            self.polyphony_class_head = nn.Linear(trunk_dim, self.polyphony_num_classes)

        if "event_logits" in self.objectives_cfg:
            self.event_head = nn.Linear(trunk_dim, 1)

        if "framewise_polyphony_reg" in self.objectives_cfg:
            self.frame_polyphony_reg_head = nn.Linear(trunk_dim, 1)

        if "framewise_polyphony_class" in self.objectives_cfg:
            self.frame_polyphony_num_classes = self.objectives_cfg["framewise_polyphony_class"].get("num_classes", 9)
            self.frame_polyphony_class_head = nn.Linear(trunk_dim, self.frame_polyphony_num_classes)

        if "species_polyphony_reg" in self.objectives_cfg:
            self.num_species_reg = self.objectives_cfg["species_polyphony_reg"]["num_species"]
            self.species_polyphony_reg_dense1 = nn.Linear(trunk_dim, 256)
            self.species_polyphony_reg_dropout = nn.Dropout(dropout_rate)
            self.species_polyphony_reg_dense2 = nn.Linear(256, 128)
            self.species_polyphony_reg_head = nn.Linear(128, self.num_species_reg)

        if "species_polyphony_class" in self.objectives_cfg:
            self.num_classes_global = self.objectives_cfg["species_polyphony_class"]["num_classes"]
            self.num_species_global = self.objectives_cfg["species_polyphony_class"]["num_species"]
            self.species_polyphony_class_dense1 = nn.Linear(trunk_dim, 256)
            self.species_polyphony_class_dropout = nn.Dropout(dropout_rate)
            self.species_polyphony_class_dense2 = nn.Linear(256, 128)
            self.species_polyphony_class_head = nn.Linear(
                128, self.num_species_global * self.num_classes_global
            )

    def forward(self, x):
        # x: (batch, feature) pooled, or (batch, time[, freq], feature) spatial
        has_time = x.dim() >= 3
        if x.dim() == 4:
            x = x.mean(dim=2)  # average out freq axis, like MultiTaskTemporalCNNHead
        elif x.dim() not in (2, 3):
            raise ValueError(
                f"Expected input with 2, 3, or 4 dims (batch[, time[, freq]], embedding), "
                f"got shape {tuple(x.shape)}"
            )

        if self.takes_spatial_embeddings and not has_time:
            needs = sorted(self.FRAMEWISE & set(self.objectives_cfg))
            raise ValueError(
                f"MultiTaskSimpleMLPHead was configured with frame-wise objective(s) {needs}, which "
                "need a time axis, but received a pooled (no time axis) embedding instead. "
                "Check that the wrapper is routing spatial_embeddings here (it reads this "
                "head's `takes_spatial_embeddings` attribute)."
            )

        for linear, dropout in zip(self.hidden_layers, self.dropout_layers):
            x = torch.relu(linear(x))
            x = dropout(x)
        features = x  # (batch, trunk_dim) or (batch, time, trunk_dim)

        pooled = features.mean(dim=1) if has_time else features

        outputs = {}

        if "polyphony_reg" in self.objectives_cfg:
            outputs["polyphony_reg"] = self.polyphony_reg_head(pooled).squeeze(-1)

        if "polyphony_class" in self.objectives_cfg:
            outputs["polyphony_class"] = self.polyphony_class_head(pooled)

        if "event_logits" in self.objectives_cfg:
            outputs["event_logits"] = self.event_head(features).squeeze(-1)  # (batch, time)

        if "framewise_polyphony_reg" in self.objectives_cfg:
            outputs["framewise_polyphony_reg"] = self.frame_polyphony_reg_head(features).squeeze(-1)  # (batch, time)

        if "framewise_polyphony_class" in self.objectives_cfg:
            outputs["framewise_polyphony_class"] = self.frame_polyphony_class_head(features)  # (batch, time, num_classes)

        if "species_polyphony_reg" in self.objectives_cfg:
            h = torch.relu(self.species_polyphony_reg_dense1(pooled))
            h = self.species_polyphony_reg_dropout(h)
            h = torch.relu(self.species_polyphony_reg_dense2(h))
            outputs["species_polyphony_reg"] = self.species_polyphony_reg_head(h)

        if "species_polyphony_class" in self.objectives_cfg:
            h = torch.relu(self.species_polyphony_class_dense1(pooled))
            h = self.species_polyphony_class_dropout(h)
            h = torch.relu(self.species_polyphony_class_dense2(h))
            h = self.species_polyphony_class_head(h)
            outputs["species_polyphony_class"] = h.view(-1, self.num_species_global, self.num_classes_global)

        return outputs

##################################
# Output classes
################################## 

@dataclass
class EmbeddingModelOutput(ModelOutput):
    """Custom output for embedding models"""
    pooled_embeddings: Optional[torch.Tensor] = None
    spatial_embeddings: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None 

@dataclass
class MultiTaskOutput(ModelOutput):
    """Custom output for multi-task learning"""
    loss: Optional[torch.Tensor] = None
    
    # Task-specific outputs
    classification_logits: Optional[torch.Tensor] = None
    regression_logits: Optional[torch.Tensor] = None
    detection_logits: Optional[torch.Tensor] = None
    
    # Task-specific losses (optional)
    classification_loss: Optional[torch.Tensor] = None
    regression_loss: Optional[torch.Tensor] = None
    detection_loss: Optional[torch.Tensor] = None
    
    # Shared representations
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    pooled_output: Optional[torch.Tensor] = None

@dataclass
class MultiTaskModelOutput(ModelOutput):
    """Output for multi-task models"""
    loss: Optional[torch.Tensor] = None
    losses: Optional[Dict[str, torch.Tensor]] = None
    logits: Optional[Dict[str, torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None

##################################
# Models
##################################

class PrecomputedEmbeddingModel(torch.nn.Module):
    """
    No-encoder stand-in for torch_train.py's harness: the input IS already the
    precomputed (pooled or spatial) embedding, so this just passes it straight
    to whichever head is attached. Lets MultiTaskSimpleMLPHead / SimpleMLPHead /
    MultiTaskTemporalCNNHead run as complete standalone models on precomputed
    embeddings, for direct comparison against the Keras head-only training.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_head = None
        self.sampling_rate = None  # unused -- no audio preprocessing in this path

    def set_sampling_rate(self, sr):
        pass  # no-op, nothing to set

    def get_head_input_size(self):
        return self.embedding_dim

    def replace_head(self, new_head):
        self.output_head = new_head

    def freeze_encoder(self):
        pass  # no encoder to freeze -- the head is the entire trainable model

    def forward(self, x):
        logits = self.output_head(x) if self.output_head else None
        return EmbeddingModelOutput(
            pooled_embeddings=x if x.dim() == 2 else None,
            spatial_embeddings=x if x.dim() > 2 else None,
            logits=logits,
        )

# Wrapper for pretrained model https://huggingface.co/DBD-research-group/EfficientNet-B1-BirdSet-XCL
# added automatic preprocessing, freeze_encoder(), replace_head(), get_head_input_size()
class BirdSetEfficientNet(torch.nn.Module):
    """
    Wrapper for pretrained EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g.
    for ImageNet. Original model: https://huggingface.co/DBD-research-group/EfficientNet-B1-BirdSet-XCL
    """
    def __init__(self, pretrained_model_path="DBD-research-group/EfficientNet-B1-BirdSet-XCM"):
        super().__init__()
        
        # Load pretrained model
        self.model = EfficientNetForImageClassification.from_pretrained(
            pretrained_model_path,
            num_channels=1,
            ignore_mismatched_sizes=True,
        )

        # Init config
        self.config = self.model.config
        self.sampling_rate = 32000 # as described here https://huggingface.co/DBD-research-group/EfficientNet-B1-BirdSet-XCL

        # Init preprocessing signal converters
        self.spectrogram_converter = torchaudio.transforms.Spectrogram(
            n_fft=2048, hop_length=256, power=2.0
        )
        self.mel_converter = torchaudio.transforms.MelScale(
            n_mels=256, n_stft=1025, sample_rate=32_000
        )
        self.power_to_db = PowerToDB(top_db=80)
    
    def preprocess(self, audio):
        """
        Preprocess the audio to the format that the model expects
        - Resample to 32kHz
        - Convert to melscale spectrogram n_fft: 2048, hop_length: 256, power: 2. melscale: n_mels: 256, n_stft: 1025
        - Normalize the melscale spectrogram with mean: -4.268, std: 4.569 (from AudioSet)

        """
        audio = audio.to(torch.float32) 
        spectrogram = self.spectrogram_converter(audio)
        #spectrogram = spectrogram.to(torch.float32)
        melspec = self.mel_converter(spectrogram)
        dbscale = self.power_to_db(melspec)
        normalized_dbscale = transforms.Normalize((-4.268,), (4.569,))(dbscale)
        
        if normalized_dbscale.dim() == 2:
            normalized_dbscale = normalized_dbscale.unsqueeze(0).unsqueeze(0)
        elif normalized_dbscale.dim() == 3:
            normalized_dbscale = normalized_dbscale.unsqueeze(1)
        return normalized_dbscale
    
    def forward(self, audio): 
        """Forward pass with automatic preprocessing"""
        spectrogram = self.preprocess(audio)

        encoder_outputs = self.model.efficientnet(spectrogram)
        last_hidden_states = encoder_outputs.last_hidden_state
        pooled_features = encoder_outputs.pooler_output
        spatial_embeddings = last_hidden_states.permute(0, 3, 2, 1) # (batch, time, freq, embeddings) to match other models

        # Heads that declare `takes_spatial_embeddings = True` (e.g. MultiTaskHead)
        # get the frame-wise sequence instead of the pooled vector.
        head_input = spatial_embeddings if getattr(self.model.classifier, "takes_spatial_embeddings", False) else pooled_features
        logits = self.model.classifier(head_input)

        return EmbeddingModelOutput(
            pooled_embeddings=pooled_features,
            spatial_embeddings=spatial_embeddings,
            logits=logits
        )
    
    def freeze_encoder(self):
        for param in self.model.efficientnet.parameters():
            param.requires_grad = False

    def replace_head(self, new_head):
        self.model.classifier = new_head

    def get_head_input_size(self):
        return self.model.classifier.in_features
    
    def get_sampling_rate(self):
        return self.sampling_rate
    
class BirdSetBirdMAE(torch.nn.Module):
    """
    Wrapper for pretrained Bird-MAE Model. Original model: https://huggingface.co/DBD-research-group/Bird-MAE-Huge
    """
    def __init__(self, pretrained_model_path="DBD-research-group/Bird-MAE-Huge", pooling="mean"):
        super().__init__()

        self.model = AutoModel.from_pretrained(pretrained_model_path, trust_remote_code=True)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_path, trust_remote_code=True)
        self.output_head = None
        self.pooling = pooling  # None / False -> spatial only, "mean", "cls"

        # Always disable internal pooling — we handle pooling ourselves below,
        # so we always get the full raw token sequence back.
        self.model.global_pool = None

        self.config = self.model.config
        self.sampling_rate = 32000

        self.time_patches = self.config.img_size_x // self.config.patch_size
        self.freq_patches = self.config.img_size_y // self.config.patch_size

    def preprocess(self, audio):
        device = next(self.parameters()).device
        mel_spectrogram = self.feature_extractor(audio, return_tensors="pt")
        mel_spectrogram = mel_spectrogram.to(device)
        return mel_spectrogram

    def forward(self, audio):
        logits = None
        device = next(self.parameters()).device
        audio = audio.to(device)
        mel_spectrogram = self.preprocess(audio)
        outputs = self.model(mel_spectrogram)

        tokens = outputs.last_hidden_state        # (B, 1+N, D), unnormalized since global_pool=None
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        spatial_embeddings = patch_tokens.reshape(
            patch_tokens.shape[0], self.time_patches, self.freq_patches, -1
        )

        pooled_embeddings = None

        if self.output_head is not None and getattr(self.output_head, "takes_spatial_embeddings", False):
            x = spatial_embeddings
        elif not self.pooling:
            x = spatial_embeddings
        elif self.pooling == "mean":
            pooled_embeddings = self.model.fc_norm(patch_tokens.mean(dim=1))
            x = pooled_embeddings
        elif self.pooling == "cls":
            pooled_embeddings = self.model.norm(cls_token)
            x = pooled_embeddings
        else:
            raise ValueError(f"Pooling option {self.pooling} not supported")

        if self.output_head:
            logits = self.output_head(x)

        return EmbeddingModelOutput(
            pooled_embeddings=pooled_embeddings,
            spatial_embeddings=spatial_embeddings,
            logits=logits
        )

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def replace_head(self, new_head: torch.nn.Module):
        self.output_head = new_head

    def get_head_input_size(self):
        return self.config.embed_dim
    
# class BirdSetBirdMAE(torch.nn.Module):
#     """
#     Wrapper for pretrained Bird-MAE Model. Original model: https://huggingface.co/DBD-research-group/Bird-MAE-Base
#     """
#     def __init__(self, pretrained_model_path="DBD-research-group/Bird-MAE-Huge"):
#         super().__init__()
        
#         # Load pretrained model and feature extractor
#         self.model = AutoModel.from_pretrained(pretrained_model_path,trust_remote_code=True)
#         self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_path, trust_remote_code=True)
#         self.output_head = None

#         # Init config
#         self.config = self.model.config
#         self.sampling_rate = 32000

#     # def preprocess(self, audio):
#     #     mel_spectrogram = self.feature_extractor(audio)
#     #     return mel_spectrogram
    
#     def preprocess(self, audio):
#         device = next(self.parameters()).device
#         # Feature extractor returns a dict-like BatchFeature, extract the tensor
#         mel_spectrogram = self.feature_extractor(audio, return_tensors="pt")
#         # mel_spectrogram = inputs["input_values"]  # or "input_features" depending on the model
#         mel_spectrogram = mel_spectrogram.to(device)
#         return mel_spectrogram
    
#     def forward(self, audio):
#         """Forward pass with automatic preprocessing and optional pooling and output head"""
#         logits = None
#         device = next(self.parameters()).device
#         audio = audio.to(device)
#         mel_spectrogram = self.preprocess(audio)
#         outputs = self.model(mel_spectrogram)
#         last_hidden_state = outputs.last_hidden_state  # (batch, embedding)
#         spatial_embeddings = None
#         pooled_embeddings = last_hidden_state

#         # Heads that declare `takes_spatial_embeddings = True` (e.g. MultiTaskHead)
#         # get the frame-wise sequence; everything else gets the pooled vector
#         # (previously this always passed the full sequence, which silently broke
#         # any single-vector head such as SimpleRegressionHead).
#         if self.output_head:
#             head_input = spatial_embeddings if getattr(self.output_head, "takes_spatial_embeddings", False) else pooled_embeddings
#             logits = self.output_head(head_input)
#         return EmbeddingModelOutput(
#             pooled_embeddings=pooled_embeddings,
#             spatial_embeddings=spatial_embeddings,
#             logits=logits
#         )
    
#     def freeze_encoder(self):
#         for param in self.model.parameters():
#             param.requires_grad = False

#     def replace_head(self, new_head: torch.nn.Module):
#         self.output_head = new_head

#     def get_head_input_size(self):
#         return self.config.embed_dim
    
#     def get_sampling_rate(self):
#         return self.sampling_rate
    
class BirdSetAudioProtoPNet(torch.nn.Module):
    """
    Wrapper for pretrained AudioProtPNet Model. 
    The ConvNeXt backbone produces embeddings with a height of Hz = 8, a width of Wz = 19, and D = 1024 channels. 
    Original model: https://huggingface.co/DBD-research-group/AudioProtoPNet-5-BirdSet-XCL
    """
    def __init__(self, pretrained_model_path="DBD-research-group/AudioProtoPNet-5-BirdSet-XCL", pooling=None):
        super().__init__()
        
        # Load pretrained model and feature extractor
        self.model = AutoModel.from_pretrained(pretrained_model_path,trust_remote_code=True)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_path, trust_remote_code=True)
        self.output_head = None

        # Init
        self.config = self.model.config
        self.sampling_rate = 32000
        self.pooling = pooling
        self._head_input_size = None
        self.mel_spectrogram = None

    def preprocess(self, audio):
        mel_spectrogram = self.feature_extractor(audio).to(device=next(self.parameters()).device)
        return mel_spectrogram
    
    def forward(self, audio): 
        """Forward pass with automatic preprocessing and optional pooling and output head"""
        logits = None
        pooled_output = None
        
        audio = audio.to(device=next(self.parameters()).device)

        mel_spectrogram = self.preprocess(audio)
        outputs = self.model(mel_spectrogram)
        last_hidden_state = outputs.last_hidden_state

        if self.output_head is not None and getattr(self.output_head, "takes_spatial_embeddings", False):
            x = last_hidden_state
        elif not self.pooling:
            x = last_hidden_state
        elif self.pooling=='mean':
            pooled_output = last_hidden_state.mean(dim=[2,3])
            x = pooled_output
        else:
            raise ValueError(f"Pooling option {self.pooling} not supported")
        
        spatial_embeddings = last_hidden_state.permute(0, 3, 2, 1) # (batch, time, freq, embeddings) to match other models
    
        if self.output_head:
            logits = self.output_head(x)

        spatial_embeddings = last_hidden_state.permute(0, 3, 2, 1) # (batch, time, freq, embeddings) to match other models

        print("Spectrogram shape:", mel_spectrogram.shape)   # (batch, channels, H, W)
        print("Last hidden state shape:", last_hidden_state.shape)  # expect (batch, embedding, freq, time)
        print("Pooled output shape:", pooled_output.shape if pooled_output is not None else None)  # expect (batch, embedding)
        print("Spatial embeddings shape:", spatial_embeddings.shape)  # expect (batch, time, freq, embedding)
        
        return EmbeddingModelOutput(
        pooled_embeddings=pooled_output,
        spatial_embeddings=spatial_embeddings, # (batch, time, freq, embeddings) to match other models
        logits=logits
    )
    
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def replace_head(self, new_head: torch.nn.Module):
        self.output_head = new_head

    def get_head_input_size(self):
        """
        Returns the input size for the output head.
        If not yet computed, performs a dummy forward pass to determine it.
        """
        if self._head_input_size is not None:
            return self._head_input_size
        
        # Perform a dummy forward pass with a small random input
        self.eval()
        with torch.no_grad():
            # Create dummy audio (1 second at 32kHz)
            dummy_audio = torch.randn(1, self.sampling_rate)
            
            mel_spectrogram = self.preprocess(dummy_audio)
            outputs = self.model(mel_spectrogram)
            last_hidden_state = outputs.last_hidden_state
            
            if not self.pooling:
                x = last_hidden_state
            elif self.pooling == 'mean':
                x = last_hidden_state.mean(dim=[2, 3])
            else:
                raise ValueError(f"Pooling option {self.pooling} not supported")
            
            # Calculate input size
            self._head_input_size = x.shape[-1]  # Last dimension is the embedding size
            
        return self._head_input_size
    
    def replace_head_with_proto_head(self):
        """
        Hacky way of replacing the output head with the protypical head from AudioProtoPNet. 
        Source: https://huggingface.co/DBD-research-group/AudioProtoPNet-5-BirdSet-XCL/blob/main/modeling_protonet.py
        """
        import sys
        modeling_protonet = sys.modules.get('modeling_protonet')

        if modeling_protonet:
            AudioProtoNetClassificationHead = modeling_protonet.AudioProtoNetClassificationHead
        else:
            for module_name in sys.modules:
                if 'modeling_protonet' in module_name.lower():
                    modeling_protonet = sys.modules[module_name]
                    AudioProtoNetClassificationHead = modeling_protonet.AudioProtoNetClassificationHead
                else:
                    return
        self.output_head = AudioProtoNetClassificationHead(self.config)

class BirdSetAST(torch.nn.Module):
    """
    Wrapper for pretrained BirdSet AST Model. Original model: https://huggingface.co/DBD-research-group/AST-BirdSet-XCM

    Paper: Gong et al. (2021): AST: Audio Spectrogram Transformer [https://arxiv.org/abs/2104.01778]
    Original AST: https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer
    """
    def __init__(self, pretrained_model_path="DBD-research-group/AST-Birdset-XCL", pooling=True):
        super().__init__()
        
        # Load pretrained model and feature extractor
        self.model = AutoModel.from_pretrained(pretrained_model_path,trust_remote_code=True)
        #self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_path, trust_remote_code=True)
        self.output_head = None

        # Init
        self.config = self.model.config
        self.sampling_rate = 16000
        self.pooling = pooling

        # Init preprocessing signal converters
        n_fft=int(0.025 * self.sampling_rate) # 25ms window
        self.spectrogram_converter = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,           # 25ms window
            hop_length=int(0.010 * self.sampling_rate),      # 10ms hop
            win_length=int(0.025 * self.sampling_rate),      # 25ms window
            window_fn=torch.hamming_window,
            power=2.0,       
            normalized=False # TODO: Does AST expect normalized input?
        )

        self.mel_converter = torchaudio.transforms.MelScale(
            n_mels=128,   
            n_stft= (n_fft // 2) + 1,   
            sample_rate=int(self.sampling_rate)
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    def preprocess(self, audio):
        """
        Preprocess the audio to the format that the model expects as described here: 
        Gong et al. (2021): AST: Audio Spectrogram Transformer [https://arxiv.org/abs/2104.01778]

        "First, the input audio waveform of t seconds is converted into a sequence of 128-dimensional log Mel filterbank (fbank) 
        features computed with a 25ms Hamming window every 10ms. 
        This results in a 128 × 100t spectrogram as input to the AST"(Gong et al. 2021, p. 572)
        
        - Pads audio to 10s
        - Pads spectrogram to 1024 bins to match config
        """
        if not self.sampling_rate:
            raise ValueError(f"Sampling rate is not set and is needed for preprocessing.")

        # # Init preprocessing signal converters if not initialized
        # if not self.spectrogram_converter or not self.mel_converter or not self.amplitude_to_db:
            
        #     n_fft=int(0.025 * self.sampling_rate) # 25ms window
        #     self.spectrogram_converter = torchaudio.transforms.Spectrogram(
        #         n_fft=n_fft,           # 25ms window
        #         hop_length=int(0.010 * self.sampling_rate),      # 10ms hop
        #         win_length=int(0.025 * self.sampling_rate),      # 25ms window
        #         window_fn=torch.hamming_window,
        #         power=2.0,       
        #         normalized=False # TODO: Does AST expect normalized input?
        #     )

        #     self.mel_converter = torchaudio.transforms.MelScale(
        #         n_mels=128,   
        #         n_stft= (n_fft // 2) + 1,   
        #         sample_rate=int(self.sampling_rate)
        #     )

        #     self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

        # pad audio to 10s
        target_samples = 10 * self.sampling_rate  # 160,000 samples = 10s
        current_samples = audio.shape[-1]
        if current_samples >= target_samples:
            audio = audio[..., :target_samples]
        pad_right = target_samples - current_samples
        padded_audio = nn.functional.pad(audio, (0, pad_right), mode='constant', value=0.0)

        spectrogram = self.spectrogram_converter(padded_audio)
        spectrogram = spectrogram.to(torch.float32)
        mel_spectrogram = self.mel_converter(spectrogram)
        log_mel_spectrogram = self.amplitude_to_db(mel_spectrogram)

        return log_mel_spectrogram
    
    def forward(self, audio): 
        """Forward pass with automatic preprocessing and optional pooling and output head"""
        logits = None
        audio = audio.to(device=next(self.parameters()).device)
        log_mel_spectrogram = self.preprocess(audio)

        # Hack to match expected input size: pad spectrogram
        target_frames = 1024  # Standard HF AST config
        current_frames = log_mel_spectrogram.shape[-1]  # Your 1001

        if current_frames < target_frames:
            pad_frames = target_frames - current_frames  # 1024 - 1001 = 23
            log_mel_spectrogram = torch.nn.functional.pad(
                log_mel_spectrogram, (0, pad_frames), mode='constant', value=0.0
            )

        outputs = self.model(log_mel_spectrogram)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output

        if self.output_head is not None and getattr(self.output_head, "takes_spatial_embeddings", False):
            x = last_hidden_state
        elif self.pooling:
            x = pooler_output
        else:
            x = last_hidden_state
    
        if self.output_head:
            logits = self.output_head(x)

        return EmbeddingModelOutput(
        pooled_embeddings=pooler_output,
        spatial_embeddings=last_hidden_state,
        logits=logits
    )
    
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def replace_head(self, new_head: torch.nn.Module):
        self.output_head = new_head

    def get_head_input_size(self):
        return self.config.hidden_size
    
    def set_sampling_rate(self, new_sampling_rate):
        # Reinitialize the spectrogram and mel converters if the sampling rate changes
        if new_sampling_rate != self.sampling_rate:
            self.sampling_rate = new_sampling_rate
            n_fft = int(0.025 * self.sampling_rate)
            self.spectrogram_converter = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=int(0.010 * self.sampling_rate),
                win_length=int(0.025 * self.sampling_rate),
                window_fn=torch.hamming_window,
                power=2.0,
                normalized=False
            ).to(next(self.parameters()).device)
            self.mel_converter = torchaudio.transforms.MelScale(
                n_mels=128,
                n_stft=(n_fft // 2) + 1,
                sample_rate=int(self.sampling_rate)
            ).to(next(self.parameters()).device)
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
                stype='power', top_db=80
            ).to(next(self.parameters()).device)

class BirdSetWav2Vec2(torch.nn.Module):
    """
    Wrapper for pretrained wav2vec2 Model. Original model: https://huggingface.co/DBD-research-group/Wav2Vec2-Base-BirdSet-XCM

    Original: https://huggingface.co/docs/transformers/en/model_doc/wav2vec2
    
    wav2vec2 has to options of spatial embeddings:
    - last_hidden_state: [None, 269, 768] (framewise embeddings)
    - xvector: [None, 269, 512] (framewise speech/species? representations)
    """
    def __init__(self, pretrained_model_path="DBD-research-group/Wav2Vec2-Base-BirdSet-XCM", pooling=None, spatial_embeddings='last_hidden_state'):
        super().__init__()
        
        # Load pretrained model and feature extractor
        self.model = AutoModel.from_pretrained(pretrained_model_path,trust_remote_code=True)
        self.output_head = None

        # Init config
        self.config = self.model.config
        self.sampling_rate = 32000
        self.pooling = pooling
        self.spatial_embeddings = spatial_embeddings
    
    def forward(self, audio): 
        """Forward pass with automatic preprocessing and optional pooling and output head"""
        logits = None
        pooled_output = None

        audio = audio.to(device=next(self.parameters()).device)

        outputs = self.model(audio)
        if self.spatial_embeddings=="last_hidden_state":
            spatial_embeddings = outputs.last_hidden_state
        elif self.spatial_embeddings=="xvector":
            spatial_embeddings = outputs.extract_features
        else:
            raise ValueError(f"Spatial embeddings key is not valid. Use 'last_hidden_state' or 'xvector'")
    
        if self.output_head is not None and getattr(self.output_head, "takes_spatial_embeddings", False):
            x = spatial_embeddings
        elif not self.pooling:
            x = spatial_embeddings
        elif self.pooling=='mean':
            pooled_output = spatial_embeddings.mean(dim=1)
            x = pooled_output
        else:
            raise ValueError(f"Pooling option {self.pooling} not supported")
        
        if self.output_head:
            logits = self.output_head(x)

        return EmbeddingModelOutput(
        pooled_embeddings=pooled_output,
        spatial_embeddings=spatial_embeddings[:, None, :], # (time, freq, embeddings) to match other models, add dummy freq dimension
        logits=logits
    )
    
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def replace_head(self, new_head: torch.nn.Module):
        self.output_head = new_head

    def get_head_input_size(self):
        return self.config.output_hidden_size
    
class NatureLMBEATs(torch.nn.Module):
    """
    Wrapper for the BEATs audio encoder extracted from NatureLM-audio, released by
    Earth Species Project. Original model:
    https://huggingface.co/EarthSpeciesProject/esp-aves2-naturelm-audio-v1-beats

    Unlike the other wrappers in this file, this is NOT loaded via
    transformers.AutoModel -- DBD-research-group has not published a BirdSet-trained
    BEATs checkpoint on Hugging Face. This is the BEATs encoder as fine-tuned inside
    NatureLM-audio (unfrozen during large-scale audio-language training), distributed
    through Earth Species Project's own `avex` library instead.

    Requires: pip install avex

    Like BirdSetWav2Vec2, BEATs consumes raw waveform directly -- avex's BEATs wrapper
    performs fbank extraction/normalization internally, so `preprocess` is a no-op here
    (kept only so the class matches the shape of the other wrappers).
    """
    def __init__(self, model_name="esp_aves2_naturelm_audio_v1_beats", device=None):
        super().__init__()

        try:
            from avex import load_model as avex_load_model
        except ImportError as e:
            raise ImportError(
                "NatureLMBEATs requires the `avex` package: pip install avex"
            ) from e

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # return_features_only=True: raw embeddings, no classifier head baked in --
        # equivalent to how the other wrappers here take model.<encoder> without
        # the HF classification head.
        self.model = avex_load_model(model_name, return_features_only=True, device=device)
        self.output_head = None

        # BEATs operates on 16kHz audio (unlike most other BirdSet wrappers here,
        # which use 32kHz) -- see set_sampling_rate() / resample your inputs.
        self.sampling_rate = 16000
        self._head_input_size = None

    def preprocess(self, audio):
        """No-op: avex's BEATs model does fbank extraction + normalization internally
        and expects a raw waveform tensor (batch, samples) at self.sampling_rate."""
        return audio

    def forward(self, audio):
        """Forward pass with automatic preprocessing and optional pooling and output head"""
        logits = None
        audio = audio.to(device=next(self.parameters()).device)
        wav = self.preprocess(audio)

        features = self.model(wav)  # (batch, time_steps, 768)
        pooled_embeddings = features.mean(dim=1)
        # add a dummy freq axis so shape matches the other wrappers'
        # (batch, time, freq, embedding) spatial_embeddings convention
        spatial_embeddings = features.unsqueeze(2)

        if self.output_head is not None:
            head_input = spatial_embeddings if getattr(self.output_head, "takes_spatial_embeddings", False) else pooled_embeddings
            logits = self.output_head(head_input)

        return EmbeddingModelOutput(
            pooled_embeddings=pooled_embeddings,
            spatial_embeddings=spatial_embeddings,
            logits=logits
        )

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def replace_head(self, new_head: torch.nn.Module):
        self.output_head = new_head

    def get_head_input_size(self):
        """
        Returns the embedding dimension for the output head (768 for BEATs-base).
        Computed via a dummy forward pass on first call, then cached.
        """
        if self._head_input_size is not None:
            return self._head_input_size

        self.eval()
        with torch.no_grad():
            dummy_audio = torch.randn(1, self.sampling_rate)  # 1 second at 16kHz
            dummy_audio = dummy_audio.to(device=next(self.parameters()).device)
            features = self.model(dummy_audio)
            self._head_input_size = features.shape[-1]

        return self._head_input_size

    def set_sampling_rate(self, new_sampling_rate):
        # BEATs' own preprocessing is baked into the avex model and expects 16kHz;
        # unlike BirdSetAST there's no local resampling machinery to reinitialize
        # here, so just resample your input audio to self.sampling_rate before
        # calling forward() if it's coming from a 32kHz-native BirdSet pipeline.
        if new_sampling_rate != self.sampling_rate:
            warnings.warn(
                f"NatureLMBEATs expects {self.sampling_rate}Hz audio; BEATs' internal "
                f"preprocessing is not reconfigurable. Resample your audio to "
                f"{self.sampling_rate}Hz (e.g. via torchaudio.functional.resample) "
                f"before calling forward()."
            )

    def get_sampling_rate(self):
        return self.sampling_rate

##################################
# Utilities
##################################

# Copied from: https://github.com/DBD-research-group/BirdSet/blob/main/birdset/datamodule/components/augmentations.py
class PowerToDB(nn.Module):
    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0):
        super(PowerToDB, self).__init__()
        # Initialize parameters
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def forward(self, S):
        # Convert S to a PyTorch tensor if it is not already
        S = torch.as_tensor(S, dtype=torch.float32)

        if self.amin <= 0:
            raise ValueError("amin must be strictly positive")

        if torch.is_complex(S):
            warnings.warn(
                "power_to_db was called on complex input so phase "
                "information will be discarded. To suppress this warning, "
                "call power_to_db(S.abs()**2) instead.",
                stacklevel=2,
            )
            magnitude = S.abs()
        else:
            magnitude = S

        # Check if ref is a callable function or a scalar
        if callable(self.ref):
            ref_value = self.ref(magnitude)
        else:
            ref_value = torch.abs(torch.tensor(self.ref, dtype=S.dtype))

        # Compute the log spectrogram
        log_spec = 10.0 * torch.log10(
            torch.maximum(magnitude, torch.tensor(self.amin, device=magnitude.device))
        )
        log_spec -= 10.0 * torch.log10(
            torch.maximum(ref_value, torch.tensor(self.amin, device=magnitude.device))
        )

        # Apply top_db threshold if necessary
        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)

        return log_spec

