from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from transformers import EfficientNetForImageClassification, AutoFeatureExtractor, AutoModel 
from transformers.modeling_outputs import ModelOutput

import librosa
import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms

from birdset.datamodule.components.augmentations import PowerToDB

##################################
# Model heads
##################################
class SimpleRegressionHead(torch.nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.2):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.regression_head(x)

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

# Wrapper for pretrained model https://huggingface.co/DBD-research-group/EfficientNet-B1-BirdSet-XCL
# added automatic preprocessing, freeze_encoder(), replace_head(), get_head_input_size()
class PretrainedBirdSetEfficientNet(torch.nn.Module):
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
        spectrogram = self.spectrogram_converter(audio)
        spectrogram = spectrogram.to(torch.float32)
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
        logits = self.model.classifier(pooled_features)

        return EmbeddingModelOutput(
            pooled_embeddings=pooled_features,
            spatial_embeddings=last_hidden_states,
            logits=logits
        )
    
    def freeze_encoder(self):
        for param in self.model.efficientnet.parameters():
            param.requires_grad = False

    def replace_head(self, new_head):
        self.model.classifier = new_head

    def get_head_input_size(self):
        return self.model.classifier.in_features
       
class PretrainedBirdMAE(torch.nn.Module):
    """
    Wrapper for pretrained Bird-MAE Model. Original model: https://huggingface.co/DBD-research-group/Bird-MAE-Base
    """
    def __init__(self, pretrained_model_path="DBD-research-group/Bird-MAE-Base"):
        super().__init__()
        
        # Load pretrained model and feature extractor
        self.model = AutoModel.from_pretrained(pretrained_model_path,trust_remote_code=True)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_path, trust_remote_code=True)
        self.output_head = None

        # Init config
        self.config = self.model.config
        self.sampling_rate = 32000

    def preprocess(self, audio):
        mel_spectrogram = self.feature_extractor(audio)
        return mel_spectrogram
    
    def forward(self, audio): 
        """Forward pass with automatic preprocessing and optional pooling and output head"""
        logits = None

        mel_spectrogram = self.preprocess(audio)
        outputs = self.model(mel_spectrogram)
        last_hidden_state = outputs.last_hidden_state
        x = last_hidden_state
    
        if self.output_head:
            logits = self.output_head(x)

        return EmbeddingModelOutput(
        pooled_embeddings=last_hidden_state,
        spatial_embeddings=None,
        logits=logits
    )
    
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def replace_head(self, new_head: torch.nn.Module):
        self.output_head = new_head

    def get_head_input_size(self):
        return self.config.embed_dim


# TODO: If pooling use this:
#
# if not self.pooling:
#     pass
# elif self.pooling=='mean':
#     shape = x.shape
#     pooled_output = x.mean(dim=0)
#     shape_2 = pooled_output.shape
# else:
#     raise ValueError(f"Pooling option {self.pooling} not supported")
        




