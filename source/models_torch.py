from transformers import EfficientNetForImageClassification
import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms

from birdset.datamodule.components.augmentations import PowerToDB

# class PowerToDB(torch.nn.Module):
#     """
#     A power spectrogram to decibel conversion layer. See birdset.datamodule.components.augmentations
#     """

#     def __init__(self, ref=1.0, amin=1e-10, top_db=80.0):
#         super(PowerToDB, self).__init__()
#         # Initialize parameters
#         self.ref = ref
#         self.amin = amin
#         self.top_db = top_db

#     def forward(self, S):
#         # Convert S to a PyTorch tensor if it is not already
#         S = torch.as_tensor(S, dtype=torch.float32)

#         if self.amin <= 0:
#             raise ValueError("amin must be strictly positive")

#         if torch.is_complex(S):
#             magnitude = S.abs()
#         else:
#             magnitude = S

#         # Check if ref is a callable function or a scalar
#         if callable(self.ref):
#             ref_value = self.ref(magnitude)
#         else:
#             ref_value = torch.abs(torch.tensor(self.ref, dtype=S.dtype))

#         # Compute the log spectrogram
#         log_spec = 10.0 * torch.log10(
#             torch.maximum(magnitude, torch.tensor(self.amin, device=magnitude.device))
#         )
#         log_spec -= 10.0 * torch.log10(
#             torch.maximum(ref_value, torch.tensor(self.amin, device=magnitude.device))
#         )

#         # Apply top_db threshold if necessary
#         if self.top_db is not None:
#             if self.top_db < 0:
#                 raise ValueError("top_db must be non-negative")
#             log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)

#         return log_spec

# Wrapper for pretrained model https://huggingface.co/DBD-research-group/EfficientNet-B1-BirdSet-XCL
# added automatic preprocessing, freeze_encoder(), replace_head(), get_head_input_size()
class PretrainedBirdSetEfficientNet(torch.nn.Module):
    """
    EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g.
    for ImageNet.
    """
    def __init__(self, pretrained_model_name=None):
        super().__init__()
        
        # Load pretrained model
        self.model = EfficientNetForImageClassification.from_pretrained(
            pretrained_model_name,
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
        print(f"Input audio shape: {audio.shape}")
    
        spectrogram = self.spectrogram_converter(audio)
        print(f"After spectrogram: {spectrogram.shape}")
        
        spectrogram = spectrogram.to(torch.float32)
        melspec = self.mel_converter(spectrogram)
        print(f"After mel_converter: {melspec.shape}")
        
        dbscale = self.power_to_db(melspec)
        print(f"After power_to_db: {dbscale.shape}")
        
        normalized_dbscale = transforms.Normalize((-4.268,), (4.569,))(dbscale)
        print(f"After normalize: {normalized_dbscale.shape}")
        
        if normalized_dbscale.dim() == 2:
            normalized_dbscale = normalized_dbscale.unsqueeze(0).unsqueeze(0)
        elif normalized_dbscale.dim() == 3:
            normalized_dbscale = normalized_dbscale.unsqueeze(1)
        
        print(f"Final output shape: {normalized_dbscale.shape}")
        return normalized_dbscale
    
    def forward(self, audio): 
        """Forward pass with automatic preprocessing"""
        pixel_values = self.preprocess(audio)
        return self.model(pixel_values)
    
    def freeze_encoder(self):
        for param in self.model.efficientnet.parameters():
            param.requires_grad = False

    def replace_head(self, new_head):
        self.model.classifier = new_head

    def get_head_input_size(self):
        return self.model.classifier.in_features
    
    def get_config(self):
        return self.model.config
    
    def get_sampling_rate(self):
        return self.sampling_rate
    
class SimpleRegressionHead(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.2):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.regression_head(x)

        




