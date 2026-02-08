from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Union
from transformers import EfficientNetForImageClassification, AutoFeatureExtractor, AutoModel 
from transformers.modeling_outputs import ModelOutput

import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms

from birdset.datamodule.components.augmentations import PowerToDB

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
       
class BirdSetBirdMAE(torch.nn.Module):
    """
    Wrapper for pretrained Bird-MAE Model. Original model: https://huggingface.co/DBD-research-group/Bird-MAE-Base
    """
    def __init__(self, pretrained_model_path="DBD-research-group/Bird-MAE-Huge"):
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

    def preprocess(self, audio):
        mel_spectrogram = self.feature_extractor(audio)
        return mel_spectrogram
    
    def forward(self, audio): 
        """Forward pass with automatic preprocessing and optional pooling and output head"""
        logits = None
        pooled_output = None

        mel_spectrogram = self.preprocess(audio)
        outputs = self.model(mel_spectrogram)
        last_hidden_state = outputs.last_hidden_state

        if not self.pooling:
            x = last_hidden_state
        elif self.pooling=='mean':
            pooled_output = last_hidden_state.mean(dim=[2,3])
            x = pooled_output
        else:
            raise ValueError(f"Pooling option {self.pooling} not supported")
    
        if self.output_head:
            logits = self.output_head(x)

        return EmbeddingModelOutput(
        pooled_embeddings=pooled_output,
        spatial_embeddings=last_hidden_state,
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
            self._head_input_size = x.shape
            
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
        self.spectrogram_converter = None
        self.mel_converter = None
        self.amplitude_to_db = None
        self.pooling = pooling

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

        # Init preprocessing signal converters if not initialized
        if not self.spectrogram_converter or not self.mel_converter or not self.amplitude_to_db:
            
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

        if self.pooling:
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
        self.sampling_rate = new_sampling_rate

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

    def preprocess(self, audio):
        return self.model(audio)
    
    def forward(self, audio): 
        """Forward pass with automatic preprocessing and optional pooling and output head"""
        logits = None
        pooled_output = None

        #mel_spectrogram = self.preprocess(audio)
        outputs = self.model(audio)
        if self.spatial_embeddings=="last_hidden_state":
            spatial_embeddings = outputs.last_hidden_state
        elif self.spatial_embeddings=="xvector":
            spatial_embeddings = outputs.extract_features
        else:
            ValueError(f"Spatial embeddings key is not valid. Use 'last_hidden_state' or 'xvector'")
    
        if not self.pooling:
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
        spatial_embeddings=spatial_embeddings,
        logits=logits
    )
    
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def replace_head(self, new_head: torch.nn.Module):
        self.output_head = new_head

    def get_head_input_size(self):
        return self.config.output_hidden_size

