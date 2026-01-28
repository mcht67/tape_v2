import os
import torch

from datasets import load_from_disk, Audio
from omegaconf import OmegaConf
from hydra.utils import instantiate

from utils.config import set_random_seeds, Params
#from utils.logs import plot_spectrogram_with_metrics, return_checkpoint_path, return_tensorboard_dir, CustomSummaryWriter, CustomSummaryWriterCallback, build_confusion_matrix_specs

from torch.utils.data import DataLoader, Dataset

from models_torch import SimpleRegressionHead

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, features, labels):
        self.dataset = hf_dataset
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Extract features - handle Audio objects
        feature_values = []
        for col in self.features:
            if isinstance(item[col], dict) and 'array' in item[col]:
                # It's an Audio object - extract the array
                feature_values.append(item[col]['array'])
            else:
                # Regular feature
                feature_values.append(item[col])
        
        # Convert to tensor
        features_tensor = torch.tensor(feature_values[0], dtype=torch.float32)
        
        # Extract labels
        label_tensor = torch.tensor(
            [item[col] for col in self.labels], 
            dtype=torch.float32  # or torch.long for classification
        )
        
        return features_tensor, label_tensor

def get_torch_dataloaders(dataset, features, labels, batch_size):
    train_dataset = HFDatasetWrapper(dataset['train'], features, labels)
    test_dataset = HFDatasetWrapper(dataset['test'], features, labels)
    val_dataset = HFDatasetWrapper(dataset['validation'], features, labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # parallel data loading
        pin_memory=True  # faster transfer to GPU
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader, val_loader

def main():

    ###################################################
    # Configuration
    ###################################################
    cfg = OmegaConf.load("params.yaml")

    experiment_name = cfg.log.experiment_name
    random_seed = cfg.general.random_seed
    
    dataset_path =  cfg.path.dataset

    input_feature_name = 'audio' #cfg.train.input_feature_name
    epochs = cfg.train.epochs
    learning_rate = cfg.train.learning_rate
    batch_size = cfg.train.batch_size
    if 'train_size_batches' in cfg.train: 
        train_size_batches = cfg.train.train_size_batches
    else:
        train_size_batches = None

    objectives_cfg = cfg.objectives
    objectives_list = list(objectives_cfg.keys()) 

    model_cfg = cfg.model

    tensorboard_subfolder = cfg.log.tensorboard_subfolder
    tensorboard_suffix = cfg.log.tensorboard_suffix

    ###################################################
    # Initialization
    ###################################################

    # Get tensorboard path based on path, dataset subset, features and datetime
    # Set DEFAULT_DIR if not set (usually when running without dvc)
    os.environ.setdefault('DEFAULT_DIR', os.getcwd())
    os.environ.setdefault('DVC_EXP_NAME', 'test-experiment')

    # tensorboard_path = return_tensorboard_dir(subfolder=experiment_name)
    # os.makedirs(tensorboard_path, exist_ok=True)

    # Load the hyperparameters from the "params.yaml" file for usage with Tensorboard SummaryWriter
    params = Params()

    set_random_seeds(random_seed)

    ###################################################
    # Prepare Model
    ###################################################

    # Instantiate model with merged config
    print(model_cfg)
    model = instantiate(model_cfg)  # instantiate model from config 
                                    # -> handle preprocessing etc. in own wrapper for every birdset models
    
    config = model.config
    print(config)

    default_sampling_rate = 32000
    
    # Set sampling rate from model
    # if not set in model fall back to default and set it in model
    if not (sampling_rate := model.sampling_rate):
        sampling_rate = default_sampling_rate
        model.set_sampling_rate(sampling_rate)
    
    input_size = model.get_head_input_size()
    model.replace_head(SimpleRegressionHead(input_size))
    model.freeze_encoder()

    ###################################################
    # Check model
    ###################################################
    # Check trainable parameters
    print(f"Amount of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Amount of freezed parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    from torchvision import transforms
    import requests
    import torchaudio
    import io

    # url = "https://xeno-canto.org/704485/download"
    # response = requests.get(url)

    import librosa
    # Load an example audio file
    audio_path = librosa.ex('robin')

    # The model is trained on audio sampled at 32,000 Hz
    audio, sample_rate = torchaudio.load(audio_path)
    duration = librosa.get_duration(y=audio, sr=sample_rate)
    print(duration)
    # audio, sample_rate = torchaudio.load(io.BytesIO(response.content), format="mp3")
    print("Original shape and sample rate: ", audio.shape, sample_rate)
    # crop to 5 seconds
    audio = audio[:, : 10 * sample_rate]
    duration = librosa.get_duration(y=audio, sr=sample_rate)
    print(duration)
    # resample to 32kHz
    resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=32000)
    audio = resample(audio)
    print("Resampled shape and sample rate: ", audio.shape, 32000)

    outputs = model(audio)

    logits = outputs.logits
    print("Logits shape:", logits.shape)
    print("Logits:", logits)

    embeddings = outputs.pooled_embeddings
    print("Embeddings shape:", embeddings.shape)
    print("Embeddings:", embeddings)

    spatial_embeddings = outputs.spatial_embeddings
    print("Spatial mbeddings shape:", spatial_embeddings.shape)
    print("Spatial embeddings:", spatial_embeddings)


    ###################################################
    # Prepare Dataset
    ###################################################

    # Load dataset
    dataset = load_from_disk(dataset_path)

    for split in dataset:
        dataset[split] = dataset[split].take(50)
        dataset[split].cast_column(input_feature_name, Audio(sampling_rate=sampling_rate))

    ###################################################
    # Prepare Metrics and HyperParams
    ###################################################

    ###################################################
    # Prepare Callbacks
    ###################################################

    ###################################################
    # Train Loop
    ###################################################
    features = [input_feature_name]
    labels = objectives_list

    train_loader, test_loader, val_loader = get_torch_dataloaders(
        dataset=dataset,
        features=features,
        labels=labels,
        batch_size=batch_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 5

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            logits = outputs.logits
            loss = criterion(logits, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1} loss: {loss.item()}")
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    ###################################################
    # Validation on examples
    ###################################################

if __name__=="__main__":
    main()