# import matplotlib
# matplotlib.use("Agg")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from omegaconf import OmegaConf
from utils.general import reshape_tensor_data
from utils.logs import return_tensorboard_path, CustomSummaryWriter, CustomSummaryWriterCallback
import os
from utils.config import set_random_seeds, Params
import datetime
from pathlib import Path
import model
from hydra.utils import instantiate
from functools import partial

from utils.logs import plot_spectrogram_with_metrics

def make_tf_dataset(hf_dataset, features, labels, batch_size, shuffle=False):
    X = np.stack(hf_dataset[features]).astype(np.float32)
    y = np.array(hf_dataset[labels]).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size)
    return ds


def get_tf_datasets(dataset, features, labels, batch_size):
    train_dataset = dataset['train'].to_tf_dataset(
        columns=features,
        label_cols=labels, 
        batch_size=batch_size,
        shuffle=True,
        prefetch=False
    )

    test_dataset = dataset['test'].to_tf_dataset(
        columns=features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=False,
        prefetch=False
    )

    val_dataset = dataset['validation'].to_tf_dataset(
        columns=features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=False,
        prefetch=False
    )

    return train_dataset, test_dataset, val_dataset

def apply_batched_reshape(dataset, features, input_dim, pooling_strategy, suffix, batch_size=100):
    """Apply reshape_tensor_data in batches to avoid PyArrow offset overflow"""
    
    # Process in batches
    processed_datasets = []
    total_samples = len(dataset)
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        print(f"Processing reshape batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
        
        # Select batch
        batch_dataset = dataset.select(range(i, end_idx))
        
        # Process batch
        batch_processed = batch_dataset.map(
            lambda x: reshape_tensor_data(x, features, input_dim, pooling_strategy, suffix),
            desc=f"Reshaping batch {i//batch_size + 1}"
        )
        
        processed_datasets.append(batch_processed)
    
    # Concatenate all processed batches
    print(f"Concatenating {len(processed_datasets)} reshape batches...")
    return concatenate_datasets(processed_datasets)

def get_predictions_and_true_labels(model, dataset):
    y_pred = []
    y_true = []

    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_pred.append(preds)
        y_true.append(y_batch.numpy())

    return y_pred, y_true

# Setup tensorboard
# If defined in cfg use tensorboard path (define it in params.yaml for debugging purposes)
# else use return_tensorboard_path (default with dvc run)
def get_tensorboard_path(cfg):
    if 'tensorboard_path' in cfg.train.keys():
        default_dir = os.getcwd()
        #dvc_exp_name = 'debug'
        current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        os.environ['DEFAULT_DIR'] = default_dir
        tensorboard_subfolder = f'{cfg.dataset.subset}/{cfg.train.input_feature_name}'
        tensorboard_path_suffix = f'_{features}'

        tensorboard_path = Path(
            f"{default_dir}/{cfg.train.tensorboard_path}/{tensorboard_subfolder}/{tensorboard_path_suffix}{current_datetime}"
        )
        
    else:
        tensorboard_path = return_tensorboard_path()
        tensorboard_subfolder = f'{cfg.dataset.subset}'
        tensorboard_path_suffix = f'_{features}'
        tensorboard_path = return_tensorboard_path(subfolder=tensorboard_subfolder, suffix=tensorboard_path_suffix) # './logs/' + features + version #return_tensorboard_path()
    os.makedirs(tensorboard_path, exist_ok=True)
    return tensorboard_path

class LossWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(
        self,
        switch_epochs,
        event_loss_weights,
        frame_loss_weights,
        count_loss_weights,
    ):
        super().__init__()
        self.switch_epochs = switch_epochs
        self.event_loss_weights = event_loss_weights
        self.frame_loss_weights = frame_loss_weights
        self.count_loss_weights = count_loss_weights

    def on_epoch_begin(self, epoch, logs=None):
        for e, ew, fw, cw in zip(
            self.switch_epochs,
            self.event_loss_weights,
            self.frame_loss_weights,
            self.count_loss_weights,
        ):
            if epoch == e:
                event_loss_weight.assign(ew)
                frame_loss_weight.assign(fw)
                count_loss_weight.assign(cw)

                print(
                    f"\n[LossWeightScheduler] epoch {epoch} | "
                    f"event={ew}, frame={fw}, count={cw}"
                )

# Configuration
cfg = OmegaConf.load("params.yaml")

# Load the hyperparameters from the "params.yaml" file for usage with Tensorboard SummaryWriter
params = Params()

random_seed = cfg.general.random_seed
set_random_seeds(random_seed)

epochs = cfg.train.epochs

features = cfg.train.input_feature_name
labels = cfg.train.labels

batch_size = cfg.train.batch_size

dataset_path =  cfg.path.dataset

for run in ["run"]:

    # Get tensorboard path based on path, dataset subset, features and datetime
    # Set DEFAULT_DIR if not set (usually when running without dvc)
    os.environ.setdefault('DEFAULT_DIR', os.getcwd())
    os.environ.setdefault('DVC_EXP_NAME', 'test-experiment')
    tensorboard_subfolder = cfg.log.tensorboard_subfolder
    tensorboard_suffix = cfg.log.tensorboard_suffix
    tensorboard_path = return_tensorboard_path(subfolder=tensorboard_subfolder, suffix=tensorboard_suffix ) #get_tensorboard_path(cfg) #TODO: refactor to work in a similar manner with dvc and without
    os.makedirs(tensorboard_path, exist_ok=True)

    # Load dataset
    dataset = load_from_disk(dataset_path)

    # Get input dim
    embeddings = dataset['train'][0][features]
    input_dim = tf.squeeze(np.array(dataset['train'][0][features])).shape


    # Get tensorflow datasets
    train_dataset, test_dataset, val_dataset = get_tf_datasets(dataset, features, labels, batch_size)
    #train_dataset = train_dataset.take(100) # take fewer batches to reduce train dataset size

    # Define model
    model = instantiate(cfg.model)

    # Create a SummaryWriter object to write the tensorboard logs
    metrics = {'loss': None, 'val_loss': None, 'mae': None, 'val_mae': None} # TODO: Investigate: What is this doing with the variable losses?

    # Add hParams
    params['dataset']['train_size'] = str(len(dataset['train']))
    params['dataset']['val_size'] = str(len(dataset['validation']))
    params['dataset']['test_size'] = str(len(dataset['test']))
    print(params)

    # Define loss weights as variables
    event_loss_weight = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    count_loss_weight = tf.Variable(0.3, trainable=False, dtype=tf.float32)
    frame_loss_weight = tf.Variable(1.0, trainable=False, dtype=tf.float32)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()
    frame_mse = tf.keras.losses.MeanSquaredError()

    def weighted_event_loss(y_true, y_pred):
        return event_loss_weight * bce(y_true, y_pred)

    def weighted_count_loss(y_true, y_pred):
        return count_loss_weight * mse(y_true, y_pred) 

    def weighted_frame_loss(y_true, y_pred):
        return frame_loss_weight * frame_mse(y_true, y_pred)

    model.build(input_dim)
    print(f'Input shape of model: {input_dim}')

    # Define confusion matrix specs
    confusion_matrix_specs = [
            {"name": "polyphony_degree", "type": "regression_round", "threshold": 0.5}, 
            {"name": "perch2_event_logits", "type": "binary", "threshold": 0.5,},
        ]

    writer = CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics, sync_interval=0)
    tensorboard_callback = CustomSummaryWriterCallback(writer=writer, include_standard_tensorboard=True, val_dataset=val_dataset, 
                 log_confusion_matrix=True, confusion_matrix_frequency=5, confusion_matrix_specs=confusion_matrix_specs, input_shape=input_dim)

    # Train model
    #model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    learning_rate = 0.0001
    model.compile(
        optimizer=Adam(learning_rate),
        loss={
            "perch2_event_logits": weighted_event_loss,
            "framewise_polyphony": weighted_frame_loss,
            "polyphony_degree": weighted_count_loss, 
        },
    )
    model.summary()
    print(len(train_dataset))
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[LossWeightScheduler(switch_epochs=[0, 10, 20, 30, 40], event_loss_weights=[10.0, 5.0, 1.0, 0.5, 0.1], frame_loss_weights=[5.0, 10.0, 5.0, 1.0, 0.5], count_loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0]),tensorboard_callback]) #LossWeightScheduler(switch_epochs=[0,10,20,30,40], event_loss_weights=[1.0, 1.0, 1.0, 0.5, 0.1], count_loss_weights=[0.1, 0.5, 1.0, 1.0, 2.0])
    
    # TODO: Save usable version
    model.save('my_model.keras')

    # TODO: Store config in file/logs
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, os.path.join(tensorboard_path, "params.yaml"))

    # for example in [dataset['train'][220], dataset['train'][20], dataset['validation'][110], dataset['test'][110]]:
    #     embedding = example['perch_v2_cpu_spatial_embeddings_audio']
    #     # Extract the features and add batch dimension
    #     single_input = np.expand_dims(embedding, axis=0)  # Add batch dim
    #     single_input = tf.constant(single_input, dtype=tf.float32)

    #     # Squeeze if needed (depends on your data)
    #     #single_input = tf.squeeze(single_input, axis=1)  # If there's an extra dimension
    #     predictions = model.predict(single_input)
    #     print(predictions)
    #     print(example['polyphony_degree'])
    #     print(example['perch2_event_logits'])
    #     print(example['framewise_polyphony'])

    # Add some examples to tensorboard
    import matplotlib.pyplot as plt

    # Get some examples
    for idx, (split_name, example_idx) in enumerate([
        ('train', 2), 
        ('train', 23), 
        ('validation', 2), 
        ('validation', 23), 
        ('test', 2),
        ('test', 23)
    ]):
        example = dataset[split_name][example_idx]
        embedding = example[features]
        
        # Make prediction
        single_input = np.expand_dims(embedding, axis=0)
        single_input = tf.constant(single_input, dtype=tf.float32)
        predictions = model.predict(single_input)
        
        # Extract data
        gt_polyphony = example['polyphony_degree']
        gt_event_logits = example['perch2_event_logits']
        pred_polyphony = predictions['polyphony_degree'][0][0]
        pred_event_logits = predictions['perch2_event_logits'][0]
        
        # Get audio and events
        audio_array = example['audio']['array']  # Adjust based on your data structure
        sampling_rate = example['audio']['sampling_rate']

        # Get all events
        all_events = []
        for events in example['raw_files_time_freq_bounds']:
            for event in events:
                print(event)
                all_events.append(event)
        
        # Create combined figure
        fig = plot_spectrogram_with_metrics(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            split_name=split_name,
            example_idx=example_idx,
            gt_polyphony=gt_polyphony,
            pred_polyphony=pred_polyphony,
            gt_event_logits=gt_event_logits,
            pred_event_logits=pred_event_logits,
            events=all_events,
            filename=example.get('filename', None)
        )
        
        writer.add_figure('test_examples_{idx}', fig, global_step=idx)
        plt.close(fig)

    # Flush to ensure all figures are written
    writer.flush()

dataset.cleanup_cache_files()

