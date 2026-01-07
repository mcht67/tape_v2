# import matplotlib
# matplotlib.use("Agg")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import numpy as np
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Sequence, Value
from omegaconf import OmegaConf
from utils.general import reshape_tensor_data
from utils.logs import return_tensorboard_path, plot_confusion_matrix, CustomSummaryWriter, CustomSummaryWriterCallback
import os
from utils.config import set_random_seeds, Params
import datetime
from pathlib import Path
import model
from hydra.utils import instantiate
from functools import partial

from utils.general import build_event_logits
from utils.dsp import num_samples_to_duration_s

def split_dataset(test_split, val_split, dataset, random_seed):
    # Split into train/test first (e.g., 90/10) -> test size = 0.1 * number of items
    train_test = dataset.train_test_split(test_size=test_split, shuffle=True, seed=random_seed)

    # Split the training set further to create validation (e.g., 80/10/10) 0.1 * number of items = x * 0.9 * number of items -> x = 0.1 / 0.9 = 0.11
    val_split_factor = val_split / (1 - test_split) 
    train_val = train_test['train'].train_test_split(test_size=val_split_factor, shuffle=True, seed=random_seed)  # 0.11 * 0.9 = 0.1 of total

    # TODO: Add stratified splitting

    # Create the final dataset dictionary
    return DatasetDict({
        'train': train_val['train'],      
        'validation': train_val['test'],   
        'test': train_test['test']        
    })

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

    test_dataset = dataset_splits['test'].to_tf_dataset(
        columns=features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=False,
        prefetch=False
    )

    val_dataset = dataset_splits['validation'].to_tf_dataset(
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

# def plot_to_image(figure):
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     image = tf.image.decode_png(buf.getvalue(), channels=4)
#     image = tf.expand_dims(image, 0)
#     plt.close(figure)
#     return image


# Configuration
cfg = OmegaConf.load("params.yaml")

# Load the hyperparameters from the "params.yaml" file for usage with Tensorboard SummaryWriter
params = Params()

random_seed = cfg.general.random_seed
set_random_seeds(random_seed)

test_split = cfg.train.test_split
val_split = cfg.train.val_split
epochs = cfg.train.epochs

features = cfg.train.features 
labels = cfg.train.labels

batch_size = cfg.train.batch_size

dataset_path =  cfg.paths.dataset

for run in ["run"]:

    # Setup tensorboard
    # If defined in cfg use tensorboard path (define it in params.yaml for debugging purposes)
    # else use return_tensorboard_path (default with dvc run)
    if 'tensorboard_path' in cfg.train.keys():
        default_dir = os.getcwd()
        #dvc_exp_name = 'debug'
        current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        os.environ['DEFAULT_DIR'] = default_dir
        tensorboard_subfolder = f'{cfg.dataset.subset}/{cfg.train.features}'
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
    print(tensorboard_path)
    
    # Load dataset
    dataset = load_from_disk(dataset_path)

    def add_duration(example):
        example["segment_duration_s"] = 5
        return example
    
    dataset = dataset.map(
        add_duration,
        keep_in_memory=False,
    )

    # Add event logits
    def add_event_logits(example, num_event_logits, logits_name):
        all_events = []
        for events in example['raw_files_time_freq_bounds']:
            all_events.extend(events)

        # sampling_rate = example['audio']['sampling_rate']
        # segment_sum_samples = len(example['audio']['array'])
        segment_duration_s = example['segment_duration_s'] #num_samples_to_duration_s(segment_sum_samples, sampling_rate)
        event_logits = build_event_logits(all_events, segment_duration_s, num_event_logits)

        example[logits_name] = event_logits

        return example

    add_event_logits_fn = partial(add_event_logits, num_event_logits=16, logits_name='perch2_event_logits')
    #dataset = dataset.cast_column('audio', Audio()) 
    dataset = dataset.map(add_event_logits_fn, keep_in_memory=False)
    event_logits_feature = Sequence(Value("float32"))
    dataset = dataset.cast_column('perch2_event_logits', event_logits_feature)

    # Split dataset
    dataset_splits = split_dataset(test_split, val_split, dataset, random_seed)

    # Get input dim
    embeddings = dataset_splits['train'][0][features]
    print(np.shape(embeddings))
    input_dim = tf.squeeze(np.array(dataset_splits['train'][0][features])).shape
    train_data = dataset_splits['train']

    # Get tensorflow datasets
    train_dataset, test_dataset, val_dataset = get_tf_datasets(dataset_splits, features, labels, batch_size)

    # Squeeze singelton dimension
    # TODO: Remove once recomputed spatial embeddings with tf.squeeze
    train_dataset = train_dataset.map(
        lambda x, y: (tf.squeeze(x, axis=1), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset = val_dataset.map(
        lambda x, y: (tf.squeeze(x, axis=1), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    example = train_dataset.take(0)

    # Define model
    model = instantiate(cfg.model)#, input_dim=input_dim)

    # Create a SummaryWriter object to write the tensorboard logs
    metrics = {'loss': None, 'val_loss': None, 'mae': None, 'val_mae': None}

    # Add hParams
    params['dataset']['train_size'] = str(len(dataset_splits['train']))
    params['dataset']['val_size'] = str(len(dataset_splits['validation']))
    params['dataset']['test_size'] = str(len(dataset_splits['test']))
    print(params)

    # Define loss weights as variables
    event_loss_weight = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    count_loss_weight = tf.Variable(0.3, trainable=False, dtype=tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()

    def weighted_event_loss(y_true, y_pred):
        return event_loss_weight * bce(y_true, y_pred)

    def weighted_count_loss(y_true, y_pred):
        return count_loss_weight * mse(y_true, y_pred)
    class LossWeightScheduler(tf.keras.callbacks.Callback):
        def __init__(self, switch_epochs, event_loss_weights, count_loss_weights):
            super().__init__()
            self.switch_epochs = switch_epochs
            self.event_loss_weights = event_loss_weights
            self.count_loss_weight = count_loss_weights

        def on_epoch_begin(self, epoch, logs=None):
            for switch_epoch, event_weight, count_weight in zip(self.switch_epochs, self.event_loss_weights, self.count_loss_weight):
                if epoch == switch_epoch:
                    event_loss_weight.assign(event_weight)
                    count_loss_weight.assign(count_weight)

                    print(
                        f"\n[LossWeightScheduler] "
                        f"Switched loss weights at epoch {epoch}: "
                        f"event={event_weight}, "
                        f"count={count_weight}"
                    )
    # class LossWeightScheduler(tf.keras.callbacks.Callback):
    #     def __init__(self, switch_epoch):
    #         super().__init__()
    #         self.switch_epoch = switch_epoch

    #     def on_epoch_begin(self, epoch, logs=None):
    #         if epoch == self.switch_epoch:
    #             event_loss_weight.assign(0.5)
    #             count_loss_weight.assign(1.0)

    #             print(
    #                 f"\n[LossWeightScheduler] "
    #                 f"Switched loss weights at epoch {epoch}: "
    #                 f"event={event_loss_weight.numpy()}, "
    #                 f"count={count_loss_weight.numpy()}"
    #             )

    model.build(input_dim)
    print(f'Input shape of model: {input_dim}')
    writer = CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics, sync_interval=0)
    tensorboard_callback = CustomSummaryWriterCallback(writer=writer, include_standard_tensorboard=True, val_dataset=val_dataset, 
                 log_confusion_matrix=True, confusion_matrix_frequency=5, input_shape=input_dim)

    # Train model
    #model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    learning_rate = 0.0001
    # model.compile(
    #     optimizer=Adam(learning_rate),
    #     loss={
    #         "perch2_event_logits": BinaryCrossentropy(from_logits=True),
    #         "polyphony_degree": MeanSquaredError(),
    #     },
    #     loss_weights={
    #         "perch2_event_logits": 0.5,
    #         "polyphony_degree": 2.0,
    #     },
    # )
    model.compile(
    optimizer=Adam(learning_rate),
    loss={
        "perch2_event_logits": weighted_count_loss,
        "polyphony_degree": weighted_event_loss,
    },
)

    model.summary()

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[LossWeightScheduler(switch_epochs=[0,10,20,30,40], event_loss_weights=[1.0, 1.0, 1.0, 0.5, 0.1], count_loss_weights=[0.1, 0.5, 1.0, 1.0, 2.0]),tensorboard_callback])
    #model.save('tape.keras')

dataset_splits.cleanup_cache_files()