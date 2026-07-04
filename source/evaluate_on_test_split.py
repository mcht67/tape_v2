from omegaconf import OmegaConf
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from hydra.utils import instantiate
from dotenv import load_dotenv
from datasets import load_dataset, Audio, load_from_disk


from utils.logs import SummaryWriter, save_to_report
from utils.metrics import compute_polyphony_metrics
import integrations.birdset as birdset
import integrations.perch as perch

def main():

    #################################
    # Configuration
    #################################
    cfg = OmegaConf.load("params.yaml")

    study_name = cfg.log.study_name

    huggingface_path = cfg.dataset.huggingface_path
    train_dataset_config = cfg.dataset.train_config
    soundscape_dataset_config = cfg.dataset.soundscape_config

    log_dir = cfg.path.eval_log_dir
    checkpoint_dir = cfg.path.checkpoint_dir
    default_dir = os.environ.get('DEFAULT_DIR', '')
    os.makedirs(log_dir, exist_ok=True)

    model_cfg = cfg.model
    objectives_cfg = cfg.objectives
    model_cfg.objectives_cfg = objectives_cfg
    labels = [objectives_cfg[x]['label'] for x in objectives_cfg]
    input_feature_name = cfg.train.input_feature_name
    input_feature = cfg.train.input_feature
    embedding_type = cfg.embeddings.type
    embedding_dim_type = cfg.embeddings.dimension_type

    num_examples = cfg.evaluation.num_examples

    #################################
    # Setup
    #################################

    # Get summary writer for TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Store config in file/logs
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, os.path.join(log_dir, "params.yaml"))

    # #################################
    # # Load dataset
    # #################################

    # val_dataset = tf.data.Dataset.load(val_dataset_path)

    # if not val_dataset:
    #     raise ValueError(f"Validation dataset not found at {val_dataset_path}. Please make sure to run the training script first to save the validation dataset for later evaluation.")

    #################################
    # Load dataset
    #################################

    # Load environment variables from .env file
    load_dotenv('local.env')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

    # from huggingface_hub import HfApi
    # api = HfApi()
    # info = api.dataset_info("mcht67/Polyphonic-BirdSet-train", token=huggingface_token)
    # for config in info.card_data.get("configs", []):
    #     if config.get("config_name") == train_dataset_config:
    #         print(config.get("data_files"))
    #         print(config.get("data_dir"))

    # # Load Dataset
    # print(f"[INFO] HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', 'NOT SET')} (ommits updating datasets to avoid hitting rate limit on Huggingface Hub)")
    # train_dataset = load_dataset(huggingface_path, train_dataset_config, token=huggingfce_token, streaming=True)

    from datasets import Dataset
    # test_dataset = load_dataset(huggingface_path, train_dataset_config, split='test', token=huggingface_token)
    # train_dataset = load_from_disk('data/HSN')
    test_dataset = load_dataset(huggingface_path, train_dataset_config, split='test', token=huggingface_token, streaming=True)
    print("Dataset loaded. Converting to in-memory format for processing...")
    test_dataset = Dataset.from_list(list(test_dataset.take(2)))

    if test_dataset is None:
        raise RuntimeError("Dataset failed to load after all retry attempts. Check network/cache or force redownload in dataset preparation.")

    # train_split, val_split, test_split

    #################################
    # Add labels
    #################################
    # TODO: do per example to avoid downloading entire dataset / is already been done in train.py

    # Get input dim
    # input_dim = tf.squeeze(np.array(train_dataset['train'][0][input_feature_name])).shape

    # Compute additional labels
    # time_dim = input_dim[0] if len(input_dim) > 1 else None
    # freq_dim = input_dim[1] if len(input_dim) > 2 else None
    # train_dataset, added_labels = add_labels(train_dataset, labels, time_dim=time_dim, freq_dim=freq_dim)

    # print("Added labels: ", added_labels)

    # existing_labels = added_labels + ['polyphony_degree']
    # missing_labels = set(labels) ^ set(existing_labels)
    # if missing_labels:
    #     raise Exception("Not all requested labels could be computed.")

    #################################
    # Load model
    #################################

    # Load best model from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "best.weights.h5")
    # TODO: remove after testing
    # checkpoint_path = 'archive/Pooled-Embeddings/perch_v2_cpu/20260703_172045_level-arcs/checkpoints/best.weights.h5'
    checkpoint_path = 'archive/Pooled-Embeddings/perch_v2_cpu/20260703_172515_brood-weld/checkpoints/best.weights.h5'
    # checkpoint_path = 'archive/Pooled-Embeddings/perch_v2_cpu/20260609_012305_bosom-byes/checkpoints/best.weights.h5'

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}. Please make sure to run the training script first to save the best model checkpoint for later evaluation.")
    
    # Define model
    # model_cfg.objectives_cfg = objectives_cfg
    model = instantiate(cfg.model)

    # Build model by calling it on a sample input
    first_example = test_dataset[0]
    print("input features", test_dataset.features)
    input_dim = int(tf.squeeze(np.array(first_example[input_feature_name])).shape[0])
    sample_input = tf.zeros((1, input_dim), dtype=tf.float32)
    _ = model(sample_input, training=False)

    print(f"Loading weights from {checkpoint_path}")
    model.load_weights(checkpoint_path)
    print("Model loaded successfully.")

    ###########################################
    # Example visualization in TensorBoard
    # ###########################################

    # for split_name in train_dataset.keys():

    #     for example_idx in range(num_examples):

    #         example = train_dataset[split_name][example_idx]
    #         embedding = example[input_feature_name]
            
    #         # Make prediction
    #         single_input = np.expand_dims(embedding, axis=0)
    #         single_input = tf.constant(single_input, dtype=tf.float32)
    #         predictions = model.predict(single_input)

    #         # Extract data
    #         objectives_list = list(objectives_cfg.keys())
    #         if 'polyphony_degree' in objectives_list:
    #             gt_polyphony = example['polyphony_degree']
    #             pred_polyphony = predictions['polyphony_degree'][0][0]
    #         else:
    #             gt_polyphony = pred_polyphony = None
                
    #         if 'event_logits' in objectives_list:
    #             gt_event_logits = example['event_logits']
    #             pred_event_logits = predictions['event_logits'][0]
    #         else:
    #             gt_event_logits = pred_event_logits = None

    #         if 'framewise_polyphony' in objectives_list:
    #             gt_framewise_polyphony = example['framewise_polyphony']
    #             pred_framewise_polyphony = predictions['framewise_polyphony'][0]
    #         else:
    #             gt_framewise_polyphony = pred_framewise_polyphony = None

    #         # Get audio and events
    #         audio_array = example['audio']['array']
    #         sampling_rate = example['audio']['sampling_rate']

    #         # Get all events
    #         all_events = []
    #         for events in example['sources_time_freq_bounds']:
    #             for event in events:
    #                 all_events.append(event)
            
    #         # Create combined figure
    #         fig = plot_spectrogram_with_metrics(
    #             audio_array=audio_array,
    #             sampling_rate=sampling_rate,
    #             split_name=split_name,
    #             example_idx=example_idx,
    #             gt_polyphony=gt_polyphony,
    #             pred_polyphony=pred_polyphony,
    #             gt_event_logits=gt_event_logits,
    #             pred_event_logits=pred_event_logits,
    #             events=all_events,
    #             filename=example.get('filename', None)
    #         )
            
    #         writer.add_figure(f'{split_name}', fig, global_step=example_idx) # 'example_{idx}'
    #         plt.close(fig)

    # # Flush to ensure all figures are written
    # writer.flush()

    ###########################################
    # Metrics computation on test split
    ###########################################

    def records_to_arrays(records, variables=None, species_names=None):
        """Convert list of per-example records into arrays for metric computation."""
        variables = variables or []
        
        predictions = np.stack([r["prediction"] for r in records])  # (N, num_species) or (N, 1)
        
        if species_names:
            y_true = np.stack([
                [r["y_true"].get(sp, 0) for sp in species_names] for r in records
            ])
        else:
            y_true = np.array([r["y_true"]["polyphony"] for r in records]).reshape(-1, 1)
        
        variable_values = {
            var: np.array([r["variable_values"][var] for r in records])
            for var in variables
        }
        
        return y_true, predictions, variable_values


    def arrays_to_records(y_true, predictions, variable_values, species_names=None):
        """Convert arrays back into self-contained per-example records for storage."""
        n = len(predictions)
        variable_rows = [
            {var: variable_values[var][i] for var in variable_values}
            for i in range(n)
        ]
        
        if species_names:
            y_true_dicts = [dict(zip(species_names, row)) for row in y_true]
        else:
            y_true_dicts = [{"polyphony": val} for val in y_true.flatten()]
        
        records = [
            {"y_true": yt, "prediction": pred, "variable_values": var_row}
            for yt, pred, var_row in zip(y_true_dicts, predictions, variable_rows)
        ]
        return records
    
    def collect_predictions(model, dataset, input_feature_name, variables=None, 
                         species_names=None, batch_size=64):
        """
        Run inference over `dataset` and return arrays ready for metric computation.

        Returns:
            y_true: (N, num_species) if species_names given, else (N, 1)
            predictions: (N, num_species) or (N, 1), matching model output
            variable_values: dict of {var_name: array of shape (N,)}
        """
        variables = variables or []
        embeddings, gt_rows, variable_rows = [], [], []

        predictions = []

        for example in dataset:
            embedding = example[input_feature_name]
            embeddings.append(embedding)
            variable_rows.append({var: example[var] for var in variables})

            if species_names:
                gt_rows.append([example[sp] for sp in species_names])
            else:
                gt_rows.append([example["polyphony_degree"]])

            # # Make prediction
            # single_input = np.expand_dims(embedding, axis=0)
            # single_input = tf.constant(single_input, dtype=tf.float32)
            # predictions = model.predict(single_input)
            # predictions.append(predictions)

        X = tf.constant(np.stack(embeddings), dtype=tf.float32)
        predictions = model.predict(X, batch_size=batch_size)  # single batched call

        y_true = np.array(gt_rows)  # (N, num_species) or (N, 1)

        variable_values = {
            var: np.array([row[var] for row in variable_rows])
            for var in variables
        }

        return y_true, predictions, variable_values

    y_true, predictions, variable_values = collect_predictions(model, test_dataset, input_feature_name)

    # Define ouput dir for metrics and results
    subset = cfg.dataset.subset
    out_dir = os.path.join(default_dir, "archive", cfg.log.study_name, cfg.path.study_subfolder, "eval_results")
    os.makedirs(out_dir, exist_ok=True)

    # Store results in a pickle file for later analysis
    records = arrays_to_records(y_true, predictions, variable_values)
    df = pd.json_normalize(records)
    df.to_pickle(os.path.join(log_dir, f"test_results.pkl"))

    if "polyphony_reg" in predictions:
        df.to_pickle(os.path.join(out_dir, f"{subset}_reg_test_results.pkl"))
    if "polyphony_class" in predictions:
        df.to_pickle(os.path.join(out_dir, f"{subset}_class_test_results.pkl"))
    if "species_polyphony_reg" in predictions:
        df.to_pickle(os.path.join(out_dir, f"{subset}_species_reg_test_results.pkl"))
    if "species_polyphony_class" in predictions:
        df.to_pickle(os.path.join(out_dir, f"{subset}_species_class_test_results.pkl"))

    # Calculate metrics for each objective and save to report
    report = {}

    num_classes = cfg.dataset.max_polyphony + 1
    species_mapping = None # TODO: load from dataset if available

    if "species_polyphony_reg" in predictions:
        report["species_polyphony_reg"] = compute_polyphony_metrics(
            y_true["species_polyphony"], predictions["species_polyphony_reg"],
            cm_type="species_regression_round", species_mapping=species_mapping)

    if "species_polyphony_class" in predictions:
        report["species_polyphony_class"] = compute_polyphony_metrics(
            y_true["species_polyphony"], predictions["species_polyphony_class"],
            cm_type="species_classification", species_mapping=species_mapping,
            num_classes=num_classes)

    if "polyphony_reg" in predictions:
        report["polyphony_reg"] = compute_polyphony_metrics(
            y_true["polyphony"][:, None], predictions["polyphony_reg"][:, None],
            cm_type="species_regression_round", per_species=False)

    if "polyphony_class" in predictions:
        report["polyphony_class"] = compute_polyphony_metrics(
            y_true["polyphony"][:, None], predictions["polyphony_class"][:, None, :],
            cm_type="species_classification", num_classes=num_classes, per_species=False)
        
    save_to_report(report, os.path.join(log_dir, "test_metrics.json"))

    def flatten_metrics(metrics_dict):
        """
        Flatten compute_polyphony_metrics() output into a flat {row_name: value} dict.
        """
        flat = dict(metrics_dict["overall"])

        if "per_species" in metrics_dict:
            for species, species_metrics in metrics_dict["per_species"].items():
                for metric_name, value in species_metrics.items():
                    flat[f"{species}/{metric_name}"] = value

        return flat


    def update_metrics_table(subset_name, metrics_dict, csv_path):
        """
        Add or overwrite the column for `subset_name` in the experiment's metrics table.
        Creates the table if it doesn't exist yet.
        """

        flat_metrics = flatten_metrics(metrics_dict)
        new_col = pd.Series(flat_metrics, name=subset_name)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            df[subset_name] = new_col  # adds new column, or overwrites if it already exists
        else:
            df = new_col.to_frame()

        df.to_csv(csv_path, float_format="%.4f")

        return df, csv_path

    # Save to metrics overview table
    for objective in report:
        metrics_dict = report[objective]
        table_name = f"{objective}_metrics"
        csv_path = os.path.join(out_dir, f"{table_name}.csv")
        df, csv_path = update_metrics_table("test", metrics_dict, csv_path)
        print(f"Saved {table_name} to {csv_path}")

    # for example in dataset['test']:

    #     embedding = example[input_feature_name]
        
    #     # Make prediction
    #     single_input = np.expand_dims(embedding, axis=0)
    #     single_input = tf.constant(single_input, dtype=tf.float32)
    #     predictions = model.predict(single_input)

    #     # Extract data
    #     objectives_list = list(objectives_cfg.keys())
    #     gt_polyphony = example['polyphony_degree']

    # # Handle regression predictions
    # if objectives_cfg.get('polyphony_reg', None) is not None:
    #     pred_polyphony_reg = predictions['polyphony_reg'][0][0]

    # # Handle classification predictions
    # # This expects the polyphony degree to match the index of the class, i.e. class 0 = polyphony 0, class 1 = polyphony 1, etc.
    # if objectives_cfg.get('polyphony_class', None) is not None:
    #     pred_polyphony_class = np.argmax(predictions['polyphony_class'][0])
    

if __name__ == "__main__":
    main()


