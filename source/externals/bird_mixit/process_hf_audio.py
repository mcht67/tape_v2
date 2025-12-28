# bird_mixit/process_wav.py
"""
Functions to run source separation using a pretrained MixIT or similar model.

This version refactors the original Google Research `process_wav.py` script
so that it can be called programmatically, not just via CLI.

Example:

    from bird_mixit import process_wav
    from datasets import load_dataset

    # Load an example HuggingFace Audio object
    ds = load_dataset("ashraq/esc50")  # for example
    audio_obj = ds["train"][0]["audio"]

    model = process_wav.load_model("/path/to/model_dir")

    separated = process_wav.separate_sources(audio_obj, model)
"""

import os
import tensorflow.compat.v1 as tf
import numpy as np
#from . import inference  # this must be in the same package or PYTHONPATH
from librosa import resample

tf.disable_v2_behavior()


# ------------------------------
# Utility functions
# ------------------------------

def load_separation_model(model_dir,
                          input_tensor='input_audio/receiver_audio:0',
                          output_tensor='denoised_waveforms:0',
                          checkpoint=None):

    tf.disable_v2_behavior()
    tf.reset_default_graph()

    meta_graph_filename = os.path.join(model_dir, 'inference.meta')
    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph(meta_graph_filename)
        meta_graph_def = graph.as_graph_def()

    checkpoint = checkpoint or tf.train.latest_checkpoint(model_dir)

    session = tf.Session(graph=graph)
    saver.restore(session, checkpoint)

    input_node = graph.get_tensor_by_name(input_tensor)
    output_node = graph.get_tensor_by_name(output_tensor)

    return session, input_node, output_node


def separate_audio(session,
                   input_node,
                   output_node,
                   audio_array,
                   sampling_rate,
                   target_sr=22050):

    # Enforce mono
    if audio_array.ndim != 1:
        audio_array = np.mean(audio_array, axis=0)

    # Resample if needed
    if sampling_rate != target_sr:
        audio_array = resample(audio_array,
                               orig_sr=sampling_rate,
                               target_sr=target_sr)

    audio_array = audio_array.astype(np.float32)
    audio_array = audio_array[np.newaxis, np.newaxis, :]  # [1, 1, samples]

    output_numpy = session.run(output_node,
                               feed_dict={input_node: audio_array})

    output_numpy = np.squeeze(output_numpy, axis=0)  # [sources, samples]
    return output_numpy, target_sr

def separate_batch(session,
                  input_node,
                  output_node,
                  audio_list,
                  sampling_rate):

    results = []
    for audio in audio_list:
        separated, sr = run_separation(session,
                                       input_node,
                                       output_node,
                                       audio,
                                       sampling_rate)
        results.append(separated)
    return results, sr

def separate_batch_stacked(session,
                         input_node,
                         output_node,
                         audio_list,
                         sampling_rate,
                         target_sr=22050):
    """
    Process a batch of numpy arrays using a single model execution.

    Returns:
        output_batch: numpy array with shape [N, num_sources, samples]
        target_sr: sampling rate (22.05 kHz for this model)
    """

    processed = []

    # Preprocess individually: enforce mono, resample, convert dtype
    for audio in audio_list:
        if audio.ndim != 1:
            audio = np.mean(audio, axis=0)

        if sampling_rate != target_sr:
            audio = resample(audio, orig_sr=sampling_rate, target_sr=target_sr)

        processed.append(audio.astype(np.float32))

    # Determine maximum length and pad to uniform size
    max_len = max(len(x) for x in processed)
    padded = [np.pad(x, (0, max_len - len(x))) for x in processed]

    # Stack into a batch tensor [N, 1, samples]
    batch_input = np.stack(padded, axis=0)
    batch_input = batch_input[:, np.newaxis, :]

    # Forward pass once for all inputs
    output_batch = session.run(output_node, feed_dict={input_node: batch_input})

    # Shape typically becomes [N, num_sources, samples]
    return output_batch, target_sr





# ------------------------------
# OLD FUNCTIONS BELOW
# ------------------------------

def load_model(model_dir, checkpoint=None, input_tensor='input_audio/receiver_audio:0',
               output_tensor='denoised_waveforms:0'):
    """
    Load the TensorFlow 1 model graph and return a session + tensors.
    """
    meta_graph_filename = os.path.join(model_dir, 'inference.meta')
    tf.logging.info(f'Importing meta graph: {meta_graph_filename}')

    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph(meta_graph_filename)
        meta_graph_def = g.as_graph_def()

    tf.reset_default_graph()

    # Placeholder waveform to feed later
    input_placeholder = tf.placeholder(tf.float32, shape=[None, None, None], name="input_placeholder")

    # Import actual model graph
    output_tensor_ref, = tf.import_graph_def(
        meta_graph_def,
        name='',
        input_map={input_tensor: input_placeholder},
        return_elements=[output_tensor]
    )

    # Squeeze dimensions
    output_tensor_ref = tf.squeeze(output_tensor_ref, 0)
    output_tensor_ref = tf.transpose(output_tensor_ref)

    # Find checkpoint if not provided
    if checkpoint is None:
        checkpoint = tf.train.latest_checkpoint(model_dir)

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info(f'Restoring model from checkpoint: {checkpoint}')
    saver.restore(sess, checkpoint)

    return {
        "session": sess,
        "input_placeholder": input_placeholder,
        "output_tensor": output_tensor_ref,
    }


# ------------------------------
# Complete separation
# ------------------------------
def complete_separation_numpy(input,
                        sampling_rate,
                        #output,
                        model_dir,
                        #input_channels=0,
                        num_sources=2,
                        output_channels=0,
                        input_tensor='input_audio/receiver_audio:0', 
                        output_tensor='denoised_waveforms:0',
                        #scale_input=False,
                        checkpoint=None,
                        #write_outputs_separately=True,
                        ):
    """
    Run source separation. Mimics runnging the script with args.

    Args:
    input: numpy audio_array (mono)
        
    Returns:
    output_numpy: list of numpy arrays with all sources
    sampling_rate: sampling_rate
        
    """
    tf.disable_v2_behavior()

    meta_graph_filename = os.path.join(model_dir, 'inference.meta')
    tf.logging.info('Importing meta graph: %s', meta_graph_filename)

    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph(meta_graph_filename)
        meta_graph_def = g.as_graph_def()

    tf.reset_default_graph()
    input_wav = np.array(input, dtype=np.float32)

    # Enforce mono 
    if not input_wav.ndim == 1:
        input_wav = np.mean(input_wav, axis=0)

    # Resample to 22.05kz as expected from the source separation model
    expected_sampling_rate = 22050
    if not sampling_rate == expected_sampling_rate:
        print("Resample to 22.05kz as expected from the source separation model")
        input_wav = resample(input_wav, orig_sr=sampling_rate, target_sr=expected_sampling_rate)

    input_wav = tf.expand_dims(input_wav, axis=0)  
    input_wav = tf.expand_dims(input_wav, axis=0) # shape: [1, 1, samples]
    output_wav, = tf.import_graph_def(
        meta_graph_def,
        name='',
        input_map={input_tensor: input_wav}, # args.input_tensor: default='input_audio/receiver_audio:0'
        return_elements=[output_tensor]) # args.ouput_tenspr: default='denoised_waveforms:0'

    # Remove writing section, keep computation
    output_wav = tf.squeeze(output_wav, 0)  # [sources, samples]

    checkpoint = checkpoint or tf.train.latest_checkpoint(model_dir)

    with tf.Session() as sess:
        tf.logging.info('Restoring from checkpoint: %s', checkpoint)
        saver.restore(sess, checkpoint)
        output_numpy = sess.run(output_wav)

    return output_numpy, expected_sampling_rate


# def complete_separation(input,
#                         sampling_rate,
#                         output,
#                         model_dir,
#                         #input_channels=0,
#                         num_sources=2,
#                         output_channels=0,
#                         input_tensor='input_audio/receiver_audio:0', 
#                         output_tensor='denoised_waveforms:0',
#                         scale_input=False,
#                         checkpoint=None,
#                         write_outputs_separately=True,
#                         ):
#     """
#     Run source separation. Mimics runnging the script with args.

#     Args:
#     input: nummpy audio_array (mono)
        
#     Returns:
        
#     """
#     tf.disable_v2_behavior()

#     meta_graph_filename = os.path.join(model_dir, 'inference.meta')
#     tf.logging.info('Importing meta graph: %s', meta_graph_filename)

#     with tf.Graph().as_default() as g:
#         saver = tf.train.import_meta_graph(meta_graph_filename)
#         meta_graph_def = g.as_graph_def()

#     tf.reset_default_graph()
#     input_wav = np.array(input, dtype=np.float32)

#     # Enforce mono 
#     if not input_wav.ndim == 1:
#         input_wav = np.mean(input_wav, axis=0)

#     input_wav = tf.expand_dims(input_wav, axis=0)  
#     input_wav = tf.expand_dims(input_wav, axis=0) # shape: [1, 1, samples]
#     output_wav, = tf.import_graph_def(
#         meta_graph_def,
#         name='',
#         input_map={input_tensor: input_wav}, # args.input_tensor: default='input_audio/receiver_audio:0'
#         return_elements=[output_tensor]) # args.ouput_tenspr: default='denoised_waveforms:0'

#     output_wav = tf.squeeze(output_wav, 0)  # shape: [sources, samples]
#     output_wav = tf.transpose(output_wav)
#     if output_channels > 0:
#         output_wav = output_wav[:, :output_channels]
#     write_output_ops = inference.write_wav_file(
#         output, output_wav, sample_rate=sampling_rate,
#         num_channels=num_sources,
#         output_channels=output_channels,
#         write_outputs_separately=write_outputs_separately,
#         channel_name='source')

#     checkpoint = checkpoint
#     if not checkpoint:
#         checkpoint = tf.train.latest_checkpoint(model_dir)
#     with tf.Session() as sess:
#         tf.logging.info('Restoring from checkpoint: %s', checkpoint)
#         saver.restore(sess, checkpoint)
#         sess.run(write_output_ops)

# def separate_sources(audio_obj, model, input_channels=0, scale_input=False,
#                      num_sources=2, output_channels=0):
#     """
#     Run source separation on a HuggingFace Audio object.

#     Args:
#         audio_obj: HuggingFace Audio object (with `.array` and `.sampling_rate`).
#         model: dict returned by load_model().
#         input_channels: int, pad/truncate input channels if needed.
#         scale_input: bool, scale waveform to 0.99 peak.
#         num_sources: number of expected sources.
#         output_channels: truncate output sources if >0.

#     Returns:
#         separated_sources: np.ndarray [samples, sources]
#         sample_rate: int
#     """
#     sess = model["session"]
#     input_tensor = model["input_placeholder"]
#     output_tensor = model["output_tensor"]

#     waveform = audio_obj["array"] if isinstance(audio_obj, dict) else audio_obj.array
#     sample_rate = audio_obj["sampling_rate"] if isinstance(audio_obj, dict) else audio_obj.sampling_rate

#     # mimic inference.read_wav_file behavior
#     if waveform.ndim == 1:
#         waveform = np.expand_dims(waveform, axis=0)  # [1, samples]
#     waveform = np.transpose(waveform)  # [samples, channels]

#     if scale_input:
#         waveform = waveform / (np.max(np.abs(waveform)) + 1e-9) * 0.99

#     # TensorFlow expects [1, mics, samples]
#     feed_input = np.expand_dims(np.transpose(waveform), 0)

#     output = sess.run(output_tensor, feed_dict={input_tensor: feed_input})
#     if output_channels > 0:
#         output = output[:, :output_channels]

#     return output, sample_rate


def process_batch(audio_objs, model, **kwargs):
    """
    Process a batch of HuggingFace Audio objects.

    Args:
        audio_objs: list of Audio objects
        model: dict from load_model()
        **kwargs: forwarded to separate_sources()

    Returns:
        list of (output_array, sample_rate)
    """
    results = []
    for audio in audio_objs:
        out = separate_sources(audio, model, **kwargs)
        results.append(out)
    return results


# ------------------------------
# Optional CLI entrypoint
# ------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Separate a WAV file using a TF1 model.')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--input_wav', required=True)
    parser.add_argument('--output_wav', required=True)
    parser.add_argument('--num_sources', type=int, default=2)
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_dir)

    # Read wav manually using inference utils
    audio_np, sr = inference.read_wav_file(args.input_wav, input_channels=0, scale_input=False)

    audio_obj = {"array": np.array(audio_np).T, "sampling_rate": sr}
    separated, _ = separate_sources(audio_obj, model, num_sources=args.num_sources)

    inference.write_wav_file(args.output_wav, separated, sample_rate=sr,
                             num_channels=args.num_sources, write_outputs_separately=True,
                             channel_name="source")

    print(f"Saved separated sources to {args.output_wav}")


if __name__ == "__main__":
    main()
