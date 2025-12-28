# batch_process_wav.py
# Batch processing script for bird_mixit with proper TensorFlow initialization

import os
import argparse
import tensorflow.compat.v1 as tf
import inference

tf.disable_v2_behavior()
strtobool = inference.strtobool


def load_model(model_dir, checkpoint=None):
    tf.reset_default_graph()

    meta_graph_filename = os.path.join(model_dir, 'inference.meta')
    print(f"[INFO] Importing meta graph: {meta_graph_filename}")
    saver = tf.train.import_meta_graph(meta_graph_filename, clear_devices=True)

    sess = tf.Session()
    if not checkpoint:
        checkpoint = tf.train.latest_checkpoint(model_dir)
    print(f"[INFO] Restoring from checkpoint: {checkpoint}")
    saver.restore(sess, checkpoint)

    graph = tf.get_default_graph()
    return sess, graph


def separate_file(sess, graph, wav_path, output_path,
                  input_channels=0, scale_input=False, num_sources=2, output_channels=0, write_outputs_separately=True):
    print(f"[INFO] Processing file: {wav_path}")

    input_tensor = graph.get_tensor_by_name('input_audio/receiver_audio:0')
    output_tensor = graph.get_tensor_by_name('denoised_waveforms:0')

    input_wav, sample_rate = inference.read_wav_file(wav_path, input_channels, scale_input)

    with tf.Graph().as_default():
        input_wav = tf.transpose(input_wav)
        input_wav = tf.expand_dims(input_wav, axis=0)
        input_wav_eval = sess.run(input_wav)

    output_wav_eval = sess.run(output_tensor, feed_dict={input_tensor: input_wav_eval})

    output_wav = tf.squeeze(output_wav_eval, 0)
    output_wav = tf.transpose(output_wav)

    if output_channels > 0:
        output_wav = output_wav[:, :output_channels]

    write_ops = inference.write_wav_file(
        output_path, output_wav, sample_rate=sample_rate,
        num_channels=num_sources,
        output_channels=output_channels,
        write_outputs_separately=write_outputs_separately,
        channel_name='source'
    )

    sess.run(write_ops)
    print(f"[INFO] Finished writing to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch source separation with bird_mixit")
    parser.add_argument("--model_dir", required=True, type=str, help="Model directory containing inference.meta")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint")
    parser.add_argument("--input_dir", required=True, type=str, help="Directory containing input WAV files")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save separated outputs")
    parser.add_argument("--num_sources", type=int, default=2, help="Number of output sources")
    parser.add_argument("--input_channels", type=int, default=0, help="Truncate/pad input channels")
    parser.add_argument("--output_channels", type=int, default=0, help="Truncate output channels")
    parser.add_argument("--scale_input", type=strtobool, default=False, help="Scale input audio")
    parser.add_argument("--write_outputs_separately", type=strtobool, default=True, help="Write each source separately")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sess, graph = load_model(args.model_dir, args.checkpoint)

    wav_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".wav")]
    wav_files.sort()

    for wav_file in wav_files:
        input_path = os.path.join(args.input_dir, wav_file)
        output_path = os.path.join(args.output_dir, wav_file.replace(".wav", "_separated.wav"))
        separate_file(sess, graph, input_path, output_path,
                      input_channels=args.input_channels, scale_input=args.scale_input,
                      num_sources=args.num_sources, output_channels=args.output_channels,
                      write_outputs_separately=args.write_outputs_separately)

    sess.close()
    print("[INFO] Batch processing complete!")


if __name__ == "__main__":
    main()
