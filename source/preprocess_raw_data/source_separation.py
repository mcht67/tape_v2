import numpy as np
import tensorflow.compat.v1 as tf
import librosa
from externals.bird_mixit.tools import inference

tf.disable_v2_behavior()


class BirdMixitSeparator:
    """
    In-memory Bird-MixIT wrapper for NumPy arrays.
    Handles resampling, normalization, and TensorFlow inference.
    """

    def __init__(self, model_dir, checkpoint=None, num_sources=4, model_sr=22050):
        """
        Args:
            model_dir (str): Directory with inference.meta and checkpoints
            checkpoint (str): Optional path to .ckpt
            num_sources (int): Number of output sources
            model_sr (int): Sampling rate expected by the Bird-MixIT model
        """
        self.model_dir = model_dir
        self.checkpoint = checkpoint
        self.num_sources = num_sources
        self.model_sr = model_sr
        self.sess, self.graph = self._load_model()

    def _load_model(self):
        tf.reset_default_graph()
        meta_path = f"{self.model_dir}/inference.meta"
        saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
        sess = tf.Session()
        ckpt = self.checkpoint or tf.train.latest_checkpoint(self.model_dir)
        saver.restore(sess, ckpt)
        return sess, tf.get_default_graph()

    def separate_array(self, audio_array, sample_rate):
        """
        Separate sources from an in-memory NumPy audio array.

        Args:
            audio_array: np.ndarray (shape [samples] or [channels, samples])
            sample_rate: int, input sampling rate
        Returns:
            separated_sources: np.ndarray [num_sources, samples] at model_sr
        """
        # Ensure mono or channel-first format
        if audio_array.ndim == 1:
            audio_array = np.expand_dims(audio_array, 0)  # mono
        elif audio_array.shape[0] < audio_array.shape[1]:
            pass  # channels x samples
        else:
            audio_array = audio_array.T

        # Resample to model sampling rate
        if sample_rate != self.model_sr:
            audio_array = np.stack(
                [librosa.resample(ch, orig_sr=sample_rate, target_sr=self.model_sr) for ch in audio_array],
                axis=0,
            )

        # Convert float waveform to PCM16-like float range [-1, 1]
        clipped = np.clip(audio_array, -1.0, 1.0)
        int16_like = (clipped * 32767).astype(np.int16)
        float_norm = (int16_like.astype(np.float32)) / 32768.0
        batch = np.expand_dims(float_norm, axis=0)  # [1, channels, samples]

        # Run TF graph
        input_tensor = self.graph.get_tensor_by_name("input_audio/receiver_audio:0")
        output_tensor = self.graph.get_tensor_by_name("denoised_waveforms:0")

        separated = self.sess.run(output_tensor, feed_dict={input_tensor: batch})
        separated = np.squeeze(separated, 0)  # [sources, samples]

        return separated

    def close(self):
        self.sess.close()
