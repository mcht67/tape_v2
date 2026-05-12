from scipy.signal import resample
#import pyfar as pf
import numpy as np

def resample_audio(audio, orig_sampling_rate, target_sampling_rate):
    if orig_sampling_rate == target_sampling_rate:
        return audio
    duration = len(audio) / orig_sampling_rate
    num_samples = int(duration * target_sampling_rate)
    return resample(audio, num_samples).astype('float32')

# def get_mono_signal_by_channel_avg(signal):
#     return pf.Signal(np.mean(signal.time, axis=0), signal.sampling_rate)

def num_samples_to_duration_s(num_samples, sampling_rate):
    return num_samples / sampling_rate

def duration_s_to_num_samples(duration_s, sampling_rate):
    return duration_s * sampling_rate

def normalize_audio_array(audio_array):
    return audio_array / (np.max(np.abs(audio_array)) + 1e-9)