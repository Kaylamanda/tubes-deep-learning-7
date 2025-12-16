"""Audio loading and preprocessing utilities for the Streamlit inference app."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Union

import librosa
import numpy as np

# Preprocessing parameters (must stay aligned with the training notebook)
AUDIO_SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
MAX_PAD_LEN = 638  # Determined from the training set (pad/truncate target)

AudioSource = Union[str, Path, bytes, io.BytesIO]


def _read_waveform(source: AudioSource, sample_rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Load mono audio at the configured sample rate from various sources."""
    if isinstance(source, (str, Path)):
        waveform, _ = librosa.load(str(source), sr=sample_rate, mono=True, res_type="scipy")
        return _normalize_waveform(waveform)

    if isinstance(source, bytes):
        buffer = io.BytesIO(source)
    elif isinstance(source, io.BytesIO):
        buffer = source
        buffer.seek(0)
    else:
        raise TypeError("Unsupported audio source type. Provide a path, bytes, or BytesIO object.")

    waveform, _ = librosa.load(buffer, sr=sample_rate, mono=True, res_type="scipy")
    return _normalize_waveform(waveform)


def _normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Scale waveform to the range [-1, 1] to avoid clipping issues."""
    if waveform.size == 0:
        raise ValueError("Audio file is empty.")
    max_val = np.max(np.abs(waveform))
    if max_val == 0:
        return waveform.astype(np.float32)
    return (waveform / max_val).astype(np.float32)


def _pad_or_truncate(log_mel: np.ndarray, target_len: int = MAX_PAD_LEN) -> np.ndarray:
    """Pad with zeros or truncate along the time axis to match the target length."""
    current_len = log_mel.shape[1]
    if current_len < target_len:
        pad_width = target_len - current_len
        return np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
    if current_len > target_len:
        return log_mel[:, :target_len]
    return log_mel


def preprocess_audio(source: AudioSource) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw audio into a TFLite-ready tensor and return the log-Mel spectrogram."""
    waveform = _read_waveform(source)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=AUDIO_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    padded_log_mel = _pad_or_truncate(log_mel)

    # Shape expected by the CNN: (batch, n_mels, max_pad_len, channels)
    network_input = padded_log_mel[np.newaxis, :, :, np.newaxis].astype(np.float32)
    return network_input, log_mel
