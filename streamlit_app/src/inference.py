"""TFLite model loading and inference helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:  # Fallback to the full TensorFlow package if available
    import tensorflow as tf

    tflite = tf.lite  # type: ignore

from .audio_processing import preprocess_audio

LABELS: Sequence[str] = ("Sedang", "Tenang", "Tinggi")


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    raw_output: np.ndarray
    log_mel: np.ndarray


class TFLiteAudioClassifier:
    """Lightweight wrapper around the TensorFlow Lite interpreter."""

    def __init__(self, model_path: Path | str, labels: Sequence[str] = LABELS) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found at {model_path}")

        self.labels = list(labels)
        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, audio_source) -> PredictionResult:
        input_tensor, log_mel = preprocess_audio(audio_source)
        input_info = self.input_details[0]
        output_info = self.output_details[0]

        model_input = input_tensor.astype(input_info["dtype"])
        self.interpreter.set_tensor(input_info["index"], model_input)
        self.interpreter.invoke()

        raw_output = self.interpreter.get_tensor(output_info["index"])[0]
        probabilities = self._softmax(raw_output)
        best_idx = int(np.argmax(probabilities))

        prob_dict = {label: float(probabilities[i]) for i, label in enumerate(self.labels)}
        return PredictionResult(
            label=self.labels[best_idx],
            confidence=float(probabilities[best_idx]),
            probabilities=prob_dict,
            raw_output=raw_output,
            log_mel=log_mel,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32)
        logits -= np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)
