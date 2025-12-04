from __future__ import annotations

from pathlib import Path

import librosa.display  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from src.audio_processing import (
    AUDIO_SAMPLE_RATE,
    HOP_LENGTH,
    MAX_PAD_LEN,
    N_FFT,
    N_MELS,
)
from src.inference import TFLiteAudioClassifier

st.set_page_config(
    page_title="Klasifikasi Tingkat Kebisingan",
    page_icon="🎧",
    layout="wide",
)

st.title("🎧 Klasifikasi Tingkat Kebisingan Ruangan")
st.write(
    "Unggah atau rekam audio untuk memprediksi kategori kebisingan (Tenang, Sedang, Tinggi) "
    "menggunakan model CNN log-Mel spectrogram yang sudah dilatih."
)


@st.cache_resource(show_spinner=False)
def load_classifier() -> TFLiteAudioClassifier:
    model_path = Path(__file__).parent / "models" / "best_cnn_model_weighted_augmented.tflite"
    return TFLiteAudioClassifier(model_path)


def plot_log_mel(log_mel: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        log_mel,
        sr=AUDIO_SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Log-Mel Spectrogram")
    st.pyplot(fig)
    plt.close(fig)


def main():
    classifier = load_classifier()

    st.sidebar.header("Info Model")
    st.sidebar.markdown(
        "- **Sample rate:** 22.05 kHz\n"
        f"- **n_mels:** {N_MELS}\n"
        f"- **n_fft:** {N_FFT}\n"
        f"- **hop_length:** {HOP_LENGTH}\n"
        f"- **Target frame length:** {MAX_PAD_LEN}\n"
        "- **Keluaran:** Sedang, Tenang, Tinggi"
    )
    st.sidebar.info("Pastikan file audio berformat WAV mono agar hasilnya konsisten.")

    mode = st.radio(
        "Pilih sumber audio",
        ("Unggah Audio (.wav)", "Rekam Langsung"),
        horizontal=True,
    )

    audio_bytes: bytes | None = None
    file_name: str | None = None

    if mode == "Unggah Audio (.wav)":
        uploaded = st.file_uploader("Unggah file WAV (maks 60 detik).", type=["wav"])
        if uploaded is not None:
            audio_bytes = uploaded.getvalue()
            file_name = uploaded.name
            st.audio(audio_bytes, format="audio/wav")
    else:
        st.write("Tekan tombol mikrofon untuk mulai/stop rekaman. Gunakan headset untuk hasil terbaik.")
        recorded_audio = audio_recorder(sample_rate=AUDIO_SAMPLE_RATE)
        if recorded_audio is not None:
            audio_bytes = recorded_audio
            file_name = "rekaman_pengguna.wav"
            st.audio(audio_bytes, format="audio/wav")

    placeholder = st.empty()
    if st.button("Klasifikasikan", type="primary"):
        if audio_bytes is None:
            st.warning("Silakan unggah atau rekam audio terlebih dahulu.")
        else:
            with st.spinner("Memproses audio ..."):
                result = classifier.predict(audio_bytes)

            st.success("Prediksi selesai!")
            st.metric(
                label="Kategori Kebisingan",
                value=result.label,
                delta=f"Confidence {result.confidence * 100:.1f}%",
            )

            prob_df = pd.DataFrame(
                {"Label": list(result.probabilities.keys()), "Probabilitas": list(result.probabilities.values())}
            ).set_index("Label")
            st.bar_chart(prob_df)

            with st.expander("Detail Probabilitas"):
                st.json(result.probabilities)

            plot_log_mel(result.log_mel)

    with placeholder.container():
        st.caption("Model menggunakan log-Mel spectrogram untuk mendeteksi pola kebisingan lingkungan.")


if __name__ == "__main__":
    main()
