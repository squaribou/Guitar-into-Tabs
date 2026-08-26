import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

from constantes import *

def get_fft(audio):
    """Compute the FFT of the audio signal"""
    N = len(audio)
    spectrum = fft(audio)
    frequencies = fftfreq(N, 1/SAMPLE_RATE)
    pos_mask = frequencies >= 0
    positive_frequencies = frequencies[pos_mask]
    magnitude = (2/N) * np.abs(spectrum[pos_mask])
    return positive_frequencies, magnitude

def main():
    print("Loading audio file...")
    raw_audio, _ = librosa.load("audio_files/E2.mp3")
    audio, _ = librosa.effects.trim(raw_audio, top_db=20)
    print("Audio file loaded.")

    _, axes = plt.subplots(2, 1, figsize=(10, 8))
    times = np.arange(len(audio)) / SAMPLE_RATE
    axes[0].plot(times, audio)
    axes[0].set_title("Signal Audio")
    axes[0].set_xlabel("Temps (s)")

    print("Computing FFT...")
    frequencies, magnitude = get_fft(audio)

    axes[1].plot(frequencies, magnitude)
    axes[1].set_title("Magnitude Spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(0, 1000)

    print("Window displayed.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()