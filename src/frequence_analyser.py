import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

from basics import loading_audio_file
from constants import *

def get_fft(audio):
    """Compute the FFT of the audio signal"""
    N = len(audio)
    spectrum = fft(audio)
    frequencies = fftfreq(N, 1/SAMPLE_RATE)
    pos_mask = frequencies >= 0
    positive_frequencies = frequencies[pos_mask]
    magnitude = (2/N) * np.abs(spectrum[pos_mask])
    return positive_frequencies, magnitude


def plot_fft(file_path):
    """Plot the fft of an andio"""
    audio = loading_audio_file(file_path)

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


def top_2_fundamental_frequencies(fft_frequencies, fft_magnitude, min_freq=60.0):
    """Get the top 2 frequencies in the spectrum"""
    peak_indices, _ = find_peaks(fft_magnitude, height=np.max(fft_magnitude) * 0.05)
    
    peak_freqs = fft_frequencies[peak_indices]
    peak_mags = fft_magnitude[peak_indices]
    
    mask = peak_freqs >= min_freq
    peak_freqs = peak_freqs[mask]
    peak_mags = peak_mags[mask]
    
    if len(peak_freqs) == 0:
        return []
    
    top_2_idx = np.argsort(peak_mags)[::-1][:2]
    return peak_freqs[top_2_idx].tolist()


if __name__ == "__main__":
    file_path = "audio_files/B3.mp3"
    plot_fft(file_path)