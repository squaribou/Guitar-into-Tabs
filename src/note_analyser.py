import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

print("Loading audio file...")
y, sr = librosa.load("audio_files/E2.mp3")
print("Audio file loaded.")
y, _ = librosa.effects.trim(y, top_db=20)
# y = y[0:12000]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
pd.Series(y).plot(lw=1, title="Signal Audio", ax=axes[0])

print("Computing FFT...")
N = len(y)

spectrum = fft(y)
frequencies = fftfreq(N, 1/sr)
pos_mask = frequencies >= 0
positive_frequencies = frequencies[pos_mask]
magnitude = (2/N) * np.abs(spectrum[pos_mask])

axes[1].plot(positive_frequencies, magnitude)
axes[1].set_title("Magnitude Spectrum")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlim(0, 1000)

plt.tight_layout()
plt.show()
