import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load("Howls moving castle (Merry-Go-Round of Life).mp3")  # y = signal audio, sr = sample rate
# print(sr)

# Plot the audio signal
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

pd.Series(y).plot(ax=axes[0], lw=1, title="Signal Audio")
pd.Series(y[3000:5000]).plot(ax=axes[1], lw=1, title="Zoom Signal Audio")

plt.tight_layout()
plt.show()