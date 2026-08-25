import librosa
import numpy as np
from scipy.fft import fft, fftfreq

from constantes import EQUAL_TEMPERAMENT

audio_files = [
    "audio_files/A2.mp3",
    "audio_files/B3.mp3",
    "audio_files/E2.mp3",
    "audio_files/E4.mp3",
    "audio_files/G3.mp3"
]

print("Loading audio file...")
y = []
for file in audio_files:
    audio, sr = librosa.load(file)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = audio[0:18000]
    y.append(audio)
print("Audio files loaded.")

print("Computing FFT...")
for audio in y:
    N = len(audio)

    spectrum = fft(audio)
    frequencies = fftfreq(N, 1/sr)
    pos_mask = frequencies >= 0
    positive_frequencies = frequencies[pos_mask]
    magnitude = (2/N) * np.abs(spectrum[pos_mask])

    fondammental_frequency = positive_frequencies[np.argmax(magnitude)]
    note = min(EQUAL_TEMPERAMENT, key=lambda x: abs(EQUAL_TEMPERAMENT[x] - fondammental_frequency))
    print(f"Fondamental frequency: {fondammental_frequency:.2f} Hz -- Detected note: {note} : {EQUAL_TEMPERAMENT[note]} Hz")