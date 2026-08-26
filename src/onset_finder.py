import librosa
import numpy as np
import matplotlib.pyplot as plt

from constantes import *
from pitch_finder import find_fundamental_frequency

# ______________Testing parameters______________
starting_time_in_seconds = 0
duration_in_seconds = 5

# delta_list = [0.1, 0.01, 0.001, 0.0001]
wait_list = [10,7,5,3,1]
_, axes = plt.subplots(len(wait_list), 1, figsize=(10, 8))

# ______________Testing script______________
print("Loading audio file...")
raw_audio, _ = librosa.load("audio_files/Howls moving castle (Merry-Go-Round of Life).mp3", sr=SAMPLE_RATE)
audio, _ = librosa.effects.trim(raw_audio, top_db=AUDIO_TRIM_DB)
duration_in_samples = min(librosa.time_to_samples(duration_in_seconds, sr=SAMPLE_RATE), len(audio))
starting_time_in_samples = librosa.time_to_samples(starting_time_in_seconds, sr=SAMPLE_RATE)
audio = audio[starting_time_in_samples:starting_time_in_samples + duration_in_samples]
print("Audio file loaded.")

for k, wait in enumerate(wait_list):
    print("Processing audio with wait =", wait)
    print("Searching for onsets...")
    onset_samples = librosa.onset.onset_detect(
        y=audio,
        sr=SAMPLE_RATE,
        units='samples',
        backtrack=False,
        delta=DELTA,
        wait=wait
    )
    onset_samples = np.concatenate(([0], onset_samples, [len(audio)]))
    onset_times = librosa.samples_to_time(onset_samples, sr=SAMPLE_RATE)

    print("Searching for fundamental frequencies...")
    for i in range(len(onset_samples) - 1):
        f0_median = find_fundamental_frequency(audio[onset_samples[i]:onset_samples[i + 1]])
        if f0_median:
            print(f"{f0_median:.2f} Hz : {librosa.hz_to_note(f0_median)}")
        else:
            print("No valid pitch detected")

    # Plot the audio signal
    times = np.arange(len(audio)) / SAMPLE_RATE
    axes[k].plot(times, audio)
    for onset in onset_times:
        axes[k].axvline(x=onset, color='r', linestyle='--', lw=1, alpha=0.7)
    axes[k].set_title(f"Zoom Signal Audio (wait = {wait})")
    axes[k].set_xlabel("Temps (s)")

print("Window displayed.")
plt.tight_layout()
plt.show()
