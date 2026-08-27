import librosa
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import sounddevice as sd
import time

from constantes import *
from basics import loading_audio_file

# ______________Testing parameters______________
speed_factor = 0.25

# ______________Testing script______________
audio = loading_audio_file("audio_files/Howls moving castle (Merry-Go-Round of Life).mp3", duration_in_seconds = 5)
audio_stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
print("Audio stretched")

print("Searching for onsets...")
onset_samples = librosa.onset.onset_detect(
    y=audio,
    sr=SAMPLE_RATE,
    units='samples',
    backtrack=False,
    delta=DELTA,
    wait=WAIT
)
onset_samples = np.concatenate(([0], onset_samples, [len(audio)]))
onset_times = librosa.samples_to_time(onset_samples, sr=SAMPLE_RATE)

fig, ax = plt.subplots(figsize=(10, 5))
stretch_times = np.arange(len(audio_stretched)) * speed_factor / SAMPLE_RATE
ax.plot(stretch_times, audio_stretched, lw=1)
for onset in onset_times:
    ax.axvline(x=onset, color='r', linestyle='--', lw=1, alpha=0.7)
ax.set_title("Lecture en cours...")
ax.set_xlabel("Temps (s)")

cursor_line = ax.axvline(x=0, color='r', lw=1.5)

# Démarre la lecture
print("Playing audio...")
sd.play(audio_stretched, SAMPLE_RATE)
start_time = time.time()

def update(frame):
    elapsed = time.time() - start_time
    audio_elapsed = elapsed * speed_factor
    cursor_line.set_xdata([audio_elapsed, audio_elapsed])
    if audio_elapsed > stretch_times[-1]:
        ani.event_source.stop()
    return cursor_line,

ani = animation.FuncAnimation(fig, update, interval=30, blit=True)
plt.show()