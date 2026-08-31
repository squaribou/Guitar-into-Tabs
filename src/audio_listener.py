import librosa
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import sounddevice as sd

from constants import *
from basics import loading_audio_file, predict_from_array, extract_note_attacks


def listen_audio(file_path, starting_time_in_seconds=0, duration_in_seconds=None, speed_factor=1):
    """
    Launch the audio and display a window with the signal,
    we can see a cursor indicate us at which time we are and the note attacks
    """

    audio = loading_audio_file(file_path, starting_time_in_seconds, duration_in_seconds)
    audio_stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
    model_output, _, _ = predict_from_array(audio)
    attacks = extract_note_attacks(model_output["onset"])

    fig, ax = plt.subplots(figsize=(10, 5))
    stretch_times = np.arange(len(audio_stretched)) * speed_factor / SAMPLE_RATE
    ax.plot(stretch_times, audio_stretched, lw=1)
    ax.set_title("Lecture en cours...")
    ax.set_xlabel("Temps (s)")

    for t, pitches in attacks:
        ax.axvline(x=t, ymin=0, ymax=0.5, color='b', linestyle='--', lw=1, alpha=0.7)

    cursor_line = ax.axvline(x=0, color='g', lw=1.5)

    frames_played = 0

    def audio_callback(outdata, frames, time_info, status):
        nonlocal frames_played
        chunk = audio_stretched[frames_played:frames_played + frames]
        if len(chunk) < frames:
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = 0
            raise sd.CallbackStop
        outdata[:, 0] = chunk
        frames_played += frames

    print("Playing audio...")
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
    stream.start()

    def update(frame):
        audio_elapsed = (frames_played / SAMPLE_RATE) * speed_factor
        cursor_line.set_xdata([audio_elapsed, audio_elapsed])
        if audio_elapsed > stretch_times[-1]:
            ani.event_source.stop()
        return cursor_line,

    ani = animation.FuncAnimation(fig, update, interval=30, blit=True)
    plt.show()