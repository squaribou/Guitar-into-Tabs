import librosa
import numpy as np
import matplotlib.pyplot as plt

from constantes import *
from pitch_finder import find_possible_pitches
from basics import loading_audio_file
from sequence_fft import get_fft

# delta_list = [0.1, 0.01, 0.001, 0.0001]
wait_list = [10,7,5,3,1]

def testing_wait_list_parameters(wait_list):
    _, axes = plt.subplots(len(wait_list), 1, figsize=(10, 8))
    audio = loading_audio_file("audio_files/Howls moving castle (Merry-Go-Round of Life).mp3", duration_in_seconds = 5)

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

        print("Searching for possible pitches...")
        for i in range(len(onset_samples) - 1):
            pitches = find_possible_pitches(audio[onset_samples[i]:onset_samples[i + 1]])
            print(f"Note number i : {pitches}")

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


def pitch_analyser():
    audio = loading_audio_file("audio_files/Howls moving castle (Merry-Go-Round of Life).mp3", duration_in_seconds = 5)

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

    print("Searching for possible pitches...")
    for i in range(len(onset_samples) - 1):
        pitches = find_possible_pitches(audio[onset_samples[i]:onset_samples[i + 1]])
        print(f"Note number {i} : {pitches}")

    # Plot the audio signal
    _, axes = plt.subplots(4, 3, figsize=(10, 8))
    times = np.arange(len(audio)) / SAMPLE_RATE
    axes[0, 0].plot(times, audio)
    for i, onset in enumerate(onset_times):
        axes[0, 0].axvline(x=onset, color='r', linestyle='--', lw=1, alpha=0.7)
        axes[0, 0].text(onset, audio.max()*0.9, i, rotation=90, fontsize=8, color='r')
    axes[0, 0].set_title(f"Zoom Signal Audio")
    axes[0, 0].set_xlabel("Temps (s)")

    n = 2
    for j in range(4):
        for k in range(3):
            if j == 0 and k == 0:
                continue
            frequencies, magnitude = get_fft(audio[onset_samples[n]:onset_samples[n+1]])
            pitches = find_possible_pitches(audio[onset_samples[n]:onset_samples[n + 1]])
            # Plot the FFT
            axes[j,k].plot(frequencies, magnitude)
            axes[j,k].set_title(f"Magnitude Spectrum ({n}e) : {pitches}")
            axes[j,k].set_xlabel("Frequency (Hz)")
            axes[j,k].set_ylabel("Amplitude")
            axes[j,k].set_xlim(0, 1000)
            n += 1

    print("Window displayed.")
    plt.tight_layout()
    plt.show()


def onset_anlayser():
    audio = loading_audio_file("audio_files/Howls moving castle (Merry-Go-Round of Life).mp3", duration_in_seconds = 4)
    
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

    # Plot the audio signal
    _, axes = plt.subplots(2, 1, figsize=(10, 8))
    times = np.arange(len(audio)) / SAMPLE_RATE
    axes[0].plot(times, audio)
    for i, onset in enumerate(onset_times):
        axes[0].axvline(x=onset, color='r', linestyle='--', lw=1, alpha=0.7)
        axes[0].text(onset, audio.max()*0.9, i, rotation=90, fontsize=8, color='r')
    axes[0].set_title(f"Zoom Signal Audio")
    axes[0].set_xlabel("Temps (s)")

    rms = librosa.feature.rms(y=audio)[0]
    times_rms = librosa.times_like(rms, sr=SAMPLE_RATE)

    axes[1].plot(times_rms, rms)
    axes[1].set_title("Intensité (RMS) du signal")
    axes[1].set_xlabel("Temps (s)")
    axes[1].set_ylabel("RMS")

    onset_env_complex = librosa.onset.onset_strength(y=audio, feature=librosa.stft, lag=2, max_size=3)

    onset_samples = librosa.onset.onset_detect(onset_envelope=onset_env_complex, sr=SAMPLE_RATE,units='samples',
            backtrack=False,
            delta=DELTA,
            wait=WAIT)
    onset_times = librosa.samples_to_time(onset_samples, sr=SAMPLE_RATE)
    for i, onset in enumerate(onset_times):
        axes[0].axvline(x=onset, color='b', linestyle='--', lw=1, alpha=0.7)

    print("Window displayed.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # testing_wait_list_parameters(wait_list)
    # pitch_analyser()
    onset_anlayser()