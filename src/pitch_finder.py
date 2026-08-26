import librosa
import numpy as np

from constantes import *

duration_in_seconds = 0.5
audio_files = [
    "audio_files/A2.mp3",
    "audio_files/B3.mp3",
    "audio_files/E2.mp3",
    "audio_files/E4.mp3",
    "audio_files/G3.mp3"
]

def find_fundamental_frequency(audio)->float:
    """Determine the fundamental frequency using librosa's pyin function"""
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz(LOWEST_NOTE), fmax=librosa.note_to_hz(HIGHEST_NOTE), sr=SAMPLE_RATE)
    mask = (voiced_flag) & (voiced_probs > VOICED_PROB_THRESHOLD)
    f0_valid = f0[mask]
    f0_median = np.median(f0_valid) if len(f0_valid) > 0 else None
    return f0_median

def main():
    print("Loading audio file...")
    y = []
    for file in audio_files:
        raw_audio, _ = librosa.load(file, sr=SAMPLE_RATE)
        audio, _ = librosa.effects.trim(raw_audio, top_db=AUDIO_TRIM_DB)
        duration_in_samples = min(int(duration_in_seconds * SAMPLE_RATE), len(audio))
        audio = audio[0:duration_in_samples]
        y.append(audio)
    print("Audio files loaded.")

    print("Searching for fundamental frequencies...")
    for audio in y:
        N = len(audio)
        f0_median = find_fundamental_frequency(audio)
        if f0_median:
            print(f"{f0_median:.2f} Hz : {librosa.hz_to_note(f0_median)}")
        else:
            print("No valid pitch detected")

    print("Done.")

if __name__ == "__main__":
    main()