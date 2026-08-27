import librosa
import numpy as np

from basics import loading_audio_file
from constantes import *

def find_fundamental_frequency(audio)->float: # Expired
    """Determine the fundamental frequency using librosa's pyin function"""
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz(LOWEST_NOTE), fmax=librosa.note_to_hz(HIGHEST_NOTE), sr=SAMPLE_RATE)
    mask = (voiced_flag) & (voiced_probs > VOICED_PROB_THRESHOLD)
    f0_valid = f0[mask]
    f0_median = np.median(f0_valid) if len(f0_valid) > 0 else None
    return f0_median


def find_possible_pitches(audio)->dict:
    """Determine the possible pitches of the audio signal using librosa's pyin function"""
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                 fmin=librosa.note_to_hz(LOWEST_NOTE), 
                                                 fmax=librosa.note_to_hz(HIGHEST_NOTE), 
                                                 sr=SAMPLE_RATE
                                                 )
    mask = (voiced_flag) & (voiced_probs > VOICED_PROB_THRESHOLD)
    f0_valid = f0[mask]
    pitches = {}
    for f in f0_valid:
        if f is not None:
            note = librosa.hz_to_note(f)
            if note not in pitches:
                pitches[note] = 1
            else:
                pitches[note] += 1
    return pitches


def testing(audio_files):
    y = []
    for file_path in audio_files:
        audio = loading_audio_file(file_path, duration_in_seconds=0.5)
        y.append(audio)

    print("Searching for possible pitches...")
    for audio in y:
        pitches = find_possible_pitches(audio)
        if pitches:
            print(f"Possible pitches: {pitches}")
        else:
            print("No valid pitch detected")

    print("Done.")


if __name__ == "__main__":
    audio_files = [
        "audio_files/A2.mp3",
        "audio_files/B3.mp3",
        "audio_files/E2.mp3",
        "audio_files/E4.mp3",
        "audio_files/G3.mp3"
    ]
    testing(audio_files=audio_files)