import librosa

import soundfile as sf
import tempfile
import os
from basic_pitch.inference import predict
from scipy.signal import find_peaks

from constants import *

def loading_audio_file(file_path, starting_time_in_seconds=0, duration_in_seconds=None):
    """Load an audio file, trim silence, and return a segment of the specified duration starting from the specified time."""
    print("Loading audio file...")
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    if duration_in_seconds is None:
        duration_in_seconds = librosa.get_duration(y=audio, sr=SAMPLE_RATE)
    duration_in_samples = min(librosa.time_to_samples(duration_in_seconds, sr=SAMPLE_RATE), len(audio))
    starting_time_in_samples = librosa.time_to_samples(starting_time_in_seconds, sr=SAMPLE_RATE)
    audio = audio[starting_time_in_samples:starting_time_in_samples + duration_in_samples]
    print("Audio file loaded.")
    return audio


def predict_from_array(audio):
    """Wrapper to use the fonction predict with an array instead of mp3 file"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio, SAMPLE_RATE)
        return predict(tmp_path,
                        frame_threshold=0.5,
                        melodia_trick=True)
    finally:
        os.remove(tmp_path)


def detect_attack_times(onset_matrix, distance_frames=8, height=0.6, prominence=0.15):
    """
    Detecte attack times based on the strong time columns

    onset_matrix : array (n_frames, n_pitches)
    distances_frames : minimal space between two attacks (in frame)
    height : minimal confidence for a columns to be considered
    prominence : how much the spike needs to stand out in his temporal neighborhood
    """
    onset_strength = onset_matrix.max(axis=1)
    
    peak_frames, properties = find_peaks(
        onset_strength,
        distance=distance_frames,
        height=height,
        prominence=prominence
    )
    
    return peak_frames, properties["peak_heights"]


def get_pitches_at_attack(onset_matrix, frame_idx, relative_threshold=0.6):
    """
    Get the active pitches in midi on a attack with their confidence

    onset_matrix : array (n_frames, n_pitches)
    relative_threshold : fraction of the column's peak maximum to be considered a genuine note (not just noise)
    """
    column = onset_matrix[frame_idx]
    max_val = column.max()
    
    active_pitches_idx, _ = find_peaks(column, height=max_val * relative_threshold)
    
    return [(idx + MIDI_OFFSET, column[idx]) for idx in active_pitches_idx]


def extract_note_attacks(onset_matrix, distance_frames=8, height=0.6, prominence=0.15,relative_threshold=0.6):
    """
    Get the attack time and active pitches on the attacks with their confidence

    onset_matrix : array (n_frames, n_pitches)
    onset_matrix : array (n_frames, n_pitches)
    distances_frames : minimal space between two attacks (in frame)
    height : minimal confidence for a columns to be considered
    prominence : how much the spike needs to stand out in his temporal neighborhood
    relative_threshold : fraction of the column's peak maximum to be considered a genuine note (not just noise)
    """
    peak_frames, _ = detect_attack_times(onset_matrix, distance_frames, height, prominence)
    
    results = []
    for frame_idx in peak_frames:
        t = frame_idx * FRAME_TIME
        pitches = get_pitches_at_attack(onset_matrix, frame_idx, relative_threshold)
        results.append((t, pitches))
    
    return results