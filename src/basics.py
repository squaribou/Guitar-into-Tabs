import librosa

from constantes import *

def loading_audio_file(file_path, duration_in_seconds=None, starting_time_in_seconds=0):
    """Load an audio file, trim silence, and return a segment of the specified duration starting from the specified time."""
    print("Loading audio file...")
    raw_audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(raw_audio, top_db=AUDIO_TRIM_DB)
    if duration_in_seconds is None:
        duration_in_seconds = librosa.get_duration(y=audio, sr=SAMPLE_RATE)
    duration_in_samples = min(librosa.time_to_samples(duration_in_seconds, sr=SAMPLE_RATE), len(audio))
    starting_time_in_samples = librosa.time_to_samples(starting_time_in_seconds, sr=SAMPLE_RATE)
    audio = audio[starting_time_in_samples:starting_time_in_samples + duration_in_samples]
    print("Audio file loaded.")
    return audio