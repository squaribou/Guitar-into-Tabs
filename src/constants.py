from basic_pitch.constants import ANNOTATION_HOP, ANNOTATIONS_BASE_FREQUENCY
import librosa

# Get constantes from basic_pitch
FRAME_TIME = ANNOTATION_HOP
MIDI_OFFSET = round(librosa.hz_to_midi(ANNOTATIONS_BASE_FREQUENCY))  # 21

# Range detection
LOWEST_NOTE = "A1"
HIGHEST_NOTE = "D6"

# Audio parameters
SAMPLE_RATE = 22050

# Note detection parameters
DISTANCE_FRAMES = 8
HEIGHT = 0.6
PROMINENCE = 0.15
RELATIVE_THERSHOLD = 0.6

# Note filter parameters
HARMONIC_INTERVALS = {
    19: 0.9,   # octave + fifth (3rd harmonic)
    28: 0.85,  # 2 octaves + major third (5th harmonic)
    31: 0.85,  # 2 octaves + a perfect fifth (6th harmonic)
}
RATIO_THRESHOLD = 0.8

# Tempo parameters
MIN_IOI = 0.05
TOLERANCE = 0.10
PERIODE_RESOLUTION = 0.005
BPM_RANGE = (40, 300)
MAX_TEMPO_BPM = 150

# Music sheet parameters
MUSESCORE_PATH = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
STANDARD_DURATIONS = [
    4.0, # whole note
    3.0, # dotted half note
    2.0, # half note
    1.5, # dotted quarter note
    1.0, # quarter note
    0.75, # dotted eighth note
    0.5, # eighth note
    0.25, # sixteenth note
    0.125, # thirty-second note
]
KEY_SIGNATURE_RANGE = range(0, 8) # only sharps
