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
