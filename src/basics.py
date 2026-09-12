import matplotlib.pyplot as plt
import math

import soundfile as sf
import tempfile
import os
from basic_pitch.inference import predict
from scipy.signal import find_peaks

from constants import *

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


def analysis(model_output, note_attacks, note_with_duration, tempo_bpm, start_time=0, end_time=None):
    """
    Heatmap to visually diagnose where and why the model generates false positives, showing estimated, 
    positioned, and/or corrected note locations.
    """

    print("Prepare plotting...")
    start_frame = int(start_time / FRAME_TIME)
    end_frame = int(end_time / FRAME_TIME)

    attack_frames = []
    attack_pitches = []
    for t, pitches in note_attacks:
        frame_idx = t / FRAME_TIME
        if not (start_frame <= frame_idx <= end_frame):
            continue
        for midi, _confidence in pitches:
            attack_frames.append(frame_idx)
            attack_pitches.append(midi - MIDI_OFFSET)

    zero_offset_frame = note_attacks[0][0] / FRAME_TIME
    quarter_duration = 60 / tempo_bpm
    duration_frames = []
    duration_pitches = []
    for pitches, quarter_position_in_measure, measure, min_d in note_with_duration:
        t = (measure * 4 + quarter_position_in_measure) * quarter_duration
        frame_idx = t / FRAME_TIME + zero_offset_frame
        if not (start_frame <= frame_idx <= end_frame):
            continue
        for midi, _confidence in pitches:
            duration_frames.append(frame_idx)
            duration_pitches.append(midi - MIDI_OFFSET)

    _, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(model_output["onset"].T, aspect="auto", origin="lower")
    axes[0].set_title("Probabilités d'onset")
    axes[0].set_ylim(top=60, bottom=20)
    axes[0].set_xlim(left=start_frame, right=end_frame)

    axes[1].imshow(model_output["note"].T, aspect="auto", origin="lower")
    axes[1].set_title("Probabilités de note")
    axes[1].set_ylim(top=60, bottom=20)
    axes[1].set_xlim(left=start_frame, right=end_frame)

    axes[0].scatter(attack_frames, attack_pitches, color="red", s=5, marker="o", zorder=3)
    axes[1].scatter(attack_frames, attack_pitches, color="red", s=5, marker="o", zorder=3)
    axes[0].scatter(duration_frames, duration_pitches, color="yellow", s=5, marker="o", zorder=3)
    axes[1].scatter(duration_frames, duration_pitches, color="yellow", s=5, marker="o", zorder=3)

    sixteenth_duration = quarter_duration / 4
    sixteenth_duration_frames = sixteenth_duration / FRAME_TIME
    n_steps_to_start = math.floor((start_frame - zero_offset_frame) / sixteenth_duration_frames)
    sixteenth_frame = zero_offset_frame + n_steps_to_start * sixteenth_duration_frames
    while sixteenth_frame <= end_frame:
        axes[0].axvline(sixteenth_frame, color="gray", linewidth=0.5, alpha=0.5, zorder=1)
        axes[1].axvline(sixteenth_frame, color="gray", linewidth=0.5, alpha=0.5, zorder=1)
        sixteenth_frame += sixteenth_duration_frames

    print("Window displayed.")
    plt.show()


def extract_note_attacks(onset_matrix, distance_frames=DISTANCE_FRAMES,height=HEIGHT,
                        prominence=PROMINENCE, relative_threshold=RELATIVE_THERSHOLD):
    """
    Get the attack time and active pitches on the attacks with their confidence

    onset_matrix : array (n_frames, n_pitches)
    onset_matrix : array (n_frames, n_pitches)
    distances_frames : minimal space between two attacks (in frame)
    height : minimal confidence for a columns to be considered
    prominence : how much the spike needs to stand out in his temporal neighborhood
    relative_threshold : fraction of the column's peak maximum to be considered a genuine note (not just noise)
    """

    onset_strength = onset_matrix.max(axis=1)
    
    peak_frames, _ = find_peaks(
        onset_strength,
        distance=distance_frames,
        height=height,
        prominence=prominence
    )
    
    results = []
    for frame_idx in peak_frames:
        t = frame_idx * FRAME_TIME
        column = onset_matrix[frame_idx]
        max_val = column.max()
        active_pitches_idx, _ = find_peaks(column, height=max_val * relative_threshold)
        pitches = [(idx + MIDI_OFFSET, column[idx]) for idx in active_pitches_idx]
        results.append((t, pitches))
    
    return results


def group_by_measure(results):
    """
    results : liste de (pitches, quarter_length, quarter_position_in_measure, measure_position)
    
    Retourne une liste où chaque élément est une mesure : une liste de tuples
    (pitches, quarter_length, quarter_position_in_measure), triée par numéro de mesure croissant.
    """
    measures_dict = {}
    
    for pitches, quarter_position_in_measure, measure_position, quarter_length in results:
        note_info = (pitches, quarter_position_in_measure, quarter_length)
        measures_dict.setdefault(measure_position, []).append(note_info)
    
    max_measure = max(measures_dict.keys())
    measures_list = [measures_dict.get(m, []) for m in range(max_measure + 1)]
    
    return measures_list
