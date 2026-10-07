import matplotlib.pyplot as plt
import math
import pandas as pd

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


def analysis(model_output, note_df, tempo_bpm, start_time=0, end_time=None, beats_per_measure=BEATS_PER_MESURE):
    """
    Heatmap to visually diagnose where and why the model generates false positives, showing estimated, 
    positioned, and/or corrected note locations.

    note_df : DataFrame avec colonnes onset_frame, onset_time, pitch, confidence, duration,
        quarter_position_in_measure, measure_position, error
    """

    print("Prepare plotting...")
    start_frame = int(start_time / FRAME_TIME)
    end_frame = int(end_time / FRAME_TIME)

    # --- points rouges : onsets bruts ---
    attack_frame_idx = note_df["onset_time"] / FRAME_TIME
    mask = (attack_frame_idx >= start_frame) & (attack_frame_idx <= end_frame)
    attack_frames = attack_frame_idx[mask].tolist()
    attack_pitches = (note_df.loc[mask, "pitch"] - MIDI_OFFSET).tolist()

    # --- points jaunes : notes positionnées/corrigées ---
    zero_offset_frame = note_df["onset_time"].min() / FRAME_TIME
    quarter_duration = 60 / tempo_bpm

    duration_t = (
        note_df["measure_position"] * beats_per_measure
        + note_df["quarter_position_in_measure"]
    ) * quarter_duration
    duration_frame_idx = duration_t / FRAME_TIME + zero_offset_frame
    mask_d = (duration_frame_idx >= start_frame) & (duration_frame_idx <= end_frame)
    duration_frames = duration_frame_idx[mask_d].tolist()
    duration_pitches = (note_df.loc[mask_d, "pitch"] - MIDI_OFFSET).tolist()

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


def extract_note_attacks(onset_matrix, distance_frames=DISTANCE_FRAMES, height=HEIGHT,
                          prominence=PROMINENCE, relative_threshold=RELATIVE_THERSHOLD):
    """
    Get the attack time and active pitches on the attacks with their confidence.

    onset_matrix : array (n_frames, n_pitches)
    distance_frames : minimal space between two attacks (in frame)
    height : minimal confidence for a column to be considered
    prominence : how much the spike needs to stand out in its temporal neighborhood
    relative_threshold : fraction of the column's peak maximum to be considered a genuine note (not just noise)

    Retourne un DataFrame avec une ligne par note (onset_frame, onset_time, pitch, confidence).
    """
    onset_strength = onset_matrix.max(axis=1)

    peak_frames, _ = find_peaks(
        onset_strength,
        distance=distance_frames,
        height=height,
        prominence=prominence
    )

    rows = []
    for frame_idx in peak_frames:
        t = frame_idx * FRAME_TIME
        column = onset_matrix[frame_idx]
        max_val = column.max()
        active_pitches_idx, _ = find_peaks(column, height=max_val * relative_threshold)

        for idx in active_pitches_idx:
            rows.append({
                "onset_frame": frame_idx,
                "onset_time": t,
                "pitch": idx + MIDI_OFFSET,
                "confidence": column[idx],
            })

    return pd.DataFrame(rows, columns=["onset_frame", "onset_time", "pitch", "confidence"])


def assign_voice(positionned_notes, lowest_melody_note=LOWEST_MELODY_NOTE):
    """
    positionned_notes : DataFrame avec au moins onset_frame, pitch (+ colonnes déjà ajoutées :
        confidence, quarter_position_in_measure, measure_position, error)

    Pour chaque onset : la première note (dans l'ordre existant des lignes, donc
    normalement par confidence décroissante suite aux filtres précédents) dont
    pitch < lowest_melody_note devient "low", toutes les autres notes de cet
    onset deviennent "melody" (même si elles sont aussi sous le seuil).

    Retourne positionned_notes avec une colonne "voice" ("melody" ou "low") ajoutée.
    """
    voice = pd.Series(index=positionned_notes.index, dtype=object)

    for onset_id, group in positionned_notes.groupby("onset_frame", sort=False):
        bass_idx = next(
            (idx for idx, pitch in zip(group.index, group["pitch"]) if pitch < lowest_melody_note),
            None
        )
        for idx in group.index:
            voice[idx] = "low" if idx == bass_idx else "melody"

    result = positionned_notes.copy()
    result["voice"] = voice
    return result


def build_voice_in_measure(note_df, voice_name):
    voice_df = note_df[note_df["voice"] == voice_name]
    if voice_df.empty:
        return []

    nb_measure = int(voice_df["measure_position"].max()) + 1
    voice_in_measure = [[] for _ in range(nb_measure)]

    for measure_idx, measure_group in voice_df.groupby("measure_position"):
        notes_in_measure = []
        for onset_frame, onset_group in measure_group.groupby("onset_frame", sort=True):
            pitches_in_midi = onset_group["pitch"].astype(int).tolist()
            quarter_position_in_measure = onset_group["quarter_position_in_measure"].iloc[0]
            quarter_length = onset_group["quarter_length"].iloc[0]
            notes_in_measure.append((pitches_in_midi, quarter_position_in_measure, quarter_length))

        notes_in_measure.sort(key=lambda x: x[1])  # ordre chronologique dans la mesure
        voice_in_measure[int(measure_idx)] = notes_in_measure

    return voice_in_measure