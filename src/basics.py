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
    positioned, and/or corrected note locations (starts and ends).

    note_df : DataFrame avec colonnes onset_frame, onset_time, pitch, confidence, duration,
        quarter_position_in_measure, measure_position, error, quarter_length
    """

    print("Prepare plotting...")
    start_frame = int(start_time / FRAME_TIME)
    end_frame = int(end_time / FRAME_TIME)

    def frames_pitches(frame_idx_series, pitch_mask_source):
        mask = (frame_idx_series >= start_frame) & (frame_idx_series <= end_frame)
        frames = frame_idx_series[mask].tolist()
        pitches = (pitch_mask_source.loc[mask] - MIDI_OFFSET).tolist()
        return frames, pitches

    # --- points rouges : débuts d'onset bruts ---
    attack_frame_idx = note_df["onset_time"] / FRAME_TIME
    attack_frames, attack_pitches = frames_pitches(attack_frame_idx, note_df["pitch"])

    # --- points jaunes : débuts positionnés/corrigés ---
    zero_offset_frame = note_df["onset_time"].min() / FRAME_TIME
    quarter_duration = 60 / tempo_bpm

    start_t_corrected = (
        note_df["measure_position"] * beats_per_measure
        + note_df["quarter_position_in_measure"]
    ) * quarter_duration
    start_frame_idx_corrected = start_t_corrected / FRAME_TIME + zero_offset_frame
    duration_frames, duration_pitches = frames_pitches(start_frame_idx_corrected, note_df["pitch"])

    # --- points bleus : fin des notes mesurée physiquement (onset_time + duration) ---
    end_t_raw = note_df["onset_time"] + note_df["duration"]
    end_frame_idx_raw = end_t_raw / FRAME_TIME
    end_frames_raw, end_pitches_raw = frames_pitches(end_frame_idx_raw, note_df["pitch"])

    # --- points violets : fin des notes corrigées (start quantifié + quarter_length) ---
    end_t_corrected = start_t_corrected + note_df["quarter_length"] * quarter_duration
    end_frame_idx_corrected = end_t_corrected / FRAME_TIME + zero_offset_frame
    end_frames_corrected, end_pitches_corrected = frames_pitches(end_frame_idx_corrected, note_df["pitch"])

    _, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(model_output["onset"].T, aspect="auto", origin="lower")
    axes[0].set_title("Probabilités d'onset")
    axes[0].set_ylim(top=60, bottom=20)
    axes[0].set_xlim(left=start_frame, right=end_frame)

    axes[1].imshow(model_output["note"].T, aspect="auto", origin="lower")
    axes[1].set_title("Probabilités de note")
    axes[1].set_ylim(top=60, bottom=20)
    axes[1].set_xlim(left=start_frame, right=end_frame)

    for ax in axes:
        ax.scatter(attack_frames, attack_pitches, color="red", s=5, marker="o", zorder=3, label="Début (brut)")
        ax.scatter(duration_frames, duration_pitches, color="yellow", s=5, marker="o", zorder=3, label="Début (corrigé)")
        ax.scatter(end_frames_raw, end_pitches_raw, color="blue", s=5, marker="o", zorder=3, label="Fin (mesurée)")
        ax.scatter(end_frames_corrected, end_pitches_corrected, color="purple", s=5, marker="o", zorder=3, label="Fin (corrigée)")

    sixteenth_duration = quarter_duration / 4
    sixteenth_duration_frames = sixteenth_duration / FRAME_TIME
    n_steps_to_start = math.floor((start_frame - zero_offset_frame) / sixteenth_duration_frames)
    sixteenth_frame = zero_offset_frame + n_steps_to_start * sixteenth_duration_frames
    while sixteenth_frame <= end_frame:
        axes[0].axvline(sixteenth_frame, color="gray", linewidth=0.5, alpha=0.5, zorder=1)
        axes[1].axvline(sixteenth_frame, color="gray", linewidth=0.5, alpha=0.5, zorder=1)
        sixteenth_frame += sixteenth_duration_frames

    axes[0].legend(loc="upper right", fontsize=6)

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
            (idx for idx, pitch in zip(group.index, group["partition_pitch"]) if pitch < lowest_melody_note),
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
            pitches_in_midi = onset_group["partition_pitch"].astype(int).tolist()
            quarter_position_in_measure = onset_group["quarter_position_in_measure"].iloc[0]
            quarter_length = onset_group["quarter_length"].iloc[0]
            notes_in_measure.append((pitches_in_midi, quarter_position_in_measure, quarter_length))

        notes_in_measure.sort(key=lambda x: x[1])  # ordre chronologique dans la mesure
        voice_in_measure[int(measure_idx)] = notes_in_measure

    return voice_in_measure