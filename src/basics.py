import numpy as np

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


def filter_chord_notes(candidates, harmonics_intervals=HARMONIC_INTERVALS):
    """
    candidates: list of (MIDI, confidence) pairs detected for the same attack.
    Returns the filtered list of actual notes (chord or single note + harmonics).
    """
    if not candidates:
        return []

    # --- STEP 1 : groups the note by degre (same note, different octave) ---
    groups = {}
    for midi, confidence in candidates:
        degre = midi % 12
        groups.setdefault(degre, []).append((midi, confidence))

    survivors = {}
    for degre, members in groups.items():
        representative = max(members, key=lambda x: x[1])
        fundamental = min(members, key=lambda x: x[0])
        survivors[degre] = {"representative": representative, "fundamental": fundamental}

    # --- STEP 2 : eliminate harmonics ---
    to_remove = set()
    changed = True
    while changed:
        changed = False
        for degre_a, data_a in survivors.items():
            if degre_a in to_remove:
                continue
            fund_midi, fund_conf = data_a["fundamental"]

            for degre_b, data_b in survivors.items():
                if degre_b == degre_a or degre_b in to_remove:
                    continue
                rep_midi, rep_conf = data_b["representative"]
                interval = rep_midi - fund_midi
                if interval <= 0:
                    continue

                for harmonic_interval, max_ratio in harmonics_intervals.items():
                    if interval == harmonic_interval:
                        ratio = rep_conf / fund_conf if fund_conf > 0 else 1.0
                        if ratio <= max_ratio:
                            to_remove.add(degre_b)
                            changed = True
                        break

    result = [data["representative"] for pc, data in survivors.items() if pc not in to_remove]
    return sorted(result, key=lambda x: x[1], reverse=True)


def stupid_filter(candidates, ratio_theshold=RATIO_THRESHOLD):
    to_remove = set()
    strong_note = max(candidates, key=lambda x: x[1])
    for candidate in candidates:
        if candidate[1] / strong_note[1] < ratio_theshold:
            to_remove.add(candidate[0])

    result = [candidate for candidate in candidates if candidate[0] not in to_remove]
    return sorted(result, key=lambda x: x[1], reverse=True)


def estimate_tempo_from_attacks(attack_times, bpm_range=BPM_RANGE, period_resolution=PERIODE_RESOLUTION,
                                tolerance=TOLERANCE, min_ioi=MIN_IOI):
    """
    Estimates the tempo (BPM) based on a list of note attack times.
    
    attack_times : list/array of start times (in seconds) for each note
    bpm_range : plausible tempo range to test (min BPM, max BPM)
    period_resolution : granularity of the period search (in seconds)
    tolerance : relative tolerance (fraction of the period) for an IOI to be considered a valid multiple
    min_ioi : ignores excessively short intervals between notes (ornaments, noise)

    Returns (tempo_bpm, confidence_score), where confidence_score is between 0 and 1.
    """
    times = np.sort(np.array(attack_times))
    iois = np.diff(times)  # inter-onset intervals
    iois = iois[iois > min_ioi]

    if len(iois) == 0:
        return None, 0.0

    period_min = 60 / bpm_range[1]
    period_max = 60 / bpm_range[0]
    candidate_periods = np.arange(period_min, period_max, period_resolution)

    best_period = None
    best_score = -1

    for period in candidate_periods:
        multiples = np.round(iois / period)
        multiples[multiples == 0] = 1
        expected = multiples * period
        relative_error = np.abs(iois - expected) / period

        score = np.sum(np.clip(1 - relative_error / tolerance, 0, 1))

        if score > best_score:
            best_score = score
            best_period = period

    tempo_bpm = 60 / best_period
    confidence = best_score / len(iois)

    return round(tempo_bpm, 1), round(confidence, 3)


def assign_note_durations(attacks, tempo_bpm, allowed_durations=STANDARD_DURATIONS, default_last_duration=1.0):
    """
    attacks: list of (t, pitches) where pitches is a list of (midi, confidence)
    tempo_bpm: estimated tempo, in beats per minute
    allowed_durations: list of valid quarterLengths
    default_last_duration: quarterLength to use for the last note
    
    Returns a list of (t, pitches, quarter_length)
    """

    quarter_note_duration = 60 / tempo_bpm
    results = []
    n = len(attacks)

    for i, (t, pitches) in enumerate(attacks):
        if i < n - 1:
            next_t = attacks[i + 1][0]
            duration_seconds = next_t - t
            raw_quarter_length = duration_seconds / quarter_note_duration
            quarter_length = min(allowed_durations, key=lambda d: abs(d - raw_quarter_length))
        else:
            quarter_length = default_last_duration

        results.append((t, pitches, quarter_length))

    return results
