import pandas as pd
from constants import *

def _filter_chord_notes(candidates, previous_pitches=None, harmonic_intervals=HARMONIC_INTERVALS,
                        previous_penalty=0.5, low_note_adjustment=LOW_NOTE_ADJUSTMENT, octave_intervals=OCTAVE_INTERVALS):
    """
    candidates: list of (MIDI, confidence, is_low_note) triplets for the same attack.
    Returns a list of (MIDI, confidence, is_low_note) triplets.
    """
    if previous_pitches is None:
        previous_pitches = set()
    if not candidates:
        return []

    # --- STEP 0 : dédoublonnage par pitch exact ---
    best = {}
    for midi, conf, is_low in candidates:
        if midi not in best or conf > best[midi][1]:
            best[midi] = (midi, conf, bool(is_low))

    # --- STEP 1 : vérification d'octave, uniquement sous la note la plus confiante ---
    strongest = max(best.values(), key=lambda x: x[1])
    s_midi, s_conf, _ = strongest

    ghosts = set()
    if s_conf > 0:
        for interval, octave_min_ratio in octave_intervals.items():
            lower = best.get(s_midi - interval)
            if lower is not None and lower[1] / s_conf < octave_min_ratio:
                ghosts.add(lower[0])
    pool = [c for c in best.values() if c[0] not in ghosts]

    # --- STEP 2 : élimination des harmoniques, du grave vers l'aigu ---
    kept = []
    for midi, conf, is_low in sorted(pool, key=lambda x: x[0]):
        eff_conf = conf * previous_penalty if midi in previous_pitches else conf
        is_harmonic = False

        for f_midi, f_conf, f_is_low in kept:
            max_ratio = harmonic_intervals.get(midi - f_midi)
            if max_ratio is None:
                continue
            if f_is_low:
                max_ratio *= low_note_adjustment
            ratio = eff_conf / f_conf if f_conf > 0 else 1.0
            if ratio <= max_ratio:
                is_harmonic = True
                break

        if not is_harmonic:
            kept.append((midi, conf, is_low))

    return sorted(kept, key=lambda x: x[1], reverse=True)


def _stupid_filter(candidates, ratio_threshold=RATIO_THRESHOLD, ratio_threshold_low_note=RATIO_THRESHOLD_LOW_NOTE):
    to_remove = set()
    strong_note = max(candidates, key=lambda x: x[1])
    for candidate in candidates:
        effective_threshold = ratio_threshold_low_note if candidate[2] else ratio_threshold
        if candidate[1] / strong_note[1] < effective_threshold:
            to_remove.add(candidate[0])

    result = [candidate for candidate in candidates if candidate[0] not in to_remove]
    return sorted(result, key=lambda x: x[1], reverse=True)


def apply_chord_filters(note_df):
    """note_df : DataFrame with at least [onset_frame, pitch, confidence, voice]"""
    filtered_groups = []
    previous_pitches = set()

    for onset_frame, group in note_df.groupby("onset_frame", sort=True):
        is_low = (group["voice"] == "low")
        candidates = list(zip(group["pitch"], group["confidence"], is_low))

        candidates = _filter_chord_notes(candidates, previous_pitches)
        candidates = _stupid_filter(candidates)

        kept_pitches = {c[0] for c in candidates}
        filtered_groups.append(group[group["pitch"].isin(kept_pitches)])
        previous_pitches = kept_pitches

    if not filtered_groups:
        return note_df.iloc[0:0]

    return pd.concat(filtered_groups, ignore_index=True)
