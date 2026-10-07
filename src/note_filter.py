import pandas as pd
from constants import *

def _filter_chord_notes(candidates, harmonics_intervals=HARMONIC_INTERVALS):
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


def _stupid_filter(candidates, ratio_theshold=RATIO_THRESHOLD):
    to_remove = set()
    strong_note = max(candidates, key=lambda x: x[1])
    for candidate in candidates:
        if candidate[1] / strong_note[1] < ratio_theshold:
            to_remove.add(candidate[0])

    result = [candidate for candidate in candidates if candidate[0] not in to_remove]
    return sorted(result, key=lambda x: x[1], reverse=True)


def apply_chord_filters(note_df, harmonics_intervals=HARMONIC_INTERVALS, ratio_threshold=RATIO_THRESHOLD):
    """
    attacks : DataFrame [onset_frame, onset_time, pitch, confidence]
    Return a DataFrame filtered.
    """
    filtered_rows = []

    for onset_frame, group in note_df.groupby("onset_frame", sort=True):
        t = group["onset_time"].iloc[0]
        candidates = list(zip(group["pitch"], group["confidence"]))

        candidates = _filter_chord_notes(candidates, harmonics_intervals)
        candidates = _stupid_filter(candidates, ratio_threshold)

        for pitch, confidence in candidates:
            filtered_rows.append({
                "onset_frame": onset_frame,
                "onset_time": t,
                "pitch": pitch,
                "confidence": confidence,
            })

    return pd.DataFrame(filtered_rows, columns=note_df.columns)
