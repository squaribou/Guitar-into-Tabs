import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd

from constants import *

def estimate_tempo_from_attacks(note_df_onset_time, bpm_range=BPM_RANGE, period_resolution=PERIODE_RESOLUTION, tolerance=TOLERANCE, min_ioi=MIN_IOI):
    """
    Estimates the tempo (BPM) based on a list of note attack times.
    
    filtered_attack_times : list of start times (in seconds) for each note
    bpm_range : plausible tempo range to test (min BPM, max BPM)
    period_resolution : granularity of the period search (in seconds)
    tolerance : relative tolerance (fraction of the period) for an IOI to be considered a valid multiple
    min_ioi : ignores excessively short intervals between notes (ornaments, noise)

    Returns (tempo_bpm, confidence_score), where confidence_score is between 0 and 1.
    """
    times = np.sort(np.array(note_df_onset_time))
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

    return round(tempo_bpm, 2), round(confidence, 3)


def assign_note_position(note_df, tempo_bpm, allowed_positions=STANDARD_POSITIONS, beats_per_measure=BEATS_PER_MESURE):
    """
    Locating and assigning the position closest to the note based on the detected tempo,
    while saving the perceived error.

    filtered_attacks : DataFrame avec colonnes onset_frame, onset_time, pitch, confidence
    Retourne : filtered_attacks avec 3 colonnes ajoutées :
        quarter_position_in_measure, measure_position, error
    """
    quarter_note_duration = 60 / tempo_bpm
    unique_times = np.sort(note_df["onset_time"].unique())
    t0 = unique_times[0] if len(unique_times) else 0.0

    candidates = np.array(list(allowed_positions) + [beats_per_measure])

    position_by_time = {}
    for t in unique_times:
        raw_quarter_position = (t - t0) / quarter_note_duration
        measure_position = int(raw_quarter_position // beats_per_measure)
        raw_position_in_measure = raw_quarter_position - measure_position * beats_per_measure

        distances = candidates - raw_position_in_measure
        best_idx = np.argmin(np.abs(distances))
        best_pos = candidates[best_idx]
        error = distances[best_idx]

        if best_pos >= beats_per_measure:
            measure_position += 1
            best_pos = 0.0

        position_by_time[t] = (best_pos, measure_position, error)

    result = note_df.copy()
    result["quarter_position_in_measure"] = result["onset_time"].map(lambda t: position_by_time[t][0])
    result["measure_position"] = result["onset_time"].map(lambda t: position_by_time[t][1])
    result["error"] = result["onset_time"].map(lambda t: position_by_time[t][2])

    return result


def correct_notes_from_true_error(note_df, smallest_step=SMALLEST_STEP, beats_per_measure=BEATS_PER_MESURE):
    """
    Corrects the theoretical note positions based on the "true" error (continuous drift, unwrapped),
    regardless of the number of SMALLEST_STEP increments exceeded.

    positionned_notes : DataFrame avec colonnes onset_frame, onset_time, pitch, confidence,
        quarter_position_in_measure, measure_position, error
    Retourne : positionned_notes avec quarter_position_in_measure et measure_position corrigées.
    """
    onset_summary = (
        note_df
        .drop_duplicates(subset="onset_time")
        .sort_values("onset_time")
    )

    y = onset_summary["error"].to_numpy()
    true_error = np.unwrap(y, period=smallest_step)

    n_steps = np.round(true_error / smallest_step).astype(int)
    offset = n_steps * smallest_step

    new_quarter_position = onset_summary["quarter_position_in_measure"].to_numpy() + offset
    new_measure_position = onset_summary["measure_position"].to_numpy().copy()

    below = new_quarter_position < 0
    new_measure_position[below] -= 1
    new_quarter_position[below] += beats_per_measure

    above = new_quarter_position >= beats_per_measure
    new_measure_position[above] += 1
    new_quarter_position[above] -= beats_per_measure

    correction_by_time = {
        t: (qpos, mpos)
        for t, qpos, mpos in zip(onset_summary["onset_time"], new_quarter_position, new_measure_position)
    }

    result = note_df.copy()
    result["quarter_position_in_measure"] = result["onset_time"].map(lambda t: correction_by_time[t][0])
    result["measure_position"] = result["onset_time"].map(lambda t: correction_by_time[t][1])

    return result


def view_tempo_drift(note_df, smallest_step=SMALLEST_STEP, beats_per_measure=BEATS_PER_MESURE):
    """Plot the perceived error and the "true" error of the notes"""

    onset_summary = (
        note_df
        .drop_duplicates(subset="onset_frame")
        .sort_values("onset_frame")
    )

    y = onset_summary["error"].to_numpy()
    x = (onset_summary["measure_position"] * beats_per_measure + onset_summary["quarter_position_in_measure"]).to_numpy()

    _, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(x, y)
    axes[0].set_xlabel("Quater Position")
    axes[0].set_ylabel("Tempo Drift")

    true_error = np.unwrap(y, period=smallest_step)
    slope, intercept, r_value, p_value, std_err = linregress(x, true_error)

    axes[1].plot(x, true_error)
    axes[1].set_xlabel("Quater Position")
    axes[1].set_ylabel("Tempo Drift Corrected")

    print(f"Pente (dérive par note) : {slope:.5f}")
    print(f"Ordonnée à l'origine   : {intercept:.5f}")
    axes[1].scatter(x, true_error, color="orange", s=15, zorder=3, label="Points sélectionnés")
    axes[1].plot(x, slope * x + intercept, color="red",
                 linestyle="--", label=f"Régression (pente={slope:.5f})")

    print("Window displayed.")
    plt.show()


def _merge_dropped_notes(note_df, eps=1e-9):
    """
    Supprime les notes de quarter_length nul, mais prolonge la note précédente
    de la même voix (si elle était contiguë) jusqu'au prochain onset conservé.
    """
    note_df = note_df.sort_values(["voice", "abs_position"])
    is_zero = note_df["quarter_length"] <= eps
    kept = note_df[~is_zero].copy()
    dropped = note_df[is_zero]

    for voice, d_group in dropped.groupby("voice"):
        voice_kept = kept[kept["voice"] == voice]
        if voice_kept.empty:
            continue

        for _, d in d_group.iterrows():
            d_pos = d["abs_position"]

            # onset conservé précédent (strictement avant la note supprimée)
            before = voice_kept[voice_kept["abs_position"] < d_pos - eps]
            if before.empty:
                continue
            prev_pos = before["abs_position"].max()
            prev_rows = kept[(kept["voice"] == voice) & (np.isclose(kept["abs_position"], prev_pos))]

            # la précédente touchait-elle la note supprimée ? (sinon c'était un vrai silence)
            prev_end = prev_pos + prev_rows["quarter_length"].max()
            if prev_end < d_pos - eps:
                continue

            # prochain onset conservé après la note supprimée
            after = voice_kept[voice_kept["abs_position"] > d_pos + eps]
            if after.empty:
                continue  # dernière note : on ne prolonge pas à l'infini
            new_length = after["abs_position"].min() - prev_pos

            # on ne fait que rallonger, jamais raccourcir
            to_extend = prev_rows.index[prev_rows["quarter_length"] < new_length - eps]
            kept.loc[to_extend, "quarter_error_adjust"] = (
                new_length - kept.loc[to_extend, "quarter_length"]
            )
            kept.loc[to_extend, "quarter_length"] = new_length

    # recalcul de l'erreur pour les notes prolongées
    if "quarter_error_adjust" in kept.columns:
        kept["quarter_length_error"] -= kept["quarter_error_adjust"].fillna(0)
        kept = kept.drop(columns="quarter_error_adjust")

    return kept.sort_index()


def assign_note_quarter_length(note_df, tempo_bpm, allowed_positions=STANDARD_POSITIONS, 
                                 beats_per_measure=BEATS_PER_MESURE, snap_to_next_ratio=0.85):
    """
    Add quarter_length to note_df based on the measured duration (in seconds),
    bounded by the next onset in the same voice. If the measured duration covers
    at least `snap_to_next_ratio` of that gap, quarter_length snaps exactly to
    the gap (the note is considered sustained until the next note).
    Notes whose quarter_length ends up being 0 are dropped.
    """
    quarter_note_duration = 60 / tempo_bpm
    candidates = np.array(allowed_positions)
    note_df = note_df.copy()

    note_df["abs_position"] = (
        note_df["measure_position"] * beats_per_measure + note_df["quarter_position_in_measure"]
    )

    # --- borne max par note : écart jusqu'au prochain onset de la même voix ---
    max_quarter_length = pd.Series(np.inf, index=note_df.index)

    for voice_name, group in note_df.groupby("voice"):
        onset_positions = (
            group.drop_duplicates(subset="onset_frame")
            .sort_values("abs_position")
        )
        abs_pos = onset_positions["abs_position"].to_numpy()

        next_gap = np.empty(len(abs_pos))
        next_gap[:-1] = np.diff(abs_pos)
        next_gap[-1] = np.inf

        gap_by_onset = dict(zip(onset_positions["onset_frame"], next_gap))
        max_quarter_length.loc[group.index] = group["onset_frame"].map(gap_by_onset)

    raw_quarter_length = (note_df["duration"] / quarter_note_duration).to_numpy()
    max_q = max_quarter_length.to_numpy()

    # --- choix du candidat le plus proche, parmi ceux qui respectent la borne ---
    candidate_matrix = np.broadcast_to(candidates, (len(note_df), len(candidates)))
    distances = candidates[np.newaxis, :] - raw_quarter_length[:, np.newaxis]
    abs_distances = np.abs(distances)

    invalid = candidate_matrix > (max_q[:, np.newaxis] + 1e-9)
    abs_distances = np.where(invalid, np.inf, abs_distances)

    best_idx = np.argmin(abs_distances, axis=1)
    chosen_quarter_length = candidates[best_idx]

    no_valid_candidate = np.isinf(abs_distances[np.arange(len(note_df)), best_idx])
    if no_valid_candidate.any():
        best_idx[no_valid_candidate] = np.argmin(candidates)
        chosen_quarter_length = candidates[best_idx]

    # --- snap : si la durée mesurée couvre suffisamment le gap, on colle exactement au gap ---
    has_bound = np.isfinite(max_q)
    should_snap = has_bound & (raw_quarter_length >= max_q * snap_to_next_ratio)
    chosen_quarter_length = np.where(should_snap, max_q, chosen_quarter_length)

    note_df["quarter_length"] = chosen_quarter_length
    note_df["quarter_length_error"] = raw_quarter_length - note_df["quarter_length"]

    # --- suppression des durées nulles, en prolongeant la note précédente ---
    note_df = _merge_dropped_notes(note_df)

    return note_df


def extract_attacks_duration(notes_df, prob_matrix, frame_time=FRAME_TIME, midi_offset=MIDI_OFFSET,
                           threshold=0.2, hysteresis_frames=3, min_duration_frames=1):
    """
    notes_df : DataFrame avec colonnes onset_frame, onset_time, pitch, confidence
    prob_matrix : array (n_frames, n_pitches)

    Return notes_df with add column "duration" (in seconds).
    """
    n_frames, n_pitches = prob_matrix.shape
    durations = np.empty(len(notes_df))

    # Pour chaque pitch, la liste triée des frames d'onset (pour borner par le prochain onset)
    onsets_by_pitch = {
        pitch: sorted(group["onset_frame"].tolist())
        for pitch, group in notes_df.groupby("pitch")
    }

    for i, note in enumerate(notes_df.itertuples()):
        column = note.pitch - midi_offset
        if not (0 <= column < n_pitches):
            durations[i] = np.nan
            continue

        same_pitch_onsets = onsets_by_pitch[note.pitch]
        next_onset_frame = next(
            (f for f in same_pitch_onsets if f > note.onset_frame), n_frames
        )

        last_active_frame = note.onset_frame
        below_streak = 0

        for frame in range(note.onset_frame, next_onset_frame):
            if prob_matrix[frame, column] >= threshold:
                last_active_frame = frame
                below_streak = 0
            else:
                below_streak += 1
                if below_streak >= hysteresis_frames:
                    break

        duration_frames = max(last_active_frame - note.onset_frame + 1, min_duration_frames)
        durations[i] = duration_frames * frame_time

    notes_df = notes_df.copy()
    notes_df["duration"] = durations
    return notes_df
