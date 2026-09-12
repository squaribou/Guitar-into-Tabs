import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from constants import (BPM_RANGE, PERIODE_RESOLUTION, TOLERANCE, MIN_IOI,
                       STANDARD_POSITIONS, TIME_SIGNATURE, SMALLEST_STEP)

def estimate_tempo_from_attacks(filtered_attack_times, bpm_range=BPM_RANGE, period_resolution=PERIODE_RESOLUTION, tolerance=TOLERANCE, min_ioi=MIN_IOI):
    """
    Estimates the tempo (BPM) based on a list of note attack times.
    
    filtered_attack_times : list of start times (in seconds) for each note
    bpm_range : plausible tempo range to test (min BPM, max BPM)
    period_resolution : granularity of the period search (in seconds)
    tolerance : relative tolerance (fraction of the period) for an IOI to be considered a valid multiple
    min_ioi : ignores excessively short intervals between notes (ornaments, noise)

    Returns (tempo_bpm, confidence_score), where confidence_score is between 0 and 1.
    """
    times = np.sort(np.array(filtered_attack_times))
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


def assign_note_position(filtered_attacks, tempo_bpm, allowed_positions=STANDARD_POSITIONS, beats_per_measure=TIME_SIGNATURE):
    """
    Locating and assigning the position closest to the note based on the detected tempo, while saving the perceived error.
    filtered_attacks : list of (t, pitches)
    Returns : raw_positionned_notes, list of note characteristics (pitches, quarter_position_in_measure, measure_position, error)
    """
    quarter_note_duration = 60 / tempo_bpm
    raw_positionned_notes = []
    set_0 = filtered_attacks[0][0] if filtered_attacks else 0.0

    candidates = list(allowed_positions) + [beats_per_measure]

    for t, pitches in filtered_attacks:
        raw_quarter_position = (t - set_0) / quarter_note_duration
        measure_position = int(raw_quarter_position // beats_per_measure)
        raw_position_in_measure = raw_quarter_position - measure_position * beats_per_measure

        best_pos = None
        error = None
        for pos in candidates:
            distance = pos - raw_position_in_measure
            if error is None or abs(distance) < abs(error):
                error = distance
                best_pos = pos

        quarter_position_in_measure = best_pos

        if quarter_position_in_measure >= beats_per_measure:
            measure_position += 1
            quarter_position_in_measure = 0.0

        raw_positionned_notes.append((pitches, quarter_position_in_measure, measure_position, error))

    return raw_positionned_notes


def correct_notes_from_true_error(raw_positionned_notes, smallest_step=SMALLEST_STEP, beats_per_measure=TIME_SIGNATURE):
    """
    Corrects the theoretical note positions based on the "true" error (continuous drift, unwrapped), regardless of the number of
    SMALLEST_STEP increments exceeded.

    raw_positionned_notes : list of (pitches, quarter_position_in_measure, measure_position, error)
    Returns : corrected_notes, list of (pitches, quarter_position_in_measure, measure_position, error)
    """

    y = [note[-1] for note in raw_positionned_notes]
    true_error = np.unwrap(np.array(y), period=SMALLEST_STEP)

    corrected_notes = []

    for i, (pitches, quarter_position_in_measure, measure_position, error) in enumerate(raw_positionned_notes):
        n_steps = round(true_error[i] / smallest_step)
        offset = n_steps * smallest_step

        new_quarter_position = quarter_position_in_measure + offset
        new_measure_position = measure_position

        if new_quarter_position < 0:
            new_measure_position -= 1
            new_quarter_position += beats_per_measure
        elif new_quarter_position >= beats_per_measure:
            new_measure_position += 1
            new_quarter_position -= beats_per_measure

        corrected_notes.append((pitches, new_quarter_position, new_measure_position, error))

    return corrected_notes


def view_tempo_drift(notes, smallest_step=SMALLEST_STEP):
    """Plot the perceive error and the "true" error of the notes"""

    _, axes = plt.subplots(2, 1, figsize=(10, 8))
    y = [note[-1] for note in notes]
    x = [note[2] * 4 + note[1] for note in notes]
    axes[0].plot(x, y)
    axes[0].set_xlabel("Quater Position")
    axes[0].set_ylabel("Tempo Drift")

    y = [note[-1] for note in notes]
    true_error = np.unwrap(np.array(y), period=smallest_step)
    x_array = np.array([note[2] * 4 + note[1] for note in notes])
    slope, intercept, r_value, p_value, std_err = linregress(x_array, true_error)

    axes[1].plot(x, true_error)
    axes[1].set_xlabel("Quater Position")
    axes[1].set_ylabel("Tempo Drift Corrected")

    print(f"Pente (dérive par note) : {slope:.5f}")
    print(f"Ordonnée à l'origine   : {intercept:.5f}")
    x_array = np.array(x)
    axes[1].scatter(x_array, true_error, color="orange", s=15, zorder=3, label="Points sélectionnés")
    axes[1].plot(x_array, slope * x_array + intercept, color="red",
              linestyle="--", label=f"Régression (pente={slope:.5f})")

    print("Window displayed.")
    plt.show()


def assign_note_durations(notes, default_last_duration=1.0):
    """
    notes: list of (t, pitches) where pitches is a list of (midi, confidence)
    tempo_bpm: estimated tempo, in beats per minute
    allowed_durations: list of valid quarterLengths
    default_last_duration (temporary solution): quarterLength to use for the last note
    
    Returns a list of (t, pitches, quarter_length)
    """

    results = []
    n = len(notes)

    for i, (pitches, quarter_position_in_measure, measure) in enumerate(notes):
        if i < n - 1:
            next_t = notes[i + 1][1] + notes[i + 1][2] * 4
            quarter_length = min(next_t - (measure * 4 + quarter_position_in_measure), 4)
        else:
            quarter_length = default_last_duration

        results.append((pitches, quarter_position_in_measure, measure, quarter_length))

    return results
