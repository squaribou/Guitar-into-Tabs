from music21 import stream, tempo, note, chord, clef, key, meter, tie, beam
import math
import subprocess
import logging
import numpy as np

from constants import KEY_SIGNATURE_RANGE, MUSESCORE_PATH, BEATS_PER_MESURE

logging.basicConfig(filename="debug.log", level=logging.DEBUG)

def _force_sharp_spelling(score: stream.Stream) -> stream.Stream:
    """Forces all altered notes to be written as sharps, never as flats."""
    for element in score.recurse().notes:
        pitches_to_check = element.pitches if hasattr(element, 'pitches') else [element.pitch]
        
        for p in pitches_to_check:
            if p.accidental is not None and p.accidental.name == 'flat':
                p.getEnharmonic(inPlace=True)
    
    return score


def _count_accidentals_needed(all_pitches: list, key_signature: int) -> int:
    """
    Count how many accidentals would be displayed
    if this key signature were used (sharps: number of sharps; negative = flats).
    """
    ks = key.KeySignature(key_signature)
    count = 0
    
    for p in all_pitches:
        step = p.step
        expected_accidental = ks.accidentalByStep(step)
        expected_alter = expected_accidental.alter if expected_accidental else 0
        actual_alter = p.accidental.alter if p.accidental else 0
        
        if actual_alter != expected_alter:
            count += 1
    
    return count


def _find_best_key_signature(score: stream.Stream, key_signature_range=KEY_SIGNATURE_RANGE) -> int:
    """
    Find the key signature (among key_signature_range) that minimizes the number 
    of accidental alterations required in the music sheet.
    """
    all_pitches = []
    for element in score.recurse().notes:
        pitches = element.pitches if hasattr(element, 'pitches') else [element.pitch]
        all_pitches.extend(pitches)
    
    best_sharps = 0
    best_count = float('inf')
    
    for s in key_signature_range:
        c = _count_accidentals_needed(all_pitches, s)
        if c < best_count or (c == best_count and abs(s) < abs(best_sharps)):
            best_count = c
            best_sharps = s

    return best_sharps


def _duration_type_and_count(quarter_length):
    """Retourne le type music21 et le nombre de barres de ligature pour une durée donnée."""
    if quarter_length >= 1.0:
        return None, 0
    elif quarter_length >= 0.5:
        return 'eighth', 1
    elif quarter_length >= 0.25:
        return '16th', 2
    elif quarter_length >= 0.125:
        return '32nd', 3
    else:
        return '64th', 4


def _apply_manual_beams(voice: stream.Voice, beats_per_measure: int):
    """
    Applique des ligatures manuelles aux croches (et plus courtes) consécutives
    à l'intérieur d'un même temps, pour contourner le beaming automatique
    parfois incorrect de music21 (notamment avec des Voice multiples).
    """
    elements = list(voice.notesAndRests)
    beat_groups = {}

    for el in elements:
        if isinstance(el, note.Rest):
            continue  # un silence casse toujours la ligature
        beat_index = int(el.offset)  # 1 noire = 1 temps, donc int(offset) = numéro du temps
        beat_groups.setdefault(beat_index, []).append(el)

    for group in beat_groups.values():
        beamable = [el for el in group if el.quarterLength < 1.0]
        if len(beamable) < 2:
            continue  # il faut au moins 2 notes pour ligaturer

        for idx, el in enumerate(beamable):
            dur_type, _ = _duration_type_and_count(el.quarterLength)
            if dur_type is None:
                continue

            el.beams = beam.Beams()
            if idx == 0:
                beam_type = 'start'
            elif idx == len(beamable) - 1:
                beam_type = 'stop'
            else:
                beam_type = 'continue'

            el.beams.fill(dur_type, type=beam_type)


def _split_at_beats(voice: stream.Voice):
    """Coupe (avec liaison) les notes < 1 temps qui commencent entre deux temps
    et enjambent le temps suivant."""
    for el in list(voice.notesAndRests):
        if el.quarterLength >= 1.0:
            continue
        start = el.offset
        end = start + el.quarterLength
        next_beat = math.floor(start + 1e-6) + 1
        starts_off_beat = (start % 1) > 1e-6
        if starts_off_beat and end > next_beat + 1e-6:
            first, second = el.splitAtQuarterLength(next_beat - start)
            voice.remove(el)
            voice.insert(start, first)
            voice.insert(next_beat, second)


def _make_element(pitches, length):
    if len(pitches) == 1:
        return note.Note(midi=pitches[0], quarterLength=length)
    return chord.Chord(pitches, quarterLength=length)


def _fill_voice(voice, notes_in_measure, pending, beats_per_measure):
    """
    Remplit une Voice : insère d'abord la continuation d'une note liée venant de la
    mesure précédente, puis les notes de la mesure, en coupant celles qui dépassent
    la barre de mesure. Retourne le report éventuel pour la mesure suivante.
    """
    new_pending = None

    # Continuation de la mesure précédente
    if pending is not None:
        length_here = min(pending["remaining"], beats_per_measure)
        el = _make_element(pending["pitches"], length_here)
        remaining = pending["remaining"] - length_here
        if remaining > 1e-6:
            el.tie = tie.Tie("continue")
            new_pending = {"pitches": pending["pitches"], "remaining": remaining}
        else:
            el.tie = tie.Tie("stop")
        voice.insert(0, el)

    # Notes de la mesure
    for pitches, position, quarter_length in notes_in_measure:
        if quarter_length <= 0:
            raise ValueError("Quarterlength cannot be 0")
        space_left = beats_per_measure - position

        if quarter_length > space_left + 1e-6:
            el = _make_element(pitches, space_left)
            el.tie = tie.Tie("start")
            voice.insert(position, el)
            new_pending = {"pitches": pitches, "remaining": quarter_length - space_left}
        else:
            voice.insert(position, _make_element(pitches, quarter_length))

    return new_pending


def _finalize_last_note(voice_in_measure, beats_per_measure):
    """
    Trouve la note qui finit le plus tard. Dans sa dernière mesure :
      - si elle y dure strictement plus de 1 temps -> elle est prolongée jusqu'à la barre de mesure
      - sinon -> elle reste telle quelle et on renvoie le silence à ajouter après elle
    Retourne (index_mesure, position_du_silence, durée_du_silence) ou None.
    """
    best = None  # (end_abs, start_abs, mesure, index)
    for i, notes in enumerate(voice_in_measure):
        for k, (pitches, position, quarter_length) in enumerate(notes):
            start_abs = i * beats_per_measure + position
            end_abs = start_abs + quarter_length
            if best is None or end_abs > best[0] + 1e-9:
                best = (end_abs, start_abs, i, k)
    if best is None:
        return None

    end_abs, start_abs, i, k = best
    last_measure = int(np.ceil(end_abs / beats_per_measure - 1e-9)) - 1
    measure_start = last_measure * beats_per_measure
    measure_end = measure_start + beats_per_measure

    if abs(end_abs - measure_end) < 1e-9:
        return None  # finit déjà pile sur la barre de mesure

    duration_in_last_measure = end_abs - max(start_abs, measure_start)

    if duration_in_last_measure > 1 + 1e-9:
        pitches, position, quarter_length = voice_in_measure[i][k]
        voice_in_measure[i][k] = (pitches, position, quarter_length + (measure_end - end_abs))
        return None

    # note courte : on la laisse, et on comble la fin de mesure avec un silence
    return (last_measure, end_abs - measure_start, measure_end - end_abs)


def create_music_sheet(melody_voice_in_measure: list, low_voice_in_measure: list, tempo_bpm: int, beats_per_measure=BEATS_PER_MESURE) -> stream.Stream:
    """Create a music_sheet"""

    score = stream.Stream()
    score.append(clef.Treble8vbClef())
    score.append(meter.TimeSignature(f"{beats_per_measure}/4"))

    melody_rest = _finalize_last_note(melody_voice_in_measure, beats_per_measure)
    low_rest = _finalize_last_note(low_voice_in_measure, beats_per_measure)

    def _last_end(voice):
        ends = [i * beats_per_measure + p + ql
                for i, notes in enumerate(voice) for _, p, ql in notes]
        return max(ends, default=0)

    last_end = max(_last_end(melody_voice_in_measure), _last_end(low_voice_in_measure))
    nb_measure = max(len(melody_voice_in_measure), len(low_voice_in_measure),
                     int(np.ceil(last_end / beats_per_measure - 1e-9)))

    melody_voice_in_measure.extend([[] for _ in range(nb_measure - len(melody_voice_in_measure))])
    low_voice_in_measure.extend([[] for _ in range(nb_measure - len(low_voice_in_measure))])

    pending_melody_tie = None
    pending_bass_tie = None

    for i in range(nb_measure):
        try:
            measure = stream.Measure(number=i+1)
            melody_voice = stream.Voice()
            low_voice = stream.Voice()

            if i == 0:
                measure.insert(0, tempo.MetronomeMark(number=tempo_bpm))

            # ___Silence manuel si la mélodie ne commence pas à 0___
            # (inutile si une note liée de la mesure précédente occupe déjà le début)
            if pending_melody_tie is None and low_voice_in_measure[i]:
                if melody_voice_in_measure[i]:
                    first_melodie_position = melody_voice_in_measure[i][0][1]
                else:
                    first_melodie_position = beats_per_measure
                if first_melodie_position > 0:
                    rest_duration = min(first_melodie_position, beats_per_measure)
                    melody_voice.insert(0, note.Rest(quarterLength=rest_duration))

            # ___Remplissage des deux voix (avec ties entre mesures)___
            pending_melody_tie = _fill_voice(melody_voice, melody_voice_in_measure[i],
                                             pending_melody_tie, beats_per_measure)
            pending_bass_tie = _fill_voice(low_voice, low_voice_in_measure[i],
                                           pending_bass_tie, beats_per_measure)

            if melody_rest and melody_rest[0] == i:
                melody_voice.insert(melody_rest[1], note.Rest(quarterLength=melody_rest[2]))
            if low_rest and low_rest[0] == i:
                low_voice.insert(low_rest[1], note.Rest(quarterLength=low_rest[2]))

            # ___Ligatures (après le découpage à la barre de mesure)___
            _split_at_beats(melody_voice)
            _apply_manual_beams(melody_voice, beats_per_measure)

            print(i+1, "mélodie:", melody_voice.highestTime, "basse:", low_voice.highestTime,
                  "attendu:", beats_per_measure)
            measure.insert(0, melody_voice)
            measure.insert(0, low_voice)
            score.append(measure)

        except Exception as e:
            print(f"Error at measure {i+1} : {e}")
            break

    try:
        _force_sharp_spelling(score)
        best_key_signature = _find_best_key_signature(score)
        ks = key.KeySignature(best_key_signature)

        first_measure = score.getElementsByClass(stream.Measure).first()
        first_measure.insert(0, ks)

        # Reboot sharp spelling after adding the key signature, to ensure consistency
        for n in score.recurse().notes:
            for p in n.pitches:
                if p.accidental is not None:
                    p.accidental.displayStatus = None
        score.makeAccidentals(useKeySignature=ks, overrideStatus=True, inPlace=True)

    except Exception as e:
        print(f"Error occurred while processing key signature: {e}")

    return score


def display_music_sheet(score: stream.Stream)->None:
    """Save and display the music_sheet in Musescore"""
    try:
        output_path = "music_sheets/my_music_sheet.xml"
        score.write("musicxml", fp=output_path)
        print(f"XML written to {output_path}")

        subprocess.Popen([MUSESCORE_PATH, output_path])
        print("Partition opened")
    except Exception as e:
        print(f"Error occurred while displaying music sheet: {e}")
