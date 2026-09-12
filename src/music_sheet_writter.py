from music21 import stream, tempo, note, chord, clef, key, meter, instrument, interval
import subprocess
import logging

from constants import KEY_SIGNATURE_RANGE, MUSESCORE_PATH, TIME_SIGNATURE

logging.basicConfig(filename="debug.log", level=logging.DEBUG)

def _force_sharp_spelling(score: stream.Stream)->stream.Stream:
    """Forces all altered notes to be written as sharps, never as flats."""
    for element in score.recurse().notes:
        pitches_to_check = element.pitches if hasattr(element, 'pitches') else [element.pitch]
        
        for p in pitches_to_check:
            if p.accidental is not None and p.accidental.name == 'flat':
                p.getEnharmonic(inPlace=True)
    
    return score


def _count_accidentals_needed(all_pitches: list, key_signature: int)->int:
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


def _find_best_key_signature(score: stream.Stream, key_signature_range=KEY_SIGNATURE_RANGE)->int:
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


def create_music_sheet(voix_melodie_notes: list, voix_basse_notes: list, tempo_bpm: int)->stream.Stream:
    """Create a music_sheet"""

    score = stream.Stream()
    score.append(clef.Treble8vbClef())
    score.append(meter.TimeSignature(f"{TIME_SIGNATURE}/4"))

    nb_measure = max(len(voix_melodie_notes), len(voix_basse_notes))
    for i in range(nb_measure):
        try:
            measure = stream.Measure()
            if i == 0:
                measure.insert(0, tempo.MetronomeMark(number=tempo_bpm))
            voix_melodie = stream.Voice()
            voix_basse = stream.Voice()
            for pitches_in_midi, quarter_position_in_measure, quarter_length in voix_melodie_notes[i]:
                if quarter_length == 0:
                    quarter_length = 0.125
                if len(pitches_in_midi) == 1:
                    voix_melodie.insert(quarter_position_in_measure, note.Note(midi=pitches_in_midi[0], quarterLength=quarter_length))
                else:
                    voix_melodie.insert(quarter_position_in_measure, chord.Chord(pitches_in_midi, quarterLength=quarter_length))
            for pitches_in_midi, quarter_position_in_measure, quarter_length in voix_basse_notes[i]:
                voix_basse.insert(quarter_position_in_measure, note.Note(midi=pitches_in_midi[0], quarterLength=quarter_length))

            measure.insert(0, voix_melodie)
            measure.insert(0, voix_basse)
            score.append(measure)
        except IndexError:
            print(f"error at measure {i}")
            break

    try:
        _force_sharp_spelling(score)
        best_key_signature = _find_best_key_signature(score)
        score.insert(0, key.KeySignature(best_key_signature))
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
