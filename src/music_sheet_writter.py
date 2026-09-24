from music21 import stream, tempo, note, chord, clef, key, meter, tie
import subprocess
import logging

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


def create_music_sheet(melody_voice_in_measure: list, low_voice_in_measure: list, tempo_bpm: int, beats_per_measure=BEATS_PER_MESURE) -> stream.Stream:
    """Create a music_sheet"""

    score = stream.Stream()
    score.append(clef.Treble8vbClef())
    score.append(meter.TimeSignature(f"{beats_per_measure}/4"))

    nb_measure = max(len(melody_voice_in_measure), len(low_voice_in_measure))
    melody_voice_in_measure.extend([[] for _ in range(nb_measure-len(melody_voice_in_measure))])
    low_voice_in_measure.extend([[] for _ in range(nb_measure-len(low_voice_in_measure))])
    pending_bass_tie = None

    for i in range(nb_measure):
        try:
            measure = stream.Measure()
            melody_voice = stream.Voice()
            low_voice = stream.Voice()

            # ___Add tempo to the first measure___ (temporary if the music have a tempo change)
            if i == 0:
                measure.insert(0, tempo.MetronomeMark(number=tempo_bpm))

            # ___Add the low tie note___
            if pending_bass_tie is not None:
                pitches = pending_bass_tie["pitches"]
                remaining = pending_bass_tie["remaining"]
                length_here = min(remaining, beats_per_measure)

                if len(pitches) == 1:
                    n = note.Note(midi=pitches[0], quarterLength=length_here)
                else:
                    n = chord.Chord(pitches, quarterLength=length_here)

                remaining -= length_here
                if remaining > 1e-6:
                    n.tie = tie.Tie("continue")
                    pending_bass_tie = {"pitches": pitches, "remaining": remaining}
                else:
                    n.tie = tie.Tie("stop")
                    pending_bass_tie = None

                low_voice.insert(0, n)

            # ___Add the melodie___ 
            for pitches_in_midi, quarter_position_in_measure, quarter_length in melody_voice_in_measure[i]:
                if quarter_length == 0:
                    ValueError("Quaterlength cannot be 0")
                if len(pitches_in_midi) == 1:
                    melody_voice.insert(quarter_position_in_measure, note.Note(midi=pitches_in_midi[0], quarterLength=quarter_length))
                else:
                    melody_voice.insert(quarter_position_in_measure, chord.Chord(pitches_in_midi, quarterLength=quarter_length))

            # ___Add low note and cut it if exceed a measure___
            for pitches_in_midi, quarter_position_in_measure, quarter_length in low_voice_in_measure[i]:
                space_left = beats_per_measure - quarter_position_in_measure

                if quarter_length == 0:
                    ValueError("Quaterlength cannot be 0")

                if quarter_length > space_left + 1e-6:
                    length_here = space_left
                    if len(pitches_in_midi) == 1:
                        n = note.Note(midi=pitches_in_midi[0], quarterLength=length_here)
                    else:
                        n = chord.Chord(pitches_in_midi, quarterLength=length_here)
                    n.tie = tie.Tie("start")
                    low_voice.insert(quarter_position_in_measure, n)

                    pending_bass_tie = {
                        "pitches": pitches_in_midi,
                        "remaining": quarter_length - length_here
                    }
                else:
                    if len(pitches_in_midi) == 1:
                        low_voice.insert(quarter_position_in_measure, note.Note(midi=pitches_in_midi[0], quarterLength=quarter_length))
                    else:
                        low_voice.insert(quarter_position_in_measure, chord.Chord(pitches_in_midi, quarterLength=quarter_length))

            measure.insert(0, melody_voice)
            measure.insert(0, low_voice)
            score.append(measure)

        except Exception as e:
            print(f"Error at measure {i} : {e}")
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
