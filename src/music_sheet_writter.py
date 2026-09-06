from music21 import stream, tempo, note, chord, clef, key, meter
import subprocess
import logging

from constants import KEY_SIGNATURE_RANGE, MUSESCORE_PATH

logging.basicConfig(filename="debug.log", level=logging.DEBUG)

def _force_sharp_spelling(music_sheet: stream.Stream)->stream.Stream:
    """Forces all altered notes to be written as sharps, never as flats."""
    for element in music_sheet.recurse().notes:
        pitches_to_check = element.pitches if hasattr(element, 'pitches') else [element.pitch]
        
        for p in pitches_to_check:
            if p.accidental is not None and p.accidental.name == 'flat':
                p.getEnharmonic(inPlace=True)
    
    return music_sheet


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


def _find_best_key_signature(music_sheet: stream.Stream, key_signature_range=KEY_SIGNATURE_RANGE)->int:
    """
    Find the key signature (among key_signature_range) that minimizes the number 
    of accidental alterations required in the music sheet.
    """
    all_pitches = []
    for element in music_sheet.recurse().notes:
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


def create_music_sheet(pitches_in_midi: list, tempo_bpm: int)->stream.Stream:
    """Create a music_sheet"""
    music_sheet = stream.Stream()
    music_sheet.append(tempo.MetronomeMark(number=tempo_bpm))
    music_sheet.append(clef.Treble8vbClef())
    for pitch_in_midi, quaterlenght in pitches_in_midi:
        if len(pitch_in_midi) == 1:
            music_sheet.append(note.Note(midi=pitch_in_midi[0], quarterLength=quaterlenght))
        else:
            music_sheet.append(chord.Chord(pitch_in_midi, quarterLength=quaterlenght))

    _force_sharp_spelling(music_sheet)
    best_key_signature = _find_best_key_signature(music_sheet)
    music_sheet.insert(0, key.KeySignature(best_key_signature))
    music_sheet.append(meter.TimeSignature('4/4'))
    return music_sheet


def display_music_sheet(music_sheet: stream.Stream)->None:
    """Save and display the music_sheet in Musescore"""
    output_path = "music_sheets/my_music_sheet.xml"
    music_sheet.write("musicxml", fp=output_path)
    print(f"XML written to {output_path}")

    subprocess.Popen([MUSESCORE_PATH, output_path])
    print("Partition opened")
