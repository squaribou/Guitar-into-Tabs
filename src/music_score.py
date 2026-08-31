from music21 import stream, note, clef
from music21 import environment
import subprocess

def create_partition_without_tempo(pitches_in_midi: list) -> None:
    """Create a partition in Musescore"""

    # Configure access path to the Musecore app (it needs to be install)
    env = environment.Environment()
    env['musicxmlPath'] = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
    env['musescoreDirectPNGPath'] = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
    print(env['musicxmlPath'])

    partition = stream.Stream()
    partition.append(clef.Treble8vbClef())
    for pitch_in_midi in pitches_in_midi:
        partition.append(note.Note(midi=pitch_in_midi, quarterLength=1))

    partition.write("musicxml", "ma_partition.xml")
    subprocess.Popen(["C:/Program Files/MuseScore 4/bin/MuseScore4.exe", "ma_partition.xml"])
    return


if __name__ == "__main__":
    pitches_in_midi = [60] #C4
    create_partition_without_tempo(pitches_in_midi)