import librosa
import matplotlib.pyplot as plt
import logging
import pandas as pd

logging.getLogger("root").setLevel(logging.ERROR)

from constants import SAMPLE_RATE, MAX_TEMPO_BPM
from basics import predict_from_array, analysis, extract_note_attacks, assign_voice, build_voice_in_measure
from frequence_analyser import get_fft, top_2_fundamental_frequencies
from audio_listener import listen_audio
from music_sheet_writter import create_music_sheet, display_music_sheet
from tempo import view_tempo_drift, assign_note_position, estimate_tempo_from_attacks, correct_notes_from_true_error, extract_attacks_duration, assign_note_quarter_length
from note_filter import apply_chord_filters


def get_note_fft(file_path, starting_time_in_seconds, start_time, end_time):
    print("Loading audio file...")
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, offset=starting_time_in_seconds)
    print("Audio file loaded.")
    start_sample = int(start_time * SAMPLE_RATE)
    end_sample = int((end_time -0.01 )* SAMPLE_RATE)
    frequencies, magnitude = get_fft(audio[start_sample:end_sample])

    print(top_2_fundamental_frequencies(frequencies, magnitude))

    _, ax = plt.subplots(figsize=(10, 8))
    ax.plot(frequencies, magnitude)
    ax.set_title("Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, 1000)

    print("Window displayed.")
    plt.show()


def main(file_path, starting_time_in_seconds=0, duration_in_seconds=None, capo=0, max_tempo_bpm=MAX_TEMPO_BPM, sample_rate=SAMPLE_RATE):
    """Main function of the project"""

    print("Loading audio file...")
    audio, _ = librosa.load(file_path, sr=sample_rate, offset=starting_time_in_seconds, duration=duration_in_seconds)
    print("Audio file loaded.")
    model_output, _, _ = predict_from_array(audio)
    note_df = extract_note_attacks(model_output["onset"])
    note_df = apply_chord_filters(note_df)
    note_df = extract_attacks_duration(note_df, model_output["note"])

    # ___Getting tempo___
    tempo_bpm, confidence = estimate_tempo_from_attacks(note_df["onset_time"].unique())
    while tempo_bpm > max_tempo_bpm:
        tempo_bpm/= 2
    print(f"BPM found : {tempo_bpm} ({confidence:.2f})")

    # ___Positionning with correction of the notes___
    note_df  = assign_note_position(note_df, tempo_bpm)
    note_df = correct_notes_from_true_error(note_df)

    # ___Apply capo___
    note_df["partition_pitch"] = note_df["pitch"].apply(lambda p: p - capo)

    note_df = assign_voice(note_df)
    note_df = assign_note_quarter_length(note_df, tempo_bpm)

    # ___Analysis of the notes___
    print(note_df.columns.tolist())
    colonnes_utiles = ['onset_time', 'onset_frame', 'partition_pitch', 'pitch', 'confidence', 'measure_position', 'quarter_position_in_measure', 'quarter_length', 'duration', 'voice']
    pd.set_option('display.max_rows', None)
    print(note_df[colonnes_utiles])
    # view_tempo_drift(note_df)
    # analysis(model_output, note_df, tempo_bpm, start_time=18, end_time=25)

    # ___Prepare the voices for the score___
    melody_voice_in_measure = build_voice_in_measure(note_df, "melody")
    low_voice_in_measure = build_voice_in_measure(note_df, "low")

    score = create_music_sheet(melody_voice_in_measure, low_voice_in_measure, int(tempo_bpm))
    display_music_sheet(score)

    return


if __name__ == "__main__":
    try:
        file_path = "audio_files/Undertale.wav" # Basic and simple music
        main(file_path, starting_time_in_seconds=0, duration_in_seconds=49, capo=2)
        # file_path = "audio_files/La Boum.wav"
        # main(file_path, starting_time_in_seconds=5, duration_in_seconds=134)

        # note_fft(2.1,  0.95, 1.10  )
        # listen_audio(file_path,starting_time_in_seconds=20, duration_in_seconds=40, speed_factor = 0.7)
    except Exception as e:
        print(f"An error occurred: {e}")