import librosa
import matplotlib.pyplot as plt
import logging

logging.getLogger("root").setLevel(logging.ERROR)

from constants import SAMPLE_RATE, MAX_TEMPO_BPM, LOWEST_MELODY_NOTE
from basics import predict_from_array, analysis, extract_note_attacks, group_by_measure
from frequence_analyser import get_fft, top_2_fundamental_frequencies
from audio_listener import listen_audio
from music_sheet_writter import create_music_sheet, display_music_sheet
from tempo import view_tempo_drift, assign_note_position, assign_note_durations, estimate_tempo_from_attacks, correct_notes_from_true_error, extract_attacks_duration
from note_filter import filter_chord_notes, stupid_filter


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
    attacks = extract_note_attacks(model_output["onset"])


    # ___Basic filtering of harmonics and ghost notes___
    filtered_attacks = []
    for t, pitches in attacks:
        pitches = filter_chord_notes(pitches)
        pitches = stupid_filter(pitches)
        filtered_attacks.append((t, pitches))
    # print(filtered_attacks[:10])

    attacks_duration = extract_attacks_duration(filtered_attacks, model_output["note"])
    print(attacks_duration[1])

    # ___Getting tempo___
    tempo_bpm, confidence = estimate_tempo_from_attacks([t for t, _ in filtered_attacks])
    while tempo_bpm > max_tempo_bpm:
        tempo_bpm/= 2
    print(f"BPM found : {tempo_bpm} ({confidence:.2f})")


    # ___Positionning with correction of the notes___
    raw_positionned_notes  = assign_note_position(filtered_attacks, tempo_bpm)
    # view_tempo_drift(raw_positionned_notes)
    positionned_notes = correct_notes_from_true_error(raw_positionned_notes)
    # analysis(model_output, attacks, raw_positionned_notes, tempo_bpm, start_time=0, end_time=20)
    # analysis(model_output, attacks, positionned_notes, tempo_bpm, start_time=0, end_time=20)


    # ___Sepration between melodie voice and low voice___
    melody_voice = []
    low_voice = []
    for pitches, quarter_position_in_measure, measure_position, min_d in positionned_notes:
        note_basse = None
        for midi, _ in pitches:
            if midi < LOWEST_MELODY_NOTE:
                low_voice.append(([int(midi) - capo], quarter_position_in_measure, measure_position, min_d))
                note_basse = midi
                break
        if len(pitches) == 1 and pitches[0][0] == note_basse:
            continue
        melody_voice.append(([int(midi) - capo for midi, _ in pitches if midi != note_basse], quarter_position_in_measure, measure_position, min_d))

    # view_tempo_drift(low_voice)
    # view_tempo_drift(melody_voice)
    simplified_melody_voice = [(pitches, quarter_position_in_measure, measure_position) for pitches, quarter_position_in_measure, measure_position, _ in melody_voice]
    simplified_low_voice = [(pitches, quarter_position_in_measure, measure_position) for pitches, quarter_position_in_measure, measure_position, _ in low_voice]


    melody_voice_with_duration = assign_note_durations(simplified_melody_voice)
    low_voice_with_duration = assign_note_durations(simplified_low_voice)


    # ___Prepare the voices for the score___
    melody_voice_in_measure = group_by_measure(melody_voice_with_duration)
    low_voice_in_measure = group_by_measure(low_voice_with_duration)

    # print(melody_voice_in_measure[:3])
    # score = create_music_sheet(melody_voice_in_measure, low_voice_in_measure, int(tempo_bpm))
    # display_music_sheet(score)

    return


if __name__ == "__main__":
    try:
        # file_path = "audio_files/Undertale.wav" # Basic and simple music
        # main(file_path, starting_time_in_seconds=0, duration_in_seconds=49, capo=2)
        file_path = "audio_files/La Boum.wav"
        main(file_path, starting_time_in_seconds=5, duration_in_seconds=134)
        # note_fft(2.1,  0.95, 1.10  )
        
        # listen_audio(file_path,starting_time_in_seconds=20, duration_in_seconds=40, speed_factor = 0.7)
    except Exception as e:
        print(f"An error occurred: {e}")