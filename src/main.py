import librosa
import matplotlib.pyplot as plt
import logging

logging.getLogger("root").setLevel(logging.ERROR)

from constants import *
from basics import predict_from_array, extract_note_attacks, filter_chord_notes, stupid_filter, estimate_tempo_from_attacks, assign_note_durations
from frequence_analyser import get_fft, top_2_fundamental_frequencies
from audio_listener import listen_audio
from music_sheet_writter import create_music_sheet, display_music_sheet


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


def analysis(file_path, start_time=0, end_time=None):
    print("Loading audio file...")
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    print("Audio file loaded.")
    model_output, _, _ = predict_from_array(audio)

    print("Prepare plotting...")
    start_frame = int(start_time / FRAME_TIME)
    end_frame = int(end_time / FRAME_TIME)

    _, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(model_output["onset"].T, aspect="auto", origin="lower")
    axes[0].set_title("Probabilités d'onset")
    axes[0].set_ylim(top=60, bottom=20)
    axes[0].set_xlim(left=start_frame, right=end_frame)

    axes[1].imshow(model_output["note"].T, aspect="auto", origin="lower")
    axes[1].set_title("Probabilités de note")
    axes[1].set_ylim(top=60, bottom=20)
    axes[1].set_xlim(left=start_frame, right=end_frame)

    # axes[2].imshow(model_output["contour"].T, aspect="auto", origin="lower")
    # axes[2].set_title("Probabilités de contour")
    # axes[2].set_ylim(top=60, bottom=20)
    # axes[2].set_xlim(left=start_frame, right=end_frame)

    print("Window displayed.")
    plt.show()


def processing_notes(file_path, starting_time_in_seconds=0, duration_in_seconds=None, capo=0):
    print("Loading audio file...")
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, offset=starting_time_in_seconds, duration=duration_in_seconds)
    print("Audio file loaded.")
    model_output, _, _ = predict_from_array(audio)

    attacks = extract_note_attacks(model_output["onset"])
    tempo_bpm, confidence = estimate_tempo_from_attacks([t for t, _ in attacks])
    while tempo_bpm > MAX_TEMPO_BPM:
        tempo_bpm/= 2
    print(tempo_bpm, confidence)
    attacks_with_duration = assign_note_durations(attacks, tempo_bpm)

    note_list = []
    for t, pitches, quaterlenght in attacks_with_duration:
        chords = filter_chord_notes(pitches)
        notes_str = ", ".join(f"{librosa.midi_to_note(int(midi))} ({confidence:.2f})" for midi, confidence in chords)
        # print(f"{t:.2f}s {quaterlenght}-> {notes_str}")

        chords_v2 = stupid_filter(chords)
        note_list.append(([int(midi) - capo for midi, _ in chords_v2], quaterlenght))

    music_sheet = create_music_sheet(note_list, int(tempo_bpm))
    display_music_sheet(music_sheet)
    return


if __name__ == "__main__":
    # file_path = "audio_files/Howls moving castle (Merry-Go-Round of Life).mp3"
    file_path = "audio_files/Undertale_fixed.wav"
    # note_fft(2.1,  0.95, 1.10  )
    processing_notes(file_path, starting_time_in_seconds=0, duration_in_seconds=49, capo=2)
    # analysis(file_path, start_time=5, end_time=30)
    # listen_audio(file_path,starting_time_in_seconds=20, duration_in_seconds=40, speed_factor = 0.7)
