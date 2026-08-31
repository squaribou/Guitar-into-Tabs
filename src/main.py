import librosa
import matplotlib.pyplot as plt

from constants import *
from basics import loading_audio_file, predict_from_array, extract_note_attacks
from frequence_analyser import get_fft, top_2_fundamental_frequencies
from audio_listener import listen_audio


def get_note_fft(starting_time_in_seconds, start_time, end_time):
    audio = loading_audio_file("audio_files/Howls moving castle (Merry-Go-Round of Life).mp3", starting_time_in_seconds)
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
    audio = loading_audio_file(file_path)
    model_output, midi_data, note_events = predict_from_array(audio)
    # note_events_sorted = sorted(note_events, key=lambda note: (note[0]))

    # Generate the complete partition base on the traitement by the model
    # for instrument in midi_data.instruments:
    #     instrument.program = 24  # 24 = Acoustic Guitar (nylon)
    #     instrument.name = "Guitar"
    # midi_data.write("sortie.mid")

    attacks = extract_note_attacks(model_output["onset"])

    for t, pitches in attacks:
        if t < start_time:
            continue
        if end_time is not None and t > end_time:
            break
        notes_str = ", ".join(f"{librosa.midi_to_note(int(midi))} ({conf:.2f})" for midi, conf in pitches)
        print(f"{t:.2f}s -> {notes_str}")

    # print("Window displayed.")
    # plt.imshow(model_output["onset"].T, aspect="auto", origin="lower")
    # plt.title("Probabilités d'onset (brutes, avant seuillage)")
    # plt.ylim(top=60, bottom=20)
    # plt.xlim(left=0, right=1500)
    # plt.show()


if __name__ == "__main__":
    file_path = "audio_files/Howls moving castle (Merry-Go-Round of Life).mp3"
    # note_fft(2.1,  0.95, 1.10  )
    analysis(file_path, start_time=20, end_time=40)
    # listen_audio(file_path,starting_time_in_seconds=20, duration_in_seconds=40, speed_factor = 0.7)