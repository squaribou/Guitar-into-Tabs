# Guitar-into-Tabs

Convert guitar audio recordings into tablature and standard music notation.

## Installation

pip install -r requirements.txt

You'll also need [MuseScore Studio](https://musescore.org) installed to view generated scores.

## Usage

python src/main.py --input path/to/audio.mp3 (not in service)

## Pipeline

1. Load and preprocess the audio (librosa)
2. Detect note onsets and estimate pitch (basic-pitch)
3. Refine pitch estimation using pyin and spectral analysis
4. Convert detected notes into a music score (music21)
5. Export and display the score via MuseScore

## Basic music knowledge used here

- The tuning system used is equal temperament.
- Guitar music is written on a treble clef with octave transposition: notes sound one octave lower than written, to avoid excessive ledger lines below the staff.

## Technical notes

### Spectrum magnitude calculation

We compute the magnitude of the (complex-valued) FFT spectrum, keeping only positive frequencies. The magnitude is normalized by dividing by `N` (number of samples) and multiplying by `2`, to compensate for the discarded negative-frequency half of the spectrum (the FFT of a real signal is symmetric).

### Note attack detection

We use the `predict` function from the `basic-pitch` library, which returns a confidence matrix of shape `(n_frames, n_pitches)` representing the likelihood of a note onset at each time/pitch.

1. We reduce this matrix to a 1D onset-strength signal by taking the max confidence across pitches for each frame.
2. We detect peaks in this signal using `scipy.signal.find_peaks`, filtering by confidence threshold and minimum distance between peaks.

## Known limitations

- Some false-positive note attacks can still occur near strong transients.
- Currently tuned for monophonic playing.